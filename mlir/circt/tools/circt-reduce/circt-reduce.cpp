//===- circt-reduce.cpp - The circt-reduce driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-reduce' tool, which is the circt analog of
// mlir-reduce, used to drive test case reduction.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcReductions.h"
#include "circt/Dialect/FIRRTL/FIRRTLReductions.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWReductions.h"
#include "circt/InitAllDialects.h"
#include "circt/Reduce/GenericReductions.h"
#include "circt/Reduce/Tester.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "circt-reduce"
#define VERBOSE(X)                                                             \
  do {                                                                         \
    if (verbose) {                                                             \
      X;                                                                       \
    }                                                                          \
  } while (false)

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("Reduction Options");
static cl::OptionCategory granularityCategory("Granularity Control Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::Required, cl::cat(mainCategory));

static cl::opt<std::string>
    outputFilename("o", cl::init("-"),
                   cl::desc("Output filename for the reduced test case"),
                   cl::cat(mainCategory));

static cl::opt<bool>
    keepBest("keep-best", cl::init(true),
             cl::desc("Keep overwriting the output with better reductions"),
             cl::cat(mainCategory));

static cl::opt<bool>
    skipInitial("skip-initial", cl::init(false),
                cl::desc("Skip checking the initial input for interestingness"),
                cl::cat(mainCategory));

static cl::opt<bool> listReductions("list", cl::init(false),
                                    cl::desc("List all available reductions"),
                                    cl::cat(mainCategory));

static cl::list<std::string>
    includeReductions("include", cl::ZeroOrMore,
                      cl::desc("Only run a subset of the available reductions"),
                      cl::cat(mainCategory));

static cl::list<std::string>
    excludeReductions("exclude", cl::ZeroOrMore,
                      cl::desc("Do not run some of the available reductions"),
                      cl::cat(mainCategory));

static cl::opt<std::string> testerCommand(
    "test", cl::Required,
    cl::desc("A command or script to check if output is interesting"),
    cl::cat(mainCategory));

static cl::list<std::string>
    testerArgs("test-arg", cl::ZeroOrMore,
               cl::desc("Additional arguments to the test"),
               cl::cat(mainCategory));

static cl::opt<bool> verbose("v", cl::init(true),
                             cl::desc("Print reduction progress to stderr"),
                             cl::cat(mainCategory));

static cl::opt<unsigned>
    maxChunks("max-chunks", cl::init(0),
              cl::desc("Stop increasing granularity beyond this number of "
                       "chunks (granularity upper bound)"),
              cl::cat(granularityCategory));

static cl::opt<unsigned> minChunks(
    "min-chunks", cl::init(0),
    cl::desc(
        "Initial granularity in number of chunks (granularity lower bound)"),
    cl::cat(granularityCategory));

static cl::opt<unsigned>
    maxChunkSize("max-chunk-size", cl::init(0),
                 cl::desc("Initial granularity in number of ops per chunk "
                          "(granularity lower bound)"),
                 cl::cat(granularityCategory));

static cl::opt<unsigned>
    minChunkSize("min-chunk-size", cl::init(0),
                 cl::desc("Stop increasing granularity below this number of "
                          "ops per chunk (granularity upper bound)"),
                 cl::cat(granularityCategory));

static cl::opt<bool> testMustFail(
    "test-must-fail", cl::init(false),
    cl::desc("Consider an input to be interesting on non-zero exit status."),
    cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Tool Implementation
//===----------------------------------------------------------------------===//

/// Helper function that writes the current MLIR module to the configured output
/// file. Called for intermediate states if the `keepBest` options has been set,
/// or at least at the very end of the run.
static LogicalResult writeOutput(ModuleOp module) {
  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    mlir::emitError(UnknownLoc::get(module.getContext()),
                    "unable to open output file \"")
        << outputFilename << "\": " << errorMessage << "\n";
    return failure();
  }
  module.print(output->os());
  output->keep();
  return success();
}

/// Execute the main chunk of work of the tool. This function reads the input
/// module and iteratively applies the reduction strategies until no options
/// make it smaller.
static LogicalResult execute(MLIRContext &context) {
  std::string errorMessage;

  // Gather the sets of included and excluded reductions.
  llvm::DenseSet<StringRef> inclusionSet(includeReductions.begin(),
                                         includeReductions.end());
  llvm::DenseSet<StringRef> exclusionSet(excludeReductions.begin(),
                                         excludeReductions.end());

  // Parse the input file.
  VERBOSE(llvm::errs() << "Reading input\n");
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseSourceFile<ModuleOp>(inputFilename, &context);
  if (!module)
    return failure();

  // Gather a list of reduction patterns that we should try.
  ReducePatternSet patterns;
  populateGenericReducePatterns(&context, patterns);
  ReducePatternInterfaceCollection reducePatternCollection(&context);
  reducePatternCollection.populateReducePatterns(patterns);
  auto reductionFilter = [&](const Reduction &reduction) {
    auto name = reduction.getName();
    return (inclusionSet.empty() || inclusionSet.count(name)) &&
           !exclusionSet.count(name);
  };
  patterns.filter(reductionFilter);
  patterns.sortByBenefit();

  // Print the list of patterns.
  if (listReductions) {
    for (unsigned i = 0; i < patterns.size(); ++i)
      llvm::outs() << patterns[i].getName() << "\n";
    return success();
  }

  // Evaluate the unreduced input.
  VERBOSE({
    llvm::errs() << "Testing input with `" << testerCommand << "`\n";
    for (auto &arg : testerArgs)
      llvm::errs() << "  with argument `" << arg << "`\n";
  });
  Tester tester(testerCommand, testerArgs, testMustFail);
  auto initialTest = tester.get(module.get());
  if (!skipInitial && !initialTest.isInteresting()) {
    mlir::emitError(UnknownLoc::get(&context), "input is not interesting");
    return failure();
  }
  auto bestSize = initialTest.getSize();
  VERBOSE(llvm::errs() << "Initial module has size " << bestSize << "\n");

  // Mechanism to write over the previous summary line, if it was the last
  // thing written to errs.
  size_t errsPosAfterLastSummary = 0;
  auto clearSummary = [&] {
    if (llvm::errs().tell() != errsPosAfterLastSummary)
      return;
    llvm::errs()
        << "\x1B[1A\x1B[2K"; // move up one line ("1A"), clear line ("2K")
  };

  // Iteratively reduce the input module by applying the current reduction
  // pattern to successively smaller subsets of the operations until we find one
  // that retains the interesting behavior.
  // ModuleExternalizer pattern;
  BitVector appliedOneShotPatterns(patterns.size(), false);
  for (unsigned patternIdx = 0; patternIdx < patterns.size();) {
    auto &pattern = patterns[patternIdx];
    if (pattern.isOneShot() && appliedOneShotPatterns[patternIdx]) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping one-shot `" << pattern.getName() << "`\n");
      ++patternIdx;
      continue;
    }
    VERBOSE({
      clearSummary();
      llvm::errs() << "Trying reduction `" << pattern.getName() << "`\n";
    });
    size_t rangeBase = 0;
    size_t rangeLength = -1;
    bool patternDidReduce = false;
    bool allDidReduce = true;

    while (rangeLength > 0) {
      // Limit the number of ops processed at once to the value requested by the
      // user.
      if (maxChunkSize > 0)
        rangeLength = std::min<size_t>(rangeLength, maxChunkSize);

      // Apply the pattern to the subset of operations selected by `rangeBase`
      // and `rangeLength`.
      size_t opIdx = 0;
      mlir::OwningOpRef<mlir::ModuleOp> newModule = module->clone();
      pattern.beforeReduction(*newModule);
      SmallVector<std::pair<Operation *, uint64_t>, 16> opBenefits;
      SmallDenseSet<Operation *> opsTouched;
      pattern.notifyOpErasedCallback = [&](Operation *op) {
        opsTouched.insert(op);
      };
      newModule->walk([&](Operation *op) {
        uint64_t benefit = pattern.match(op);
        if (benefit > 0) {
          opIdx++;
          opBenefits.push_back(std::make_pair(op, benefit));
        }
      });
      std::sort(opBenefits.begin(), opBenefits.end(),
                [](auto a, auto b) { return a.second > b.second; });
      for (size_t idx = rangeBase, num = 0;
           num < rangeLength && idx < opBenefits.size(); ++idx) {
        auto *op = opBenefits[idx].first;
        if (opsTouched.contains(op))
          continue;
        if (pattern.match(op)) {
          op->walk([&](Operation *subop) { opsTouched.insert(subop); });
          (void)pattern.rewrite(op);
          ++num;
        }
      }
      pattern.afterReduction(*newModule);
      pattern.notifyOpErasedCallback = nullptr;
      if (opIdx == 0) {
        VERBOSE({
          clearSummary();
          llvm::errs() << "- No more ops where the pattern applies\n";
        });
        break;
      }

      // Reduce the chunk size to achieve the minimum number of chunks requested
      // by the user.
      if (minChunks > 0)
        rangeLength = std::min<size_t>(rangeLength,
                                       std::max<size_t>(opIdx / minChunks, 1));

      // Show some progress indication.
      VERBOSE({
        size_t boundLength = std::min(rangeLength, opIdx);
        size_t numDone = rangeBase / boundLength + 1;
        size_t numTotal = (opIdx + boundLength - 1) / boundLength;
        clearSummary();
        llvm::errs() << "  [" << numDone << "/" << numTotal << "; "
                     << (numDone * 100 / numTotal) << "%; " << opIdx << " ops, "
                     << boundLength << " at once; " << pattern.getName()
                     << "]\n";
        errsPosAfterLastSummary = llvm::errs().tell();
      });

      // Check if this reduced module is still interesting, and its overall size
      // is smaller than what we had before.
      auto shouldAccept = [&](TestCase &test) {
        if (!test.isValid())
          return false; // don't write to disk if module is busted
        if (test.getSize() >= bestSize && !pattern.acceptSizeIncrease())
          return false; // don't run test if size already bad
        return test.isInteresting();
      };
      auto test = tester.get(newModule.get());
      if (shouldAccept(test)) {
        // Make this reduced module the new baseline and reset our search
        // strategy to start again from the beginning, since this reduction may
        // have created additional opportunities.
        patternDidReduce = true;
        bestSize = test.getSize();
        VERBOSE({
          clearSummary();
          llvm::errs() << "- Accepting module of size " << bestSize << "\n";
        });
        module = std::move(newModule);

        // We leave `rangeBase` and `rangeLength` untouched in this case. This
        // causes the next iteration of the loop to try the same pattern again
        // at the same offset. If the pattern has reached a fixed point, nothing
        // changes and we proceed. If the pattern has removed an operation, this
        // will already operate on the next batch of operations which have
        // likely moved to this point. The only exception are operations that
        // are marked as "one shot", which explicitly ask to not be re-applied
        // at the same location.
        if (pattern.isOneShot())
          rangeBase += rangeLength;

        // Write the current state to disk if the user asked for it.
        if (keepBest)
          if (failed(writeOutput(module.get())))
            return failure();
      } else {
        allDidReduce = false;
        // Try the pattern on the next `rangeLength` number of operations.
        rangeBase += rangeLength;
      }

      // If we have gone past the end of the input, reduce the size of the chunk
      // of operations we're reducing and start again from the top.
      if (rangeBase >= opIdx) {
        // If this is a one-shot pattern and it applied everywhere there's no
        // need to try again at reduced chunk size. Simply move forward in that
        // case.
        if (pattern.isOneShot() && allDidReduce) {
          rangeLength = 0;
          rangeBase = 0;
        } else {
          rangeLength = std::min(rangeLength, opIdx) / 2;
          rangeBase = 0;

          // Stop increasing granularity if the number of ops processed at once
          // has fallen below the lower limit set by the user.
          if (rangeLength < minChunkSize)
            rangeLength = 0;

          // Stop increasing granularity if the number of chunks has increased
          // beyond the upper limit set by the user.
          if (rangeLength > 0 && maxChunks > 0 &&
              (opIdx + rangeLength - 1) / rangeLength > maxChunks)
            rangeLength = 0;

          if (rangeLength > 0) {
            VERBOSE({
              clearSummary();
              llvm::errs() << "- Trying " << rangeLength << " ops at once\n";
            });
          }
        }
      }
    }

    // If this was a one-shot pattern, mark it as having been applied. This will
    // prevent further reapplication.
    if (pattern.isOneShot())
      appliedOneShotPatterns.set(patternIdx);

    // If the pattern provided a successful reduction, restart with the first
    // pattern again, since we might have uncovered additional reduction
    // opportunities. Otherwise we just keep going to try the next pattern.
    if (patternDidReduce && patternIdx > 0) {
      VERBOSE({
        clearSummary();
        llvm::errs() << "- Reduction `" << pattern.getName()
                     << "` was successful, starting at the top\n\n";
      });
      patternIdx = 0;
    } else {
      ++patternIdx;
    }
  }

  // Write the reduced test case to the output.
  clearSummary();
  VERBOSE(llvm::errs() << "All reduction strategies exhausted\n");
  VERBOSE(llvm::errs() << "Final size: " << bestSize << " ("
                       << (100 - bestSize * 100 / initialTest.getSize())
                       << "% reduction)\n");
  return writeOutput(module.get());
}

/// The entry point for the `circt-reduce` tool. Configures and parses the
/// command line options, registers all dialects with a context, and calls the
/// `execute` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  // Register and hide default LLVM options, other than for this tool.
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();
  cl::HideUnrelatedOptions({&mainCategory, &granularityCategory});

  // Parse the command line options provided by the user.
  cl::ParseCommandLineOptions(argc, argv, "CIRCT test case reduction tool\n");

  // Register all the dialects and create a context to work wtih.
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<func::FuncDialect, scf::SCFDialect, LLVM::LLVMDialect>();
  arc::registerReducePatternDialectInterface(registry);
  firrtl::registerReducePatternDialectInterface(registry);
  hw::registerReducePatternDialectInterface(registry);
  mlir::MLIRContext context(registry);

  // Do the actual processing and use `exit` to avoid the slow teardown of the
  // context.
  exit(failed(execute(context)));
}
