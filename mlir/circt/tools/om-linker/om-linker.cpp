//===- om-linker.cpp - An utility for linking objectmodel  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'om-linker', which links separated OM dialect IRs into a
// single IR.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Support/Version.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

static cl::OptionCategory mainCategory("om-linker Options");

static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input files>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    emitBytecode("emit-bytecode",
                 cl::desc("Emit bytecode when generating MLIR output"),
                 cl::init(false), cl::cat(mainCategory));

/// Check output stream before writing bytecode to it.
/// Warn and return true if output is known to be displayed.
static bool checkBytecodeOutputToConsole(raw_ostream &os) {
  if (os.is_displayed()) {
    errs() << "WARNING: You're attempting to print out a bytecode file.\n"
              "This is inadvisable as it may cause display problems. If\n"
              "you REALLY want to taste MLIR bytecode first-hand, you\n"
              "can force output with the `-f' option.\n\n";
    return true;
  }
  return false;
}

static cl::opt<bool> force("f", cl::desc("Enable binary output on terminals"),
                           cl::init(false), cl::cat(mainCategory));

/// Print the operation to the specified stream, emitting bytecode when
/// requested and politely avoiding dumping to terminal unless forced.
static LogicalResult printOp(Operation *op, raw_ostream &os) {
  if (emitBytecode && (force || !checkBytecodeOutputToConsole(os)))
    return writeBytecodeToFile(op, os,
                               mlir::BytecodeWriterConfig(getCirctVersion()));
  op->print(os);
  return success();
}

/// This implements the top-level logic for the om-linker command, invoked once
/// command line options are parsed and LLVM/MLIR are all set up and ready to
/// go.
static LogicalResult executeOMLinker(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  PassManager pm(&context);
  auto ts = tm.getRootScope();
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  // Set up the input files.
  struct ParsedInput {
    StringRef name;
    OwningOpRef<ModuleOp> mod;
    SourceMgr mgr;
  };

  struct LeakyInputs {
    SmallVector<ParsedInput> inputs;
    LeakyInputs(size_t n) { inputs.resize(n); }
    ~LeakyInputs() {
      // Leak
      for (auto &input : inputs)
        input.mod.release();
    };
  };

  auto numFiles = inputFilenames.size();
  LeakyInputs leakyInputs(numFiles);
  auto &srcs = leakyInputs.inputs;
  auto parserTimer = ts.nest("Parsing inputs");
  auto loadFile = [&](size_t i) -> LogicalResult {
    auto &s = srcs[i];
    s.name = inputFilenames[i];
    auto fileParseTimer = parserTimer.nest(s.name);
    s.mod = parseSourceFile<ModuleOp>(s.name, s.mgr, &context);
    if (!s.mod)
      return failure();
    // Use a file name (w/o extension) as a linker namespace.
    // e.g. "/tmp/work/foo.mlir" -> "foo"
    auto fileName = StringAttr::get(
        &context, llvm::sys::path::filename(s.name).split(".").first);
    s.mod.get()->setAttr("om.namespace", fileName);
    return success();
  };

  if (failed(failableParallelForEachN(&context, 0, numFiles, loadFile))) {
    errs() << "error reading inputs\n";
    return failure();
  }

  // This is the result module we are linking into.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));

  // Concat input modules without flatting ModuleOp. For example we construct
  // an IR like this:
  // ```
  // module {
  //   module {
  //     om.class @A(%arg: i1) {}
  //   }
  //   module {
  //     om.class.extern @A(%arg: i1) {}
  //     om.class @B(%arg: i2) {}
  //   }
  // }
  // ```
  // Actual linking will be performed by the subsequent pass pipeline.
  auto builder = OpBuilder::atBlockEnd(module.get().getBody());
  for (auto &in : srcs)
    builder.insert(in.mod.get());

  // Construct a linker pipeline.
  pm.addPass(om::createOMLinkModulesPass());
  if (failed(pm.run(module.get())))
    return failure();

  // Create the output file.
  std::string errorMessage;
  auto outputFile = openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    errs() << errorMessage << "\n";
    return failure();
  }

  // Dump output.
  if (failed(printOp(module.get(), outputFile->os())))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  outputFile->keep();

  // We intentionally "leak" the Module into the MLIRContext instead of
  // deallocating it.  There is no need to deallocate it right before process
  // exit.
  (void)module.release();
  return success();
}

/// Main driver for om-linker command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeOMLinker'.  This is set
/// up so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "OM linker\n");

  MLIRContext context;
  // Register OM dialect.
  context.loadDialect<om::OMDialect>();

  // Do the guts of the om-linker process.
  auto result = executeOMLinker(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
