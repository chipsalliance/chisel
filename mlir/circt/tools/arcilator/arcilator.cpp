//===- arcilator.cpp - An experimental circuit simulator ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'arcilator' compiler, which converts HW designs into
// a corresponding LLVM-based software model.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToArith.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcInterfaces.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Support/Version.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <iostream>
#include <optional>

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command Line Arguments
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("arcilator Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool> observePorts("observe-ports",
                                  cl::desc("Make all ports observable"),
                                  cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> observeWires("observe-wires",
                                  cl::desc("Make all wires observable"),
                                  cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    observeNamedValues("observe-named-values",
                       cl::desc("Make values with `sv.namehint` observable"),
                       cl::init(false), cl::cat(mainCategory));

static cl::opt<std::string> stateFile("state-file", cl::desc("State file"),
                                      cl::value_desc("filename"), cl::init(""),
                                      cl::cat(mainCategory));

static cl::opt<bool> shouldInline("inline", cl::desc("Inline arcs"),
                                  cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> shouldDedup("dedup", cl::desc("Deduplicate arcs"),
                                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    shouldMakeLUTs("lookup-tables",
                   cl::desc("Optimize arcs into lookup tables"), cl::init(true),
                   cl::cat(mainCategory));

static cl::opt<bool> printDebugInfo("print-debug-info",
                                    cl::desc("Print debug information"),
                                    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false), cl::Hidden, cl::cat(mainCategory));

// Options to control early-out from pipeline.
enum Until {
  UntilPreprocessing,
  UntilArcConversion,
  UntilArcOpt,
  UntilStateLowering,
  UntilStateAlloc,
  UntilLLVMLowering,
  UntilEnd
};
static auto runUntilValues = cl::values(
    clEnumValN(UntilPreprocessing, "preproc", "Input preprocessing"),
    clEnumValN(UntilArcConversion, "arc-conv", "Conversion of modules to arcs"),
    clEnumValN(UntilArcOpt, "arc-opt", "Arc optimizations"),
    clEnumValN(UntilStateLowering, "state-lowering", "Stateful arc lowering"),
    clEnumValN(UntilStateAlloc, "state-alloc", "State allocation"),
    clEnumValN(UntilLLVMLowering, "llvm-lowering", "Lowering to LLVM"),
    clEnumValN(UntilEnd, "all", "Run entire pipeline (default)"));
static cl::opt<Until>
    runUntilBefore("until-before",
                   cl::desc("Stop pipeline before a specified point"),
                   runUntilValues, cl::init(UntilEnd), cl::cat(mainCategory));
static cl::opt<Until>
    runUntilAfter("until-after",
                  cl::desc("Stop pipeline after a specified point"),
                  runUntilValues, cl::init(UntilEnd), cl::cat(mainCategory));

// Options to control the output format.
enum OutputFormat { OutputMLIR, OutputLLVM, OutputDisabled };
static cl::opt<OutputFormat> outputFormat(
    cl::desc("Specify output format"),
    cl::values(clEnumValN(OutputMLIR, "emit-mlir", "Emit MLIR dialects"),
               clEnumValN(OutputLLVM, "emit-llvm", "Emit LLVM"),
               clEnumValN(OutputDisabled, "disable-output",
                          "Do not output anything")),
    cl::init(OutputLLVM), cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Main Tool Logic
//===----------------------------------------------------------------------===//

/// Populate a pass manager with the arc simulator pipeline for the given
/// command line options.
static void populatePipeline(PassManager &pm) {
  auto untilReached = [](Until until) {
    return until >= runUntilBefore || until > runUntilAfter;
  };

  // Pre-process the input such that it no longer contains any SV dialect ops
  // and external modules that are relevant to the arc transformation are
  // represented as intrinsic ops.
  if (untilReached(UntilPreprocessing))
    return;
  pm.addPass(seq::createLowerFirMemPass());
  pm.addPass(
      arc::createAddTapsPass(observePorts, observeWires, observeNamedValues));
  pm.addPass(arc::createStripSVPass());
  pm.addPass(arc::createInferMemoriesPass(observePorts));
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Restructure the input from a `hw.module` hierarchy to a collection of arcs.
  if (untilReached(UntilArcConversion))
    return;
  pm.addPass(createConvertToArcsPass());
  if (shouldDedup)
    pm.addPass(arc::createDedupPass());
  pm.addPass(arc::createInlineModulesPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Perform arc-level optimizations that are not specific to software
  // simulation.
  if (untilReached(UntilArcOpt))
    return;
  pm.addPass(arc::createSplitLoopsPass());
  if (shouldDedup)
    pm.addPass(arc::createDedupPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
  if (shouldMakeLUTs)
    pm.addPass(arc::createMakeTablesPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // TODO: the following is commented out because the backend does not support
  // StateOp resets yet.
  // pm.addPass(arc::createInferStatePropertiesPass());
  // InferStateProperties does not remove all ops it bypasses and inserts a lot
  // of constant ops that should be uniqued
  // pm.addPass(createSimpleCanonicalizerPass());
  // Now some arguments may be unused because reset conditions are not passed as
  // inputs anymore pm.addPass(arc::createRemoveUnusedArcArgumentsPass());
  // Because we replace a lot of StateOp inputs with constants in the enable
  // patterns we may be able to sink a lot of them
  // TODO: maybe merge RemoveUnusedArcArguments with SinkInputs?
  // pm.addPass(arc::createSinkInputsPass());
  // pm.addPass(createCSEPass());
  // pm.addPass(createSimpleCanonicalizerPass());
  // Removing some muxes etc. may lead to additional dedup opportunities
  // if (shouldDedup)
  // pm.addPass(arc::createDedupPass());

  // Lower stateful arcs into explicit state reads and writes.
  if (untilReached(UntilStateLowering))
    return;
  pm.addPass(arc::createLowerStatePass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // TODO: LowerClocksToFuncsPass might not properly consider scf.if operations
  // (or nested regions in general) and thus errors out when muxes are also
  // converted in the hw.module or arc.model
  // TODO: InlineArcs seems to not properly handle scf.if operations, thus the
  // following is commented out
  // pm.addPass(arc::createMuxToControlFlowPass());

  if (shouldInline) {
    pm.addPass(arc::createInlineArcsPass());
    pm.addPass(arc::createArcCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  pm.addPass(arc::createGroupResetsAndEnablesPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Allocate states.
  if (untilReached(UntilStateAlloc))
    return;
  pm.addPass(arc::createLegalizeStateUpdatePass());
  pm.nest<arc::ModelOp>().addPass(arc::createAllocateStatePass());
  if (!stateFile.empty())
    pm.addPass(arc::createPrintStateInfoPass(stateFile));
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Lower the arcs and update functions to LLVM.
  if (untilReached(UntilLLVMLowering))
    return;
  pm.addPass(arc::createLowerClocksToFuncsPass());
  pm.addPass(createConvertCombToArithPass());
  pm.addPass(createLowerArcToLLVMPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
}

static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  {
    auto parserTimer = ts.nest("Parse MLIR input");
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  }
  if (!module)
    return failure();

  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  populatePipeline(pm);

  if (printDebugInfo && outputFormat == OutputLLVM)
    pm.nest<LLVM::LLVMFuncOp>().addPass(LLVM::createDIScopeForLLVMFuncOpPass());

  if (failed(pm.run(module.get())))
    return failure();

  // Handle MLIR output.
  if (runUntilBefore != UntilEnd || runUntilAfter != UntilEnd ||
      outputFormat == OutputMLIR) {
    OpPrintingFlags printingFlags;
    // Only set the debug info flag to true in order to not overwrite MLIR
    // printer CLI flags when the custom debug info option is not set.
    if (printDebugInfo)
      printingFlags.enableDebugInfo(printDebugInfo);
    auto outputTimer = ts.nest("Print MLIR output");
    module->print(outputFile.value()->os(), printingFlags);
    return success();
  }

  // Handle LLVM output.
  if (outputFormat == OutputLLVM) {
    auto outputTimer = ts.nest("Print LLVM output");
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule)
      return failure();
    llvmModule->print(outputFile.value()->os(), nullptr);
    return success();
  }

  return success();
}

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult processInputSplit(
    MLIRContext &context, TimingScope &ts,
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return processBuffer(context, ts, sourceMgr, outputFile);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, ts, sourceMgr, outputFile);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> input,
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (!splitInputFile)
    return processInputSplit(context, ts, std::move(input), outputFile);

  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer), outputFile);
      },
      llvm::outs());
}

static LogicalResult executeArcilator(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Register our dialects.
  DialectRegistry registry;
  registry.insert<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                  sv::SVDialect, arc::ArcDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::func::FuncDialect,
                  mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect>();

  arc::initAllExternalInterfaces(registry);

  mlir::func::registerInlinerExtension(registry);

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    outputFile.value()->keep();

  return success();
}

/// Main driver for the command. This sets up LLVM and MLIR, and parses command
/// line options before passing off to 'executeArcilator'. This is set up so we
/// can `exit(0)` at the end of the program to avoid teardown of the MLIRContext
/// and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register passes before parsing command-line options, so that they are
  // available for use with options like `--mlir-print-ir-before`.
  {
    // MLIR transforms:
    // Don't use registerTransformsPasses, pulls in too much.
    registerCSEPass();
    registerCanonicalizerPass();
    registerStripDebugInfoPass();

    // Dialect passes:
    arc::registerPasses();
  }

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR-based circuit simulator\n");

  MLIRContext context;
  auto result = executeArcilator(context);

  // Use "exit" instead of returning to signal completion. This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
