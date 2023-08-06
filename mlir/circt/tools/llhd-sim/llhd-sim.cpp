//===- llhd-sim.cpp - LLHD simulator tool -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a command line tool to run LLHD simulation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LLHDToLLVM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/Simulator/Engine.h"
#include "circt/Dialect/LLHD/Simulator/Trace.h"
#include "circt/Support/Version.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::func;
using namespace circt;
using namespace circt::llhd::sim;

static cl::OptionCategory mainCategory("llhd-sim Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input-file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<int> nSteps("n", cl::desc("Set the maximum number of steps"),
                           cl::value_desc("max-steps"), cl::cat(mainCategory));

static cl::opt<uint64_t> maxTime(
    "T",
    cl::desc("Stop the simulation after the given amount of simulation time in "
             "picoseconds, including all sub-steps for that real-time step"),
    cl::value_desc("max-time"), cl::cat(mainCategory));

static cl::opt<bool>
    dumpLLVMDialect("dump-llvm-dialect",
                    cl::desc("Dump the LLVM IR dialect module"),
                    cl::cat(mainCategory));

static cl::opt<bool> dumpLLVMIR("dump-llvm-ir",
                                cl::desc("Dump the LLVM IR module"),
                                cl::cat(mainCategory));

static cl::opt<bool> dumpMLIR("dump-mlir",
                              cl::desc("Dump the original MLIR module"),
                              cl::cat(mainCategory));

static cl::opt<bool> dumpLayout("dump-layout",
                                cl::desc("Dump the gathered instance layout"),
                                cl::cat(mainCategory));

static cl::opt<std::string> root(
    "root",
    cl::desc("Specify the name of the entity to use as root of the design"),
    cl::value_desc("root_name"), cl::init("root"), cl::cat(mainCategory));
static cl::alias rootA("r", cl::desc("Alias for -root"), cl::aliasopt(root),
                       cl::cat(mainCategory));

enum OptLevel { O0, O1, O2, O3 };

static cl::opt<OptLevel>
    optimizationLevel(cl::desc("Choose optimization level:"), cl::init(O2),
                      cl::values(clEnumVal(O0, "Run passes and codegen at O0"),
                                 clEnumVal(O1, "Run passes and codegen at O1"),
                                 clEnumVal(O2, "Run passes and codegen at O2"),
                                 clEnumVal(O3, "Run passes and codegen at O3")),
                      cl::cat(mainCategory));

static cl::opt<TraceMode> traceMode(
    "trace-format", cl::desc("Choose the dump format:"),
    cl::init(TraceMode::Full),
    cl::values(
        clEnumValN(TraceMode::Full, "full",
                   "Dump signal changes for every time step and sub-step, "
                   "for all instances"),
        clEnumValN(TraceMode::Reduced, "reduced",
                   "Dump signal changes for every time-step and "
                   "sub-step, only for the top-level instance"),
        clEnumValN(TraceMode::Merged, "merged",
                   "Only dump changes for real-time steps, for all instances"),
        clEnumValN(TraceMode::MergedReduce, "merged-reduce",
                   "Only dump changes for real-time steps, only for the "
                   "top-level instance"),
        clEnumValN(
            TraceMode::NamedOnly, "named-only",
            "Only dump changes for real-time steps, only for top-level "
            "instance and signals not having the default name '(sig)?[0-9]*'"),
        clEnumValN(TraceMode::None, "none", "Don't dump a signal trace")),
    cl::cat(mainCategory));

static cl::list<std::string>
    sharedLibs("shared-libs",
               cl::desc("Libraries to link dynamically. Specify absolute path "
                        "to llhd-signals-runtime-wrappers for GCC or Windows. "
                        "Optional otherwise."),
               cl::ZeroOrMore, cl::MiscFlags::CommaSeparated,
               cl::cat(mainCategory));

static int dumpLLVM(ModuleOp module, MLIRContext &context) {
  if (dumpLLVMDialect) {
    module.dump();
    llvm::errs() << "\n";
    return 0;
  }

  // Translate the module, that contains the LLVM dialect, to LLVM IR.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  auto llvmTransformer =
      makeOptimizingTransformer(optimizationLevel, 0, nullptr);

  if (auto err = llvmTransformer(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }

  llvm::errs() << *llvmModule << "\n";
  return 0;
}

static LogicalResult applyMLIRPasses(ModuleOp module) {
  PassManager pm(module.getContext());

  pm.addPass(createConvertLLHDToLLVMPass());

  return pm.run(module);
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  // Hide default LLVM options, other than for this tool.
  cl::HideUnrelatedOptions(mainCategory);

  cl::ParseCommandLineOptions(argc, argv, "LLHD simulator\n");

  // Set up the input and output files.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Parse the input file.
  SourceMgr mgr;
  mgr.AddNewSourceBuffer(std::move(file), SMLoc());

  MLIRContext context;
  // Load the dialects
  context
      .loadDialect<llhd::LLHDDialect, LLVM::LLVMDialect, FuncDialect,
                   hw::HWDialect, comb::CombDialect, cf::ControlFlowDialect>();
  mlir::registerLLVMDialectTranslation(context);
  mlir::registerBuiltinDialectTranslation(context);

  mlir::OwningOpRef<mlir::ModuleOp> module(
      parseSourceFile<ModuleOp>(mgr, &context));

  if (dumpMLIR) {
    module->dump();
    llvm::errs() << "\n";
    return 0;
  }

  SmallVector<StringRef, 1> sharedLibPaths(sharedLibs.begin(),
                                           sharedLibs.end());

  llhd::sim::Engine engine(
      output->os(), *module, &applyMLIRPasses,
      makeOptimizingTransformer(optimizationLevel, 0, nullptr), root, traceMode,
      sharedLibPaths);

  if (dumpLLVMDialect || dumpLLVMIR) {
    return dumpLLVM(engine.getModule(), context);
  }

  if (dumpLayout) {
    engine.dumpStateLayout();
    engine.dumpStateSignalTriggers();
    return 0;
  }

  engine.simulate(nSteps, maxTime);

  output->keep();
  return 0;
}
