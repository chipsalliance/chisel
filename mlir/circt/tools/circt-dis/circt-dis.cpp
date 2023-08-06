//===- circt-dis.cpp - Convert MLIRBC to MLIR -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert MLIR bytecode (MLIRBC) input to MLIR textual format.
//
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "circt/Support/Version.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

#include <string>

using namespace llvm;
using namespace mlir;
using namespace circt;

static constexpr const char toolName[] = "circt-dis";
static cl::OptionCategory mainCategory("circt-dis Options");

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .mlirbc file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::value_desc("filename"),
                                           cl::cat(mainCategory));

/// Print error and return failure.
static LogicalResult emitError(const Twine &err) {
  WithColor::error(errs(), toolName) << err << "\n";
  return failure();
}

namespace {
/// Wrapper for OwningOpRef that leaks the module.
struct LeakModule {
  OwningOpRef<ModuleOp> module;
  ~LeakModule() { (void)module.release(); }
};
} // end anonymous namespace

static LogicalResult execute(MLIRContext &context) {
  // Figure out where we're writing the output.
  if (outputFilename.empty()) {
    StringRef input = inputFilename;
    if (input == "-")
      outputFilename = "-";
    else {
      input.consume_back(".mlirbc");
      outputFilename = (input + ".mlir").str();
    }
  }

  // Open output for writing, early error if problem.
  std::string err;
  auto output = openOutputFile(outputFilename, &err);
  if (!output)
    return emitError(err);

  // Read input MLIR bytecode.
  SourceMgr srcMgr;
  SourceMgrDiagnosticHandler handler(srcMgr, &context);

  LeakModule leakMod{
      parseSourceFile<ModuleOp>(inputFilename, srcMgr, &context)};
  auto &module = leakMod.module;
  if (!module)
    return failure();

  // Write MLIR.
  module->print(output->os());
  output->keep();

  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  mlir::DialectRegistry registry;

  circt::registerAllDialects(registry);

  // From circt-opt, register subset of MLIR dialects.
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::emitc::EmitCDialect>();

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions({&mainCategory, &llvm::getColorCategory()});

  registerAsmPrinterCLOptions();

  cl::ParseCommandLineOptions(argc, argv,
                              "CIRCT .mlirbc -> .mlir disassembler\n");

  MLIRContext context(registry);
  exit(failed(execute(context)));
}
