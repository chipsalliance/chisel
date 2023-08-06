//===- ExportSystemC.cpp - SystemC Emitter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SystemC emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Target/ExportSystemC.h"
#include "EmissionPrinter.h"
#include "RegisterAllEmitters.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include <regex>

using namespace circt;
using namespace circt::ExportSystemC;

#define DEBUG_TYPE "export-systemc"

/// Helper to convert a file-path to a macro name that can be used to guard a
/// header file.
static std::string pathToMacroName(StringRef path) {
  // Replace characters that represent a path hierarchy with underscore to match
  // the usual header guard formatting.
  auto str = std::regex_replace(path.upper(), std::regex("[\\\\./]"), "_");
  // Remove invalid characters. TODO: a digit is not allowed as the first
  // character, but not fixed here.
  return std::regex_replace(str, std::regex("[^a-zA-Z0-9_$]+"), "");
}

/// Emits the given operation to a file represented by the passed ostream and
/// file-path.
static LogicalResult emitFile(ArrayRef<Operation *> operations,
                              StringRef filePath, raw_ostream &os) {
  mlir::raw_indented_ostream ios(os);

  ios << "// " << filePath << "\n";
  std::string macroname = pathToMacroName(filePath);
  ios << "#ifndef " << macroname << "\n";
  ios << "#define " << macroname << "\n\n";

  bool failed = false;

  if (!operations.empty()) {
    OpEmissionPatternSet opPatterns;
    registerAllOpEmitters(opPatterns, operations[0]->getContext());
    TypeEmissionPatternSet typePatterns;
    registerAllTypeEmitters(typePatterns);
    AttrEmissionPatternSet attrPatterns;
    registerAllAttrEmitters(attrPatterns);
    EmissionPrinter printer(ios, opPatterns, typePatterns, attrPatterns,
                            operations[0]->getLoc());

    for (auto *op : operations)
      printer.emitOp(op);

    failed = printer.exitState().failed();
  }

  ios << "\n#endif // " << macroname << "\n\n";

  return failure(failed);
}

//===----------------------------------------------------------------------===//
// Unified and Split Emitter implementation
//===----------------------------------------------------------------------===//

LogicalResult ExportSystemC::exportSystemC(ModuleOp module,
                                           llvm::raw_ostream &os) {
  return emitFile({module}, "stdout.h", os);
}

LogicalResult ExportSystemC::exportSplitSystemC(ModuleOp module,
                                                StringRef directory) {
  // Collect all includes to emit them in every file.
  SmallVector<Operation *> includes;
  module->walk([&](mlir::emitc::IncludeOp op) { includes.push_back(op); });

  for (Operation &op : module.getRegion().front()) {
    if (auto symbolOp = dyn_cast<mlir::SymbolOpInterface>(op)) {
      // Create the output directory if needed.
      if (std::error_code error = llvm::sys::fs::create_directories(directory))
        return module.emitError("cannot create output directory \"")
               << directory << "\": " << error.message();

      // Open or create the output file.
      std::string fileName = symbolOp.getName().str() + ".h";
      SmallString<128> filePath(directory);
      llvm::sys::path::append(filePath, fileName);
      std::string errorMessage;
      auto output = mlir::openOutputFile(filePath, &errorMessage);
      if (!output)
        return module.emitError(errorMessage);

      // Emit the content to the file.
      SmallVector<Operation *> opsInThisFile(includes);
      opsInThisFile.push_back(symbolOp);
      if (failed(emitFile(opsInThisFile, filePath, output->os())))
        return symbolOp->emitError("failed to emit to file \"")
               << filePath << "\"";

      // Do not delete the file if emission was successful.
      output->keep();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// circt-translate registration
//===----------------------------------------------------------------------===//

void ExportSystemC::registerExportSystemCTranslation() {

  static llvm::cl::opt<std::string> directory(
      "export-dir", llvm::cl::desc("Directory path to write the files to."),
      llvm::cl::init("./"));

  static mlir::TranslateFromMLIRRegistration toSystemC(
      "export-systemc", "export SystemC",
      [](ModuleOp module, raw_ostream &output) {
        return ExportSystemC::exportSystemC(module, output);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<hw::HWDialect, comb::CombDialect,
                        systemc::SystemCDialect, mlir::emitc::EmitCDialect>();
      });

  static mlir::TranslateFromMLIRRegistration toSplitSystemC(
      "export-split-systemc", "export SystemC (split)",
      [](ModuleOp module, raw_ostream &output) {
        return ExportSystemC::exportSplitSystemC(module, directory);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<hw::HWDialect, comb::CombDialect,
                        systemc::SystemCDialect, mlir::emitc::EmitCDialect>();
      });
}
