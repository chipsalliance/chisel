//===- ExportChiselInterface.h - Chisel Interface Emitter -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the Chisel interface emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_EXPORTCHISELINTERFACE_H_
#define CIRCT_CONVERSION_EXPORTCHISELINTERFACE_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {

std::unique_ptr<mlir::Pass>
createExportChiselInterfacePass(llvm::raw_ostream &os);

std::unique_ptr<mlir::Pass>
createExportSplitChiselInterfacePass(mlir::StringRef outputDirectory = "./");

std::unique_ptr<mlir::Pass> createExportChiselInterfacePass();

} // namespace circt

#endif // CIRCT_CONVERSION_EXPORTCHISELINTERFACE_H_
