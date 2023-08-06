//===- CalyxEmitter.h - Calyx dialect to .futil emitter ---------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the .futil file emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXEMITTER_H
#define CIRCT_DIALECT_CALYX_CALYXEMITTER_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace circt {
namespace calyx {

mlir::LogicalResult exportCalyx(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerToCalyxTranslation();

} // namespace calyx
} // namespace circt

#endif // CIRCT_DIALECT_CALYX_CALYXEMITTER_H
