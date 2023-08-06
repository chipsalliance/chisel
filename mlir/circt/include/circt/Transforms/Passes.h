//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for CIRCT transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSFORMS_PASSES_H
#define CIRCT_TRANSFORMS_PASSES_H

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <limits>

namespace circt {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createMapArithToCombPass();
std::unique_ptr<mlir::Pass> createFlattenMemRefPass();
std::unique_ptr<mlir::Pass> createFlattenMemRefCallsPass();
std::unique_ptr<mlir::Pass> createStripDebugInfoWithPredPass(
    const std::function<bool(mlir::Location)> &pred);

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

// Returns true if the provided memref is considered unidimensional (having a
// shape of size 1).
bool isUniDimensional(mlir::MemRefType memref);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Transforms/Passes.h.inc"

} // namespace circt

#endif // CIRCT_TRANSFORMS_PASSES_H
