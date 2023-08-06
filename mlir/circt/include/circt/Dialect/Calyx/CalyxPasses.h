//===- CalyxPasses.h - Calyx pass entry points ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXPASSES_H
#define CIRCT_DIALECT_CALYX_CALYXPASSES_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {
namespace calyx {

std::unique_ptr<mlir::Pass> createCompileControlPass();
std::unique_ptr<mlir::Pass> createGoInsertionPass();
std::unique_ptr<mlir::Pass> createRemoveCombGroupsPass();
std::unique_ptr<mlir::Pass> createRemoveGroupsPass();
std::unique_ptr<mlir::Pass> createClkInsertionPass();
std::unique_ptr<mlir::Pass> createResetInsertionPass();
std::unique_ptr<mlir::Pass> createGroupInvariantCodeMotionPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"

} // namespace calyx
} // namespace circt

#endif // CIRCT_DIALECT_CALYX_CALYXPASSES_H
