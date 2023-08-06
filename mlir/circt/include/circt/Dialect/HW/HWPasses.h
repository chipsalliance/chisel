//===- Passes.h - HW pass entry points --------------------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_HW_HWPASSES_H
#define CIRCT_DIALECT_HW_HWPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace circt {
namespace hw {

std::unique_ptr<mlir::Pass> createPrintInstanceGraphPass();
std::unique_ptr<mlir::Pass> createHWSpecializePass();
std::unique_ptr<mlir::Pass> createPrintHWModuleGraphPass();
std::unique_ptr<mlir::Pass> createFlattenIOPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/HW/Passes.h.inc"

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_HWPASSES_H
