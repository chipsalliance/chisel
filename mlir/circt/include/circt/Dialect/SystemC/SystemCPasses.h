//===- SystemCPasses.h - SystemC pass entry points --------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCPASSES_H
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

// Forward declarations.
namespace mlir {
class RewritePatternSet;
class MLIRContext;
} // namespace mlir

namespace circt {
namespace systemc {

/// Populate the rewrite patterns for SystemC's instance-side interop lowerings.
void populateSystemCLowerInstanceInteropPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

std::unique_ptr<mlir::Pass> createSystemCLowerInstanceInteropPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/SystemC/Passes.h.inc"

} // namespace systemc
} // namespace circt

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCPASSES_H
