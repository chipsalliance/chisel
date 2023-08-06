//===- Passes.h - Ibis pass entry points -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_IBISPASSES_H
#define CIRCT_DIALECT_IBIS_IBISPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace circt {
namespace ibis {

std::unique_ptr<Pass> createCallPrepPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Ibis/IbisPasses.h.inc"

} // namespace ibis
} // namespace circt

#endif // CIRCT_DIALECT_IBIS_IBISPASSES_H
