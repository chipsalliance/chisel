//===- Passes.h - FSM pass entry points -------------------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_FSM_FSMPASSES_H
#define CIRCT_DIALECT_FSM_FSMPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace circt {
namespace fsm {

std::unique_ptr<mlir::Pass> createPrintFSMGraphPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/FSM/Passes.h.inc"

} // namespace fsm
} // namespace circt

#endif // CIRCT_DIALECT_FSM_FSMPASSES_H
