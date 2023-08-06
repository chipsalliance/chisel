//===- PassDetails.h - FSM pass class details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different FSM passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_FSM_TRANSFORMS_PASSDETAILS_H
#define DIALECT_FSM_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/FSM/FSMOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace fsm {

#define GEN_PASS_CLASSES
#include "circt/Dialect/FSM/Passes.h.inc"

} // namespace fsm
} // namespace circt

#endif // DIALECT_FSM_TRANSFORMS_PASSDETAILS_H
