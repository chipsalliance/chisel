//===- PassDetails.h - Calyx pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the stuff shared between the different Calyx passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the
// header guard on some systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_CALYX_TRANSFORMS_PASSDETAILS_H
#define DIALECT_CALYX_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace calyx {

#define GEN_PASS_CLASSES
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"

} // namespace calyx
} // namespace circt

#endif // DIALECT_CALYX_TRANSFORMS_PASSDETAILS_H
