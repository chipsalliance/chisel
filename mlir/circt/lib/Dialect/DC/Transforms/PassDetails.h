//===- PassDetails.h - DC pass class details --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the stuff shared between the different DC passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the
// header guard on some systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_DC_TRANSFORMS_PASSDETAILS_H
#define DIALECT_DC_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace dc {

#define GEN_PASS_CLASSES
#include "circt/Dialect/DC/DCPasses.h.inc"

} // namespace dc
} // namespace circt

#endif // DIALECT_DC_TRANSFORMS_PASSDETAILS_H
