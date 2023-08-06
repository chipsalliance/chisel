//===- PassDetails.h - SystemC pass class details ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different SystemC passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_SYSTEMC_TRANSFORMS_PASSDETAILS_H
#define DIALECT_SYSTEMC_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Interop/InteropDialect.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace systemc {

#define GEN_PASS_CLASSES
#include "circt/Dialect/SystemC/Passes.h.inc"

} // namespace systemc
} // namespace circt

#endif // DIALECT_SYSTEMC_TRANSFORMS_PASSDETAILS_H
