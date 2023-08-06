//===- PassDetails.h - Comb pass class details ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_COMB_TRANSFORMS_PASSDETAILS_H
#define DIALECT_COMB_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace comb {

#define GEN_PASS_CLASSES
#include "circt/Dialect/Comb/Passes.h.inc"

} // namespace comb
} // namespace circt

#endif // DIALECT_COMB_TRANSFORMS_PASSDETAILS_H
