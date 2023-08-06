//===- PassDetails.h - FIRRTL pass class details ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different FIRRTL passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_FIRRTL_TRANSFORMS_PASSDETAILS_H
#define DIALECT_FIRRTL_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

namespace circt {

namespace hw {
class HWDialect;
}

namespace sv {
class SVDialect;
}

namespace firrtl {

#define GEN_PASS_CLASSES
#include "circt/Dialect/FIRRTL/Passes.h.inc"

} // namespace firrtl
} // namespace circt

#endif // DIALECT_FIRRTL_TRANSFORMS_PASSDETAILS_H
