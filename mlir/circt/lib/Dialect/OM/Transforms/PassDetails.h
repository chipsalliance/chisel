//===- PassDetails.h - OM pass class details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different Seq passes.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_OM_TRANSFORMS_PASSDETAILS_H
#define DIALECT_OM_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/OM/OMOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace om {

#define GEN_PASS_CLASSES
#include "circt/Dialect/OM/OMPasses.h.inc"

} // namespace om
} // namespace circt

#endif // DIALECT_SEQ_TRANSFORMS_PASSDETAILS_H
