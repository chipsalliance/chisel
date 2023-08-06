//===- PassDetails.h - LLHD pass class details ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LLHD_TRANSFORMS_PASSDETAILS_H
#define DIALECT_LLHD_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace llhd {

#define GEN_PASS_CLASSES
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"

} // namespace llhd
} // namespace circt

#endif // DIALECT_LLHD_TRANSFORMS_PASSDETAILS_H
