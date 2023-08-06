//===- PassDetails.h - SSP pass class details --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the stuff shared between the different SSP passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the
// header guard on some systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_SSP_TRANSFORMS_PASSDETAILS_H
#define DIALECT_SSP_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/SSP/SSPAttributes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Dialect/SSP/SSPPasses.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Problems.h"

#include "mlir/IR/BuiltinOps.h"

namespace circt {
namespace ssp {

#define GEN_PASS_CLASSES
#include "circt/Dialect/SSP/SSPPasses.h.inc"

} // namespace ssp
} // namespace circt

#endif // DIALECT_SSP_TRANSFORMS_PASSDETAILS_H
