//===- MSFTDialect.h - Microsoft dialect declaration ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MSFT MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_MSFTDIALECT_H
#define CIRCT_DIALECT_MSFT_MSFTDIALECT_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

#include "llvm/Support/ManagedStatic.h"

#include <functional>

#include "circt/Dialect/MSFT/MSFTDialect.h.inc"
#include "circt/Dialect/MSFT/MSFTEnums.h.inc"

#endif // CIRCT_DIALECT_MSFT_MSFTDIALECT_H
