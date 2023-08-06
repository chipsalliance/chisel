//===- LTLOps.h - LTL dialect operations --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLOPS_H
#define CIRCT_DIALECT_LTL_LTLOPS_H

#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLTypes.h"

#define GET_OP_CLASSES
#include "circt/Dialect/LTL/LTL.h.inc"

#endif // CIRCT_DIALECT_LTL_LTLOPS_H
