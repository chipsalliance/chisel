//===- HWArithOps.h - Definition HWArith dialect types ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HWARITH_HWARITHTYPES_H
#define CIRCT_DIALECT_HWARITH_HWARITHTYPES_H

#include "circt/Support/LLVM.h"

#include "HWArithDialect.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/HWArith/HWArithTypes.h.inc"

namespace circt {
namespace hwarith {

// Check whether a specified type satisfies the constraints for the
// HWArithIntegerType
bool isHWArithIntegerType(::mlir::Type type);

} // namespace hwarith
} // namespace circt

#endif
