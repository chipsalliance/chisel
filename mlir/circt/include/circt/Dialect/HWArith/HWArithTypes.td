//===- HWArithTypes.td - HWArith data type definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data types for the HW arithmetic dialect.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HWArith/HWArithDialect.td"

// Type constraint to enforce signedness semantics
def HWArithIntegerType : DialectType<HWArithDialect,
    CPred<"isHWArithIntegerType($_self)">,
    "an arbitrary precision integer with signedness semantics",
    "::mlir::IntegerType">;
