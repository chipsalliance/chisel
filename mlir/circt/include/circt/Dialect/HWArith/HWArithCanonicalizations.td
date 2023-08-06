//===- HWArithCanonicalizations.td - HWArith canonicalization -*- tablegen ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Canonicalization patterns for the HWArith dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HWARITH_HWARITHCANONICALIZATIONS_TD
#define CIRCT_DIALECT_HWARITH_HWARITHCANONICALIZATIONS_TD


include "circt/Dialect/HWArith/HWArith.td"
include "mlir/IR/PatternBase.td"

def IdenticalTypePred : Constraint<CPred<"$0.getType() == $1.getType()">>;

def EliminateCast : Pat<(HWArith_CastOp:$out $in), (replaceWithValue $in),
    [(IdenticalTypePred $in, $out)]>;

#endif // CIRCT_DIALECT_HWARITH_HWARITHCANONICALIZATIONS_TD
