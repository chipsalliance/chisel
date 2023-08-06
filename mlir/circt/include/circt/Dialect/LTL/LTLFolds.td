//===- LTLFolds.td - LTL dialect declarative rewrites ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LTL_LTLFOLDS_TD
#define CIRCT_DIALECT_LTL_LTLFOLDS_TD

include "circt/Dialect/LTL/LTLOps.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

def valueHead : NativeCodeCall<"$0[0]">;
def valueTail : NativeCodeCall<"$0.drop_front()">;
def concatValues : NativeCodeCall<
  "concatValues(ValueRange{$0}, ValueRange{$1})">;

//===----------------------------------------------------------------------===//
// DelayOp
//===----------------------------------------------------------------------===//

// delay(delay(s, I), J) -> delay(s, I+J)
// delay(delay(s, I), J, N) -> delay(s, I+J)
// delay(delay(s, I, N), J) -> delay(s, I+J)
// delay(delay(s, I, N), J, M) -> delay(s, I+J, N+M)
def mergedDelays : NativeCodeCall<
  "$_builder.getI64IntegerAttr($0.getInt() + $1.getInt())">;
def mergedLengths : NativeCodeCall<[{
  $0 && $1 ? $_builder.getI64IntegerAttr($0.getInt() + $1.getInt())
           : IntegerAttr{}
}]>;
def NestedDelays : Pat<
  (DelayOp (DelayOp $input, $delay1, $length1), $delay2, $length2),
  (DelayOp $input, (mergedDelays $delay1, $delay2),
                   (mergedLengths $length1, $length2))>;

// delay(concat(s, ...), N, M) -> concat(delay(s, N, M), ...)
def MoveDelayIntoConcat : Pat<
  (DelayOp (ConcatOp $inputs), $delay, $length),
  (ConcatOp (concatValues (DelayOp (valueHead $inputs), $delay, $length),
                          (valueTail $inputs)))>;

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// concat(..., concat(s0, s1), ...) -> concat(..., s0, s1, ...)
def AnyDefiningOpIsConcat : Constraint<CPred<[{
    llvm::any_of($0, [](auto v){
      return !!v.template getDefiningOp<ConcatOp>();
    })
  }]>,
  "any value is defined by ConcatOp">;
def flattenConcats : NativeCodeCall<"flattenConcats($0)">;
def FlattenConcats : Pat<
  (ConcatOp $inputs), (ConcatOp (flattenConcats $inputs)),
  [(AnyDefiningOpIsConcat $inputs)]>;

#endif // CIRCT_DIALECT_LTL_LTLFOLDS_TD
