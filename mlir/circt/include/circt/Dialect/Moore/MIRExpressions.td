//===- MIRExpressions.td - Moore MIR expression ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for Moore MIR expressions.
//
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/InferTypeOpInterface.td"


def ConstantOp : MIROp<"constant", [Pure]> {
  let summary = "A constant value";

  let arguments = (ins I32Attr:$value);
  let results = (outs MooreRValueType:$result);
  let assemblyFormat = "$value attr-dict `:` qualified(type($result))";
}

def ConcatOp : MIROp<"concat", [
    Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "A concatenation of expressions";
  let description = [{
    This operation represents the SystemVerilog concatenation expression
    `{x, y, z}`. See IEEE 1800-2017 §11.4.12 "Concatenation operators".

    All operands must be simple bit vector types.

    The concatenation result is a simple bit vector type. The result is unsigned
    regardless of the sign of the operands (see concatenation-specific rules in
    IEEE 1800-2017 §11.8.1 "Rules for expression types"). The size of the result
    is the sum of the sizes of all operands. If any of the operands is
    four-valued, the result is four-valued; otherwise it is two-valued.
  }];
  let arguments = (ins Variadic<SimpleBitVectorType>:$values);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    $values attr-dict `:` functional-type($values, $result)
  }];
}

//===----------------------------------------------------------------------===//
// Shift operations
//===----------------------------------------------------------------------===//

class ShiftOp<string name> : MIROp<name, [
    Pure, 
    TypesMatchWith<"value and result types must match",
                   "value", "result", "$_self">
]> {
  let arguments = (ins SimpleBitVectorType:$value,
                       SimpleBitVectorType:$amount,
                       UnitAttr:$arithmetic);
  let results = (outs SimpleBitVectorType:$result);
  let assemblyFormat = [{
    ( `arithmetic` $arithmetic^ )? $value `,` $amount attr-dict
    `:` type($value) `,` type($amount)
  }];
}

def ShlOp : ShiftOp<"shl"> {
  let summary = "A logical or arithmetic left-shift expression";
  let description = [{
    This operation represents the SystemVerilog logical and arithmetic
    left-shift expressions `<<` and `<<<`.
    See IEEE 1800-2017 §11.4.10 "Shift operators".

    The value to be shifted and the amount must be simple bit vector types.
    The shift result is of the same type as the input value.

    The logical and arithmetic shift both insert zeros in place of the shifted
    bits. 
  }];
}

def ShrOp : ShiftOp<"shr"> {
  let summary = "A logical or arithmetic right-shift expression";
  let description = [{
    This operation represents the SystemVerilog logical and arithmetic
    right-shift expressions `>>` and `>>>`.
    See IEEE 1800-2017 §11.4.10 "Shift operators".

    The value to be shifted and the amount must be simple bit vector types.
    The shift result is of the same type as the input value.

    The logical shift always inserts zeros in place of the shifted bits.
    The arithmetic shift inserts zeros if the result type is unsigned or the 
    MSB (sign bit) if the result type is signed.
  }];
}
