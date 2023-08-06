//===- DCTypes.td - DC dialect types -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DC_TYPES_TD
#define CIRCT_DIALECT_DC_TYPES_TD

include "circt/Dialect/DC/DCDialect.td"
include "mlir/IR/AttrTypeBase.td"

class DCTypeDef<string name> : TypeDef<DCDialect, name> { }

def TokenType : DCTypeDef<"Token"> {
  let mnemonic = "token";
  let parameters = (ins );
  let assemblyFormat = "";
  let description = [{
    A `!dc.token`-typed value represents a control value with handshake semantics.
  }];
}

// A value type is a value which is wrapped with token semantics.
def ValueType : DCTypeDef<"Value"> {
  let mnemonic = "value";
  let parameters = (ins "Type":$innerType);
  let assemblyFormat = "`<`$innerType`>`";
  let description = [{
    A `!dc.value`-typed value represents a value which is wrapped with
    `!dc.token` semantics. This type is used to attach control semantics to
    values. The inner value may be of any type.
  }];
}

def I1ValueType : Type<
  CPred<"circt::dc::isI1ValueType($_self)">, "must be a !dc.value<i1> type">,
  BuildableType<"$_builder.getType<dc::ValueType>($_builder.getI1Type())"> {
}

def ValueOrTokenType : Type<
  CPred<"$_self.isa<dc::ValueType, dc::TokenType>()">,
  "must be a !dc.value or !dc.token type"> {
}



#endif // CIRCT_DIALECT_DC_TYPES_TD
