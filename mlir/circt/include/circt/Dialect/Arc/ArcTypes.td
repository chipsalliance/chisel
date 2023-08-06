//===- ArcTypes.td - Arc dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCTYPES_TD
#define CIRCT_DIALECT_ARC_ARCTYPES_TD

include "circt/Dialect/Arc/ArcDialect.td"
include "mlir/IR/AttrTypeBase.td"

class ArcTypeDef<string name> : TypeDef<ArcDialect, name> { }

def StateType : ArcTypeDef<"State"> {
  let mnemonic = "state";
  let parameters = (ins "::mlir::IntegerType":$type);
  let assemblyFormat = "`<` $type `>`";
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::IntegerType":$type), [{
      return $_get(type.getContext(), type);
    }]>
  ];
}

def MemoryType : ArcTypeDef<"Memory"> {
  let mnemonic = "memory";
  let parameters = (ins "unsigned":$numWords,
                        "::mlir::IntegerType":$wordType,
                        "::mlir::IntegerType":$addressType);
  let assemblyFormat = "`<` $numWords `x` $wordType `,` $addressType `>`";

  let extraClassDeclaration = [{
    unsigned getStride();
  }];
}

def StorageType : ArcTypeDef<"Storage"> {
  let mnemonic = "storage";
  let parameters = (ins OptionalParameter<"unsigned">:$size);
  let assemblyFormat = "(`<` $size^ `>`)?";
}

#endif // CIRCT_DIALECT_ARC_ARCTYPES_TD
