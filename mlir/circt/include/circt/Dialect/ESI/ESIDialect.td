//===- ESIDialect.td - ESI dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_DIALECT_TD
#define CIRCT_DIALECT_ESI_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

def ESI_Dialect : Dialect {
  let name = "esi";
  let cppNamespace = "::circt::esi";

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register all ESI types.
    void registerTypes();
  }];
}

class ESI_Op<string mnemonic, list<Trait> traits = []> :
    Op<ESI_Dialect, mnemonic, traits>;

// Divide ESI ops into abstract and physical. Abstract ops are ones which
// need to be lowered to physical ops before they get lowered further (into
// HW). The abstract to physical lowering may not be fully constrained -- this
// is where decisions will have to be made concerning implementations. Physical
// ops are generally trivial to lower to HW -- they represent concrete
// constructs with no decisions left to be made.
class ESI_Abstract_Op<string mnemonic, list<Trait> traits = []> :
    ESI_Op<mnemonic, traits>;
class ESI_Physical_Op<string mnemonic, list<Trait> traits = []> :
    ESI_Op<mnemonic, traits>;

class ESI_Type<string name> : TypeDef<ESI_Dialect, name> { }

#endif // CIRCT_DIALECT_ESI_DIALECT_TD
