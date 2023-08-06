//===- Ibis.td - Definition of Ibis dialect -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_TD
#define CIRCT_DIALECT_IBIS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"

def IbisDialect : Dialect {
  let name = "ibis";
  let cppNamespace = "::circt::ibis";

  let summary = "Types and operations for Ibis dialect";
  let description = [{
    The `ibis` dialect is intended to support porting and eventual open sourcing
    of an internal hardware development language.
  }];

  // TODO: uncomment this once we introduce some types. The default functions
  // aren't generated without typedefs.
  // let useDefaultTypePrinterParser = 1;
}

// Base class for the types in this dialect.
class IbisType<string name> : TypeDef<IbisDialect, name> {}

// Base class for the operations in this dialect.
class IbisOp<string mnemonic, list<Trait> traits = []> :
    Op<IbisDialect, mnemonic, traits>;

include "circt/Dialect/Ibis/IbisTypes.td"
include "circt/Dialect/Ibis/IbisOps.td"

#endif // CIRCT_DIALECT_IBIS_TD
