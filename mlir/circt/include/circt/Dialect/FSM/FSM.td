//===- FSM.td - Definition of FSM dialect ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_TD
#define CIRCT_DIALECT_FSM_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"

def FSMDialect : Dialect {
  let name = "fsm";
  let cppNamespace = "::circt::fsm";

  let summary = "Types and operations for FSM dialect";
  let description = [{
    This dialect defines the `fsm` dialect, which is intended to represent
    finite-state machines.
  }];

  let useDefaultTypePrinterParser = 1;
}

// Base class for the types in this dialect.
class FSMType<string name> : TypeDef<FSMDialect, name> {}

// Base class for the operations in this dialect.
class FSMOp<string mnemonic, list<Trait> traits = []> :
    Op<FSMDialect, mnemonic, traits>;

include "circt/Dialect/FSM/FSMTypes.td"
include "circt/Dialect/FSM/FSMOps.td"

#endif // CIRCT_DIALECT_FSM_TD
