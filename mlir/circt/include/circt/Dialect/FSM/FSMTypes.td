//===- FSMTypes.td - Definition of FSM dialect types ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_FSMTYPES_TD
#define CIRCT_DIALECT_FSM_FSMTYPES_TD

def InstanceType : FSMType<"Instance"> {
  let summary = "An FSM instance type";
  let description = [{
    Represents an FSM instance.
  }];
  let mnemonic = "instance";
  let parameters = (ins);
}

#endif // CIRCT_DIALECT_FSM_FSMTYPES_TD
