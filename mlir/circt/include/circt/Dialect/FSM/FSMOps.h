//===- FSMOps.h - Definition of FSM dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_FSMOPS_H
#define CIRCT_DIALECT_FSM_FSMOPS_H

#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/FSM/FSMTypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define GET_OP_CLASSES
#include "circt/Dialect/FSM/FSM.h.inc"

#endif // CIRCT_DIALECT_FSM_FSMOPS_H
