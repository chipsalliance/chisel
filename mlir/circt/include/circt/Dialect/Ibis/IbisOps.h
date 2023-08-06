//===- IbisOps.h - Definition of Ibis dialect ops ----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_IBISOPS_H
#define CIRCT_DIALECT_IBIS_IBISOPS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "circt/Dialect/Ibis/IbisInterfaces.h.inc"

namespace circt {
namespace ibis {
class ContainerOp;
} // namespace ibis
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/Ibis/Ibis.h.inc"

#endif // CIRCT_DIALECT_IBIS_IBISOPS_H
