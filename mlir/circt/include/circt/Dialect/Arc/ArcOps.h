//===- ArcOps.h - Arc dialect operations ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCOPS_H
#define CIRCT_DIALECT_ARC_ARCOPS_H

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcTypes.h"

#include "circt/Dialect/Arc/ArcInterfaces.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Arc/Arc.h.inc"

#endif // CIRCT_DIALECT_ARC_ARCOPS_H
