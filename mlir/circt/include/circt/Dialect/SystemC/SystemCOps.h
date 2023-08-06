//===- SystemCOps.h - Declare SystemC dialect operations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCOPS_H
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCOPS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InstanceImplementation.h"
#include "circt/Dialect/SystemC/SystemCAttributes.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "circt/Dialect/SystemC/SystemCOpInterfaces.h"
#include "circt/Dialect/SystemC/SystemCTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "circt/Dialect/SystemC/SystemC.h.inc"

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCOPS_H
