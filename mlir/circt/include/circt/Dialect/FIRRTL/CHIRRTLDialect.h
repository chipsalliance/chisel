//===- CHIRRTLDialect.h - CHIRRTL dialect declaration ----------*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an MLIR dialect for the CHIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_CHIRRTLDIALECT_H
#define CIRCT_DIALECT_FIRRTL_CHIRRTLDIALECT_H

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Pull in the dialect definition.
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h.inc"

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/FIRRTL/CHIRRTLTypes.h.inc"

// Include generated ops.
#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/CHIRRTL.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_CHIRRTLDIALECT_H
