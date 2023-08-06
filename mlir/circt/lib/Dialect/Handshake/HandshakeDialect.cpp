//===- HandshakeDialect.cpp - Implement the Handshake dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::handshake;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void HandshakeDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Handshake/Handshake.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Handshake/HandshakeAttributes.cpp.inc"
      >();
}

// Provide implementations for the enums, attributes and interfaces that we use.
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Handshake/HandshakeAttributes.cpp.inc"
#include "circt/Dialect/Handshake/HandshakeDialect.cpp.inc"
#include "circt/Dialect/Handshake/HandshakeEnums.cpp.inc"
#include "circt/Dialect/Handshake/HandshakeInterfaces.cpp.inc"
