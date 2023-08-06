//===- HandshakeDialect.h - Handshake dialect declaration -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Handshake MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_HANDSHAKEDIALECT_H
#define CIRCT_DIALECT_HANDSHAKE_HANDSHAKEDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

// Pull in the Dialect definition.
#include "circt/Dialect/Handshake/HandshakeDialect.h.inc"

// Pull in all enum type definitions, attributes,
// and utility function declarations.
#include "circt/Dialect/Handshake/HandshakeEnums.h.inc"

#endif // CIRCT_DIALECT_HANDSHAKE_HANDSHAKEDIALECT_H
