//===- OMOps.h - Object Model operation declarations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model operation declarations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMOPS_H
#define CIRCT_DIALECT_OM_OMOPS_H

#include "circt/Dialect/OM/OMOpInterfaces.h"
#include "circt/Dialect/OM/OMTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "circt/Dialect/OM/OM.h.inc"

#endif // CIRCT_DIALECT_OM_OMOPS_H
