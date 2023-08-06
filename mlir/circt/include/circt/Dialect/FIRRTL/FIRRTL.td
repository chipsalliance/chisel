//===- FIRRTL.td - FIRRTL dialect definition ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTL_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTL_TD

// Dialect
include "FIRRTLDialect.td"

// Operations
include "FIRRTLStructure.td"
include "FIRRTLDeclarations.td"
include "FIRRTLStatements.td"
include "FIRRTLExpressions.td"

// Types
include "FIRRTLTypes.td"
include "FIRRTLTypesImpl.td"

// Attributes
include "FIRRTLAttributes.td"

// Op Interfaces
include "FIRRTLOpInterfaces.td"

#endif // CIRCT_DIALECT_FIRRTL_FIRRTL_TD
