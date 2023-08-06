//===- HW.td - HW dialect definition -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HW_TD
#define CIRCT_DIALECT_HW_HW_TD

include "circt/Dialect/HW/HWDialect.td"
include "circt/Dialect/HW/HWAttributes.td"
include "circt/Dialect/HW/HWAttributesNaming.td"

include "circt/Dialect/HW/HWTypesImpl.td"
include "circt/Dialect/HW/HWTypes.td"

include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypeInterfaces.td"
include "circt/Dialect/HW/HWMiscOps.td"
include "circt/Dialect/HW/HWAggregates.td"
include "circt/Dialect/HW/HWStructure.td"
include "circt/Dialect/HW/HWTypeDecls.td"

#endif // CIRCT_DIALECT_HW_HW_TD
