//===- ESI.td - ESI dialect top level tblgen ---------------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top level tablegen file for the ESI dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ESI_TD
#define ESI_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "circt/Dialect/ESI/ESIDialect.td"

include "circt/Dialect/ESI/ESIInterfaces.td"
include "circt/Dialect/ESI/ESITypes.td"
include "circt/Dialect/ESI/ESIChannels.td"
include "circt/Dialect/ESI/ESIServices.td"
include "circt/Dialect/ESI/ESIStdServices.td"
include "circt/Dialect/ESI/ESIStructure.td"

#endif // ESI_TD
