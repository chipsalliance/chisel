//===- OM.td - Object Model dialect definition ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Object Model dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OM_TD
#define CIRCT_DIALECT_OM_OM_TD

include "mlir/IR/OpBase.td"

include "circt/Dialect/OM/OMDialect.td"

include "circt/Dialect/OM/OMAttributes.td"

include "circt/Dialect/OM/OMTypes.td"

include "circt/Dialect/OM/OMOps.td"

#endif // CIRCT_DIALECT_OM_OM_TD
