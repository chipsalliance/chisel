//===- MSFTAttributes.h - Microsoft dialect attributes ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSFT dialect custom attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_MSFTATTRIBUTES_H
#define CIRCT_DIALECT_MSFT_MSFTATTRIBUTES_H

#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/MSFT/MSFTAttributes.h.inc"

namespace circt {
namespace msft {

/// Parse and append a PhysLocAttr. Options are '*' for null location, <x, y,
/// num> for a location which is implicitily a FF, or a full phys location
/// attribute.
LogicalResult parseOptionalRegLoc(SmallVectorImpl<PhysLocationAttr> &locs,
                                  AsmParser &p);
/// Print out the above.
void printOptionalRegLoc(PhysLocationAttr loc, AsmPrinter &p);

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_MSFTATTRIBUTES_H
