//===- SSPAttributes.h - SSP attribute definitions --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SSP (static scheduling problem) dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SSP_SSPATTRIBUTES_H
#define CIRCT_DIALECT_SSP_SSPATTRIBUTES_H

#include "circt/Dialect/SSP/SSPDialect.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SSP/SSPAttributes.h.inc"

namespace circt {
namespace ssp {

/// Parse an array of attributes while recognizing the properties of the SSP
/// dialect even without a `#ssp.` prefix. Any attributes supplied in \p
/// alreadyParsed are prepended to the parsed ones.
mlir::OptionalParseResult
parseOptionalPropertyArray(ArrayAttr &attr, AsmParser &parser,
                           ArrayRef<Attribute> alreadyParsed = {});

/// Print an array attribute, suppressing the `#ssp.` prefix for properties
/// defined in the SSP dialect. Attributes mentioned in \p alreadyPrinted are
/// skipped.
void printPropertyArray(ArrayAttr attr, AsmPrinter &p,
                        ArrayRef<Attribute> alreadyPrinted = {});

} // namespace ssp
} // namespace circt

#endif // CIRCT_DIALECT_SSP_SSPATTRIBUTES_H
