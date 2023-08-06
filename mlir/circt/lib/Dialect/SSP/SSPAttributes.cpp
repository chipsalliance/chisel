//===- SSPAttributes.cpp - SSP attribute implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SSP (static scheduling problem) dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SSP/SSPAttributes.h"
#include "circt/Dialect/SSP/SSPDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace ssp;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SSP/SSPAttributes.cpp.inc"

void SSPDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/SSP/SSPAttributes.cpp.inc"
      >();
}

mlir::OptionalParseResult
ssp::parseOptionalPropertyArray(ArrayAttr &attr, AsmParser &parser,
                                ArrayRef<Attribute> alreadyParsed) {
  auto &builder = parser.getBuilder();

  if (parser.parseOptionalLSquare()) {
    if (!alreadyParsed.empty()) {
      attr = builder.getArrayAttr(alreadyParsed);
      return success();
    }
    return {};
  }

  SmallVector<Attribute> elements;
  elements.append(alreadyParsed.begin(), alreadyParsed.end());

  auto parseListResult = parser.parseCommaSeparatedList([&]() -> ParseResult {
    Attribute elem;

    // Try to parse a generic attribute.
    auto parseGenericAttrResult = parser.parseOptionalAttribute(elem);
    if (parseGenericAttrResult.has_value()) {
      if (failed(*parseGenericAttrResult))
        return failure();

      elements.push_back(elem);
      return success();
    }

    // Try to parse one of the built-in SSP property attributes.
    StringRef mnemonic;
    auto parseShortformAttrResult =
        generatedAttributeParser(parser, &mnemonic, Type(), elem);

    if (!parseShortformAttrResult.has_value()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "carries unknown shortform property: ")
             << mnemonic;
    }

    if (failed(*parseShortformAttrResult))
      return failure();

    elements.push_back(elem);
    return success();
  });

  if (parseListResult || parser.parseRSquare())
    return failure();

  attr = builder.getArrayAttr(elements);
  return success();
}

void ssp::printPropertyArray(ArrayAttr attr, AsmPrinter &p,
                             ArrayRef<Attribute> alreadyPrinted) {
  auto elementsToPrint =
      llvm::make_filter_range(attr.getAsRange<Attribute>(), [&](Attribute a) {
        return !llvm::is_contained(alreadyPrinted, a);
      });
  if (elementsToPrint.empty())
    return;

  p << '[';
  llvm::interleaveComma(elementsToPrint, p, [&](Attribute elem) {
    // Try to emit the shortform for the built-in SSP property attributes, and
    // if that fails, fall back to the generic form.
    if (failed(generatedAttributePrinter(elem, p)))
      p.printAttribute(attr);
  });
  p << ']';
}
