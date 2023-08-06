//===- MSFTAttributes.cpp - Implement MSFT dialect attributes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSFT dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/MSFT/MSFTAttributes.cpp.inc"

Attribute PhysLocationAttr::parse(AsmParser &p, Type type) {
  llvm::SMLoc loc = p.getCurrentLocation();
  std::string subPath;
  StringRef devTypeStr;
  uint64_t x, y, num;

  if (p.parseLess() || p.parseKeyword(&devTypeStr) || p.parseComma() ||
      p.parseInteger(x) || p.parseComma() || p.parseInteger(y) ||
      p.parseComma() || p.parseInteger(num) || p.parseGreater())
    return Attribute();

  std::optional<PrimitiveType> devType = symbolizePrimitiveType(devTypeStr);
  if (!devType) {
    p.emitError(loc, "Unknown device type '" + devTypeStr + "'");
    return Attribute();
  }
  PrimitiveTypeAttr devTypeAttr =
      PrimitiveTypeAttr::get(p.getContext(), *devType);
  auto phy = PhysLocationAttr::get(p.getContext(), devTypeAttr, x, y, num);
  return phy;
}

void PhysLocationAttr::print(AsmPrinter &p) const {
  p << "<" << stringifyPrimitiveType(getPrimitiveType().getValue()) << ", "
    << getX() << ", " << getY() << ", " << getNum() << '>';
}

Attribute PhysicalBoundsAttr::parse(AsmParser &p, Type type) {
  uint64_t xMin, xMax, yMin, yMax;
  if (p.parseLess() || p.parseKeyword("x") || p.parseColon() ||
      p.parseLSquare() || p.parseInteger(xMin) || p.parseComma() ||
      p.parseInteger(xMax) || p.parseRSquare() || p.parseComma() ||
      p.parseKeyword("y") || p.parseColon() || p.parseLSquare() ||
      p.parseInteger(yMin) || p.parseComma() || p.parseInteger(yMax) ||
      p.parseRSquare() || p.parseGreater()) {
    llvm::SMLoc loc = p.getCurrentLocation();
    p.emitError(loc, "unable to parse PhysicalBounds");
    return Attribute();
  }

  return PhysicalBoundsAttr::get(p.getContext(), xMin, xMax, yMin, yMax);
}

void PhysicalBoundsAttr::print(AsmPrinter &p) const {
  p << "<";
  p << "x: [" << getXMin() << ", " << getXMax() << "], ";
  p << "y: [" << getYMin() << ", " << getYMax() << ']';
  p << '>';
}

LogicalResult LocationVectorAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, TypeAttr type,
    ArrayRef<PhysLocationAttr> locs) {
  int64_t typeBitWidth = hw::getBitWidth(type.getValue());
  if (typeBitWidth < 0)
    return emitError() << "cannot compute bit width of type '" << type << "'";
  if ((uint64_t)typeBitWidth != locs.size())
    return emitError() << "must specify " << typeBitWidth << " locations";
  return success();
}

LogicalResult
circt::msft::parseOptionalRegLoc(SmallVectorImpl<PhysLocationAttr> &locs,
                                 AsmParser &p) {
  MLIRContext *ctxt = p.getContext();
  if (!p.parseOptionalStar()) {
    locs.push_back({});
    return success();
  }

  PhysLocationAttr loc;
  if (p.parseOptionalAttribute(loc).has_value()) {
    locs.push_back(loc);
    return success();
  }

  uint64_t x, y, n;
  if (p.parseLess() || p.parseInteger(x) || p.parseComma() ||
      p.parseInteger(y) || p.parseComma() || p.parseInteger(n) ||
      p.parseGreater())
    return failure();
  locs.push_back(PhysLocationAttr::get(
      ctxt, PrimitiveTypeAttr::get(ctxt, PrimitiveType::FF), x, y, n));
  return success();
}

void circt::msft::printOptionalRegLoc(PhysLocationAttr loc, AsmPrinter &p) {
  if (loc && loc.getPrimitiveType().getValue() == PrimitiveType::FF)
    p << '<' << loc.getX() << ", " << loc.getY() << ", " << loc.getNum() << '>';
  else if (loc)
    p << loc;
  else
    p << "*";
}

Attribute LocationVectorAttr::parse(AsmParser &p, Type) {
  MLIRContext *ctxt = p.getContext();
  TypeAttr type;
  SmallVector<PhysLocationAttr, 32> locs;

  if (p.parseLess() || p.parseAttribute(type) || p.parseComma() ||
      p.parseLSquare() || p.parseCommaSeparatedList([&]() {
        return parseOptionalRegLoc(locs, p);
      }) ||
      p.parseRSquare() || p.parseGreater())
    return {};

  return LocationVectorAttr::getChecked(p.getEncodedSourceLoc(p.getNameLoc()),
                                        ctxt, type, locs);
}

void LocationVectorAttr::print(AsmPrinter &p) const {
  p << '<' << getType() << ", [";
  llvm::interleaveComma(getLocs(), p, [&p](PhysLocationAttr loc) {
    printOptionalRegLoc(loc, p);
  });
  p << "]>";
}

void MSFTDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/MSFT/MSFTAttributes.cpp.inc"
      >();
}
