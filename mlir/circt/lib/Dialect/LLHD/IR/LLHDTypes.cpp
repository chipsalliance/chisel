//===- LLHDTypes.cpp - LLHD types and attributes code defs ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for LLHD data types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt::llhd;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/LLHD/IR/LLHDTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/LLHD/IR/LLHDAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// Time Attribute
//===----------------------------------------------------------------------===//

/// Parse a time attribute.
/// Syntax: timeattr ::= #llhd.time<[time][timeUnit], [delta]d, [epsilon]e>
Attribute TimeAttr::parse(AsmParser &p, Type type) {
  llvm::StringRef timeUnit;
  unsigned time = 0;
  unsigned delta = 0;
  unsigned eps = 0;

  // parse the time value
  if (p.parseLess() || p.parseInteger(time) || p.parseKeyword(&timeUnit))
    return {};

  // parse the delta step value
  if (p.parseComma() || p.parseInteger(delta) || p.parseKeyword("d"))
    return {};

  // parse the epsilon value
  if (p.parseComma() || p.parseInteger(eps) || p.parseKeyword("e") ||
      p.parseGreater())
    return {};

  // return a new instance of time attribute
  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  return getChecked(mlir::detail::getDefaultDiagnosticEmitFn(loc),
                    p.getContext(), time, timeUnit, delta, eps);
}

void TimeAttr::print(AsmPrinter &p) const {
  p << "<" << getTime() << getTimeUnit() << ", " << getDelta() << "d, "
    << getEpsilon() << "e>";
}

LogicalResult TimeAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               TimeType type, unsigned time,
                               llvm::StringRef timeUnit, unsigned delta,
                               unsigned epsilon) {
  // Check the time unit is a legal SI unit
  std::vector<std::string> legalUnits{"ys", "zs", "as", "fs", "ps",
                                      "ns", "us", "ms", "s"};
  if (std::find(legalUnits.begin(), legalUnits.end(), timeUnit) ==
      legalUnits.end())
    return emitError() << "Illegal time unit.";

  return success();
}

//===----------------------------------------------------------------------===//
// Register attributes and types to the LLHD Dialect
//===----------------------------------------------------------------------===//

void LLHDDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/LLHD/IR/LLHDTypes.cpp.inc"
      >();
}

void LLHDDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/LLHD/IR/LLHDAttributes.cpp.inc"
      >();
}
