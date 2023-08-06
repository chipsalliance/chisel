//===- Json.cpp - Json Utilities --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/JSON.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

namespace json = llvm::json;

using namespace circt;
using mlir::UnitAttr;

// NOLINTBEGIN(misc-no-recursion)
LogicalResult circt::convertAttributeToJSON(llvm::json::OStream &json,
                                            Attribute attr) {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<DictionaryAttr>([&](auto attr) {
        json.objectBegin();
        for (auto subAttr : attr) {
          json.attributeBegin(subAttr.getName());
          if (failed(convertAttributeToJSON(json, subAttr.getValue())))
            return failure();
          json.attributeEnd();
        }
        json.objectEnd();
        return success();
      })
      .Case<ArrayAttr>([&](auto attr) {
        json.arrayBegin();
        for (auto subAttr : attr)
          if (failed(convertAttributeToJSON(json, subAttr)))
            return failure();
        json.arrayEnd();
        return success();
      })
      .Case<BoolAttr, StringAttr>([&](auto attr) {
        json.value(attr.getValue());
        return success();
      })
      .Case<IntegerAttr>([&](auto attr) -> LogicalResult {
        // If the integer can be accurately represented by a double, print
        // it as an integer. Otherwise, convert it to an exact decimal string.
        const auto &apint = attr.getValue();
        if (!apint.isSignedIntN(64))
          return failure();
        json.value(apint.getSExtValue());
        return success();
      })
      .Case<FloatAttr>([&](auto attr) -> LogicalResult {
        const auto &apfloat = attr.getValue();
        json.value(apfloat.convertToDouble());
        return success();
      })
      .Default([&](auto) -> LogicalResult { return failure(); });
}
// NOLINTEND(misc-no-recursion)

// NOLINTBEGIN(misc-no-recursion)
Attribute circt::convertJSONToAttribute(MLIRContext *context,
                                        json::Value &value, json::Path p) {
  // String or quoted JSON
  if (auto a = value.getAsString()) {
    // Test to see if this might be quoted JSON (a string that is actually
    // JSON).  Sometimes FIRRTL developers will do this to serialize objects
    // that the Scala FIRRTL Compiler doesn't know about.
    auto unquotedValue = json::parse(*a);
    auto err = unquotedValue.takeError();
    // If this parsed without an error and we didn't just unquote a number, then
    // it's more JSON and recurse on that.
    //
    // We intentionally do not want to unquote a number as, in JSON, the string
    // "0" is different from the number 0.  If we conflate these, then later
    // expectations about annotation structure may be broken.  I.e., an
    // annotation expecting a string may see a number.
    if (!err && !unquotedValue.get().getAsNumber())
      return convertJSONToAttribute(context, unquotedValue.get(), p);
    // If there was an error, then swallow it and handle this as a string.
    handleAllErrors(std::move(err), [&](const json::ParseError &a) {});
    return StringAttr::get(context, *a);
  }

  // Integer
  if (auto a = value.getAsInteger())
    return IntegerAttr::get(IntegerType::get(context, 64), *a);

  // Float
  if (auto a = value.getAsNumber())
    return FloatAttr::get(mlir::FloatType::getF64(context), *a);

  // Boolean
  if (auto a = value.getAsBoolean())
    return BoolAttr::get(context, *a);

  // Null
  if (auto a = value.getAsNull())
    return mlir::UnitAttr::get(context);

  // Object
  if (auto *a = value.getAsObject()) {
    NamedAttrList metadata;
    for (auto b : *a)
      metadata.append(
          b.first, convertJSONToAttribute(context, b.second, p.field(b.first)));
    return DictionaryAttr::get(context, metadata);
  }

  // Array
  if (auto *a = value.getAsArray()) {
    SmallVector<Attribute> metadata;
    for (size_t i = 0, e = (*a).size(); i != e; ++i)
      metadata.push_back(convertJSONToAttribute(context, (*a)[i], p.index(i)));
    return ArrayAttr::get(context, metadata);
  }

  llvm_unreachable("Impossible unhandled JSON type");
}
// NOLINTEND(misc-no-recursion)
