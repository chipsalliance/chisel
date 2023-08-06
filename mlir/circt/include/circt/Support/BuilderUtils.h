//===- BuilderUtils.h - Operation builder utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_BUILDERUTILS_H
#define CIRCT_SUPPORT_BUILDERUTILS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {

/// A helper union that can represent a `StringAttr`, `StringRef`, or `Twine`.
/// It is intended to be used as arguments to an op's `build` function. This
/// allows a single builder to accept any flavor value for a string attribute.
/// The `get` function can then be used to obtain a `StringAttr` from any of the
/// possible variants `StringAttrOrRef` can take.
class StringAttrOrRef {
  using Value = llvm::PointerUnion<StringAttr, StringRef *, Twine *>;
  Value value;

public:
  StringAttrOrRef() : value() {}
  StringAttrOrRef(StringAttr attr) : value(attr) {}
  StringAttrOrRef(const StringRef &str)
      : value(const_cast<StringRef *>(&str)) {}
  StringAttrOrRef(const Twine &twine) : value(const_cast<Twine *>(&twine)) {}

  /// Return the represented string as a `StringAttr`.
  StringAttr get(MLIRContext *context) const {
    return TypeSwitch<Value, StringAttr>(value)
        .Case<StringAttr>([&](auto value) { return value; })
        .Case<StringRef *, Twine *>(
            [&](auto value) { return StringAttr::get(context, *value); })
        .Default([](auto) { return StringAttr{}; });
  }
};

} // namespace circt

#endif // CIRCT_SUPPORT_BUILDERUTILS_H
