//===- ParsingUtils.h - CIRCT parsing common functions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to help with parsing.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PARSINGUTILS_H
#define CIRCT_SUPPORT_PARSINGUTILS_H

#include "mlir/IR/BuiltinAttributes.h"

#include "circt/Support/LLVM.h"

namespace circt {
namespace parsing_util {

/// Get a name from an SSA value string, if said value name is not a
/// number.
static StringAttr getNameFromSSA(MLIRContext *context, StringRef name) {
  if (!name.empty()) {
    // Ignore numeric names like %42
    assert(name.size() > 1 && name[0] == '%' && "Unknown MLIR name");
    if (isdigit(name[1]))
      name = StringRef();
    else
      name = name.drop_front();
  }
  return StringAttr::get(context, name);
}

} // namespace parsing_util
} // namespace circt

#endif // CIRCT_SUPPORT_PARSINGUTILS_H
