//===- Json.h - Json Utilities ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for JSON-to-Attribute conversion.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_JSON_H
#define CIRCT_SUPPORT_JSON_H

#include "circt/Support/LLVM.h"
#include "llvm/Support/JSON.h"

namespace circt {

/// Convert a simple attribute to JSON.
LogicalResult convertAttributeToJSON(llvm::json::OStream &json, Attribute attr);

/// Convert arbitrary JSON to an MLIR Attribute.
Attribute convertJSONToAttribute(MLIRContext *context, llvm::json::Value &value,
                                 llvm::json::Path p);

} // namespace circt

#endif // CIRCT_SUPPORT_JSON_H
