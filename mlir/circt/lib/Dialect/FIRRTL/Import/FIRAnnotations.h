//===- FIRAnnotations.h - .fir file annotation interface --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains private APIs the parser uses to load annotations.
//
//===----------------------------------------------------------------------===//

#ifndef FIRANNOTATIONS_H
#define FIRANNOTATIONS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
namespace json {
class Path;
class Value;
} // namespace json
} // namespace llvm

namespace mlir {
class ArrayAttr;
}

namespace circt {
namespace firrtl {

class CircuitOp;
class PrintFOp;

/// Convert a JSON value containing OMIR JSON (an array of OMNodes), convert
/// this to an OMIRAnnotation, and add it to a mutable `annotationMap` argument.
bool fromOMIRJSON(llvm::json::Value &value,
                  SmallVectorImpl<Attribute> &annotations,
                  llvm::json::Path path, MLIRContext *context);

bool fromJSONRaw(llvm::json::Value &value,
                 SmallVectorImpl<Attribute> &annotations, llvm::json::Path path,
                 MLIRContext *context);

ParseResult foldWhenEncodedVerifOp(PrintFOp printOp);

} // namespace firrtl
} // namespace circt

#endif // FIRANNOTATIONS_H
