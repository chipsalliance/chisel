//===- SVDialect.h - SV dialect declaration ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an SV MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_DIALECT_H
#define CIRCT_DIALECT_SV_DIALECT_H

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringSet.h"

namespace circt {
namespace sv {

/// Given string \p origName, generate a new name if it conflicts with any
/// keyword or any other name in the map \p nextGeneratedNameIDs. Use the value
/// of \p nextGeneratedNameIDs as a counter for suffix. Update the \p
/// nextGeneratedNameIDs with the generated name and return the StringRef.
llvm::StringRef
resolveKeywordConflict(llvm::StringRef origName,
                       llvm::StringMap<size_t> &nextGeneratedNameIDs);

/// Legalize the specified name for use in SV output. Auto-uniquifies the name
/// through \c resolveKeywordConflict if required. If the name is empty, a
/// unique temp name is created.
StringRef legalizeName(llvm::StringRef name,
                       llvm::StringMap<size_t> &nextGeneratedNameIDs);

/// Check if a name is valid for use in SV output by only containing characters
/// allowed in SV identifiers.
///
/// Call \c legalizeName() to obtain a legal version of the name.
bool isNameValid(llvm::StringRef name);

} // namespace sv
} // namespace circt

// Pull in the dialect definition.
#include "circt/Dialect/SV/SVDialect.h.inc"

#endif // CIRCT_DIALECT_SV_DIALECT_H
