//===- FIRRTLAttributes.h - FIRRTL dialect attributes -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the FIRRTL dialect custom attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_H

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// PortDirections
//===----------------------------------------------------------------------===//

/// This represents the direction of a single port.
enum class Direction { In, Out };

/// Prints the Direction to the stream as either "in" or "out".
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Direction &dir);

namespace direction {

/// Return an output direction if \p isOutput is true, otherwise return an
/// input direction.
inline Direction get(bool isOutput) { return (Direction)isOutput; }

/// Flip a port direction.
Direction flip(Direction direction);

inline StringRef toString(Direction direction) {
  return direction == Direction::In ? "in" : "out";
}

/// Return a \p IntegerAttr containing the packed representation of an array
/// of directions.
IntegerAttr packAttribute(MLIRContext *context, ArrayRef<Direction> directions);

/// Turn a packed representation of port attributes into a vector that can
/// be worked with.
SmallVector<Direction> unpackAttribute(IntegerAttr directions);

} // namespace direction
} // namespace firrtl
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h.inc"

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLATTRIBUTES_H
