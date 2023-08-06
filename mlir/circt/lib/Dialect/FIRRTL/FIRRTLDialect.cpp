//===- FIRRTLDialect.cpp - Implement the FIRRTL dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void FIRRTLDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/FIRRTL/FIRRTL.cpp.inc"
      >();
}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *FIRRTLDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {

  // Boolean constants. Boolean attributes are always a special constant type
  // like ClockType and ResetType.  Since BoolAttrs are also IntegerAttrs, its
  // important that this goes first.
  if (auto attrValue = dyn_cast<BoolAttr>(value)) {
    assert((isa<ClockType, AsyncResetType, ResetType>(type) &&
            "BoolAttrs can only be materialized for special constant types."));
    return builder.create<SpecialConstantOp>(loc, type, attrValue);
  }

  // Integer constants.
  if (auto attrValue = dyn_cast<IntegerAttr>(value)) {
    // Integer attributes (ui1) might still be special constant types.
    if (attrValue.getValue().getBitWidth() == 1 &&
        isa<ClockType, AsyncResetType, ResetType>(type))
      return builder.create<SpecialConstantOp>(
          loc, type, builder.getBoolAttr(attrValue.getValue().isAllOnes()));

    assert((!type_cast<IntType>(type).hasWidth() ||
            (unsigned)type_cast<IntType>(type).getWidthOrSentinel() ==
                attrValue.getValue().getBitWidth()) &&
           "type/value width mismatch materializing constant");
    return builder.create<ConstantOp>(loc, type, attrValue);
  }

  // Aggregate constants.
  if (auto arrayAttr = dyn_cast<ArrayAttr>(value)) {
    if (isa<BundleType, FVectorType>(type))
      return builder.create<AggregateConstantOp>(loc, type, arrayAttr);
  }

  // String constants.
  if (auto stringAttr = dyn_cast<StringAttr>(value)) {
    if (type_isa<StringType>(type))
      return builder.create<StringConstantOp>(loc, type, stringAttr);
  }

  return nullptr;
}

// Provide implementations for the enums we use.
#include "circt/Dialect/FIRRTL/FIRRTLEnums.cpp.inc"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.cpp.inc"
