//===- HWDialect.cpp - Implement the HW dialect ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HW dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "circt/Dialect/HW/HWDialect.cpp.inc"

namespace {

// We implement the OpAsmDialectInterface so that HW dialect operations
// automatically interpret the name attribute on operations as their SSA name.
struct HWOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op, OpAsmSetValueNameFn setNameFn) const {}
};
} // end anonymous namespace

namespace {
/// This class defines the interface for handling inlining with HW operations.
struct HWInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *op, Region *, bool,
                       mlir::IRMapping &) const final {
    return isa<ConstantOp>(op) || isa<AggregateConstantOp>(op) ||
           isa<EnumConstantOp>(op) || isa<BitcastOp>(op) ||
           isa<ArrayCreateOp>(op) || isa<ArrayConcatOp>(op) ||
           isa<ArraySliceOp>(op) || isa<ArrayGetOp>(op) ||
           isa<StructCreateOp>(op) || isa<StructExplodeOp>(op) ||
           isa<StructExtractOp>(op) || isa<StructInjectOp>(op) ||
           isa<UnionCreateOp>(op) || isa<UnionExtractOp>(op);
  }

  bool isLegalToInline(Region *, Region *, bool,
                       mlir::IRMapping &) const final {
    return false;
  }
};
} // end anonymous namespace

void HWDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HW/HW.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<HWOpAsmDialectInterface, HWInlinerInterface>();
}

// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *HWDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  // Integer constants can materialize into hw.constant
  if (auto intType = type.dyn_cast<IntegerType>())
    if (auto attrValue = value.dyn_cast<IntegerAttr>())
      return builder.create<ConstantOp>(loc, type, attrValue);

  // Aggregate constants.
  if (auto arrayAttr = value.dyn_cast<ArrayAttr>()) {
    if (type.isa<StructType, ArrayType, UnpackedArrayType>())
      return builder.create<AggregateConstantOp>(loc, type, arrayAttr);
  }

  // Parameter expressions materialize into hw.param.value.
  auto parentOp = builder.getBlock()->getParentOp();
  auto curModule = dyn_cast<HWModuleOp>(parentOp);
  if (!curModule)
    curModule = parentOp->getParentOfType<HWModuleOp>();
  if (curModule && isValidParameterExpression(value, curModule))
    return builder.create<ParamValueOp>(loc, type, value);

  return nullptr;
}

// Provide implementations for the enums we use.
#include "circt/Dialect/HW/HWEnums.cpp.inc"
