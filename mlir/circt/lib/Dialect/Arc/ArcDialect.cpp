//===- ArcDialect.cpp - Arc dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace circt;
using namespace arc;

namespace {
/// This class defines the interface for handling inlining with Arc operations.
struct ArcInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const override {
    if (auto stateOp = dyn_cast<StateOp>(call);
        stateOp && stateOp.getLatency() == 0)
      return true;

    return isa<CallOp>(call);
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    // TODO
    return false;
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    // TODO
    return false;
  }
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const override {
    assert(isa<arc::OutputOp>(op)); // arc does not have another terminator op
    for (auto [from, to] : llvm::zip(valuesToRepl, op->getOperands()))
      from.replaceAllUsesWith(to);
  }
};
} // end anonymous namespace

void ArcDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Arc/Arc.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<ArcInlinerInterface>();
}

/// Registered hook to materialize a single constant operation from a given
/// attribute value with the desired resultant type. This method should use
/// the provided builder to create the operation without changing the
/// insertion position. The generated operation is expected to be constant
/// like, i.e. single result, zero operands, non side-effecting, etc. On
/// success, this hook should return the value generated to represent the
/// constant value. Otherwise, it should return null on failure.
Operation *ArcDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  // Integer constants.
  if (auto intType = type.dyn_cast<IntegerType>())
    if (auto attrValue = value.dyn_cast<IntegerAttr>())
      return builder.create<hw::ConstantOp>(loc, type, attrValue);

  // Parameter expressions materialize into hw.param.value.
  auto *parentOp = builder.getBlock()->getParentOp();
  auto curModule = dyn_cast<hw::HWModuleOp>(parentOp);
  if (!curModule)
    curModule = parentOp->getParentOfType<hw::HWModuleOp>();
  if (curModule && isValidParameterExpression(value, curModule))
    return builder.create<hw::ParamValueOp>(loc, type, value);

  return nullptr;
}

#include "circt/Dialect/Arc/ArcDialect.cpp.inc"
#include "circt/Dialect/Arc/ArcEnums.cpp.inc"
