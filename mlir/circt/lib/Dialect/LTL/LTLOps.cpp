//===- LTLOps.cpp ==-------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/LTL/LTLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace circt;
using namespace ltl;
using namespace mlir;

#define GET_OP_CLASSES
#include "circt/Dialect/LTL/LTL.cpp.inc"

//===----------------------------------------------------------------------===//
// AndOp / OrOp
//===----------------------------------------------------------------------===//

static LogicalResult inferAndLikeReturnTypes(MLIRContext *context,
                                             ValueRange operands,
                                             SmallVectorImpl<Type> &results) {
  if (llvm::any_of(operands, [](auto operand) {
        return isa<PropertyType>(operand.getType());
      })) {
    results.push_back(PropertyType::get(context));
  } else {

    results.push_back(SequenceType::get(context));
  }
  return success();
}

LogicalResult
AndOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferAndLikeReturnTypes(context, operands, inferredReturnTypes);
}

LogicalResult
OrOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                       ValueRange operands, DictionaryAttr attributes,
                       OpaqueProperties properties, RegionRange regions,
                       SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferAndLikeReturnTypes(context, operands, inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ClockOp
//===----------------------------------------------------------------------===//

LogicalResult
ClockOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                          ValueRange operands, DictionaryAttr attributes,
                          OpaqueProperties properties, RegionRange regions,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  if (isa<PropertyType>(operands[0].getType())) {
    inferredReturnTypes.push_back(PropertyType::get(context));
  } else {
    inferredReturnTypes.push_back(SequenceType::get(context));
  }
  return success();
}
