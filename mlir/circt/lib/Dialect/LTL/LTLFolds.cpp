//===- LTLFolds.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace ltl;
using namespace mlir;

/// Check if an attribute is an integer zero.
static bool isConstantZero(Attribute attr) {
  if (!attr)
    return false;
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  return false;
}

/// Concatenate two value ranges into a larger range. Useful for declarative
/// rewrites.
static SmallVector<Value> concatValues(ValueRange a, ValueRange b) {
  SmallVector<Value> v;
  v.append(a.begin(), a.end());
  v.append(b.begin(), b.end());
  return v;
}

/// Inline all `ConcatOp`s in a range of values.
static SmallVector<Value> flattenConcats(ValueRange values) {
  SmallVector<Value> flatInputs;
  for (auto value : values) {
    if (auto concatOp = value.getDefiningOp<ConcatOp>()) {
      auto inputs = concatOp.getInputs();
      flatInputs.append(inputs.begin(), inputs.end());
    } else {
      flatInputs.push_back(value);
    }
  }
  return flatInputs;
}

//===----------------------------------------------------------------------===//
// Declarative Rewrites
//===----------------------------------------------------------------------===//

namespace patterns {
#include "circt/Dialect/LTL/LTLFolds.cpp.inc"
} // namespace patterns

//===----------------------------------------------------------------------===//
// DelayOp
//===----------------------------------------------------------------------===//

OpFoldResult DelayOp::fold(FoldAdaptor adaptor) {
  // delay(s, 0, 0) -> s
  if (adaptor.getDelay() == 0 && adaptor.getLength() == 0)
    return getInput();

  return {};
}

void DelayOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<patterns::NestedDelays>(results.getContext());
  results.add<patterns::MoveDelayIntoConcat>(results.getContext());
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  // concat(s) -> s
  if (getInputs().size() == 1)
    return getInputs()[0];

  return {};
}

void ConcatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<patterns::FlattenConcats>(results.getContext());
}

//===----------------------------------------------------------------------===//
// DisableOp
//===----------------------------------------------------------------------===//

OpFoldResult DisableOp::fold(FoldAdaptor adaptor) {
  // disable(p, false) -> p
  if (isConstantZero(adaptor.getCondition()))
    return getInput();

  return {};
}
