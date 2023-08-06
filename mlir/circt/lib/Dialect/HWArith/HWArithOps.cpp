//===- HWArithOps.cpp - Implement the HW arithmetic operations ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the HW arithmetic ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HWArith/HWArithOps.h"
#include "circt/Dialect/HWArith/HWArithTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APSInt.h"

using namespace circt;
using namespace hwarith;

namespace circt {
namespace hwarith {
#include "circt/Dialect/HWArith/HWArithCanonicalizations.h.inc"

}
} // namespace circt

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  auto inType = getIn().getType();
  auto outType = getOut().getType();
  bool isInSignless = !isHWArithIntegerType(inType);
  bool isOutSignless = !isHWArithIntegerType(outType);

  if (isInSignless && isOutSignless)
    return emitError("at least one type needs to carry sign semantics (ui/si)");

  if (isInSignless) {
    unsigned inBitWidth = inType.getIntOrFloatBitWidth();
    unsigned outBitWidth = outType.getIntOrFloatBitWidth();
    if (inBitWidth < outBitWidth)
      return emitError("bit extension is undefined for a signless type");
  }

  return success();
}

void CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<EliminateCast>(context);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

APSInt ConstantOp::getConstantValue() { return getRawValueAttr().getAPSInt(); }

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getRawValueAttr();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttribute(getRawValueAttr());
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{getRawValueAttrName()});
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  IntegerAttr valueAttr;

  if (parser.parseAttribute(valueAttr, getRawValueAttrName(result.name),
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(valueAttr.getType());
  return success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::inferReturnTypes(MLIRContext *context,
                                      std::optional<Location> loc,
                                      ValueRange operands, DictionaryAttr attrs,
                                      mlir::OpaqueProperties properties,
                                      mlir::RegionRange regions,
                                      SmallVectorImpl<Type> &results) {
  auto lhs = operands[0].getType().cast<IntegerType>();
  auto rhs = operands[1].getType().cast<IntegerType>();
  IntegerType::SignednessSemantics signedness;
  unsigned resultWidth = inferAddResultType(signedness, lhs, rhs);

  results.push_back(IntegerType::get(context, resultWidth, signedness));
  return success();
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult SubOp::inferReturnTypes(MLIRContext *context,
                                      std::optional<Location> loc,
                                      ValueRange operands, DictionaryAttr attrs,
                                      mlir::OpaqueProperties properties,
                                      mlir::RegionRange regions,
                                      SmallVectorImpl<Type> &results) {
  auto lhs = operands[0].getType().cast<IntegerType>();
  auto rhs = operands[1].getType().cast<IntegerType>();
  // The result type rules are identical to the ones for an addition
  // With one exception: all results are signed!
  IntegerType::SignednessSemantics signedness;
  unsigned resultWidth = inferAddResultType(signedness, lhs, rhs);
  signedness = IntegerType::Signed;

  results.push_back(IntegerType::get(context, resultWidth, signedness));
  return success();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

static IntegerType::SignednessSemantics
getSignedInheritedSignedness(IntegerType lhs, IntegerType rhs) {
  // Signed operands are dominant and enforce a signed result
  if (lhs.getSignedness() == rhs.getSignedness()) {
    // the signedness is also identical to the operands
    return lhs.getSignedness();
  } else {
    // For mixed signedness the result is always signed
    return IntegerType::Signed;
  }
}

LogicalResult MulOp::inferReturnTypes(MLIRContext *context,
                                      std::optional<Location> loc,
                                      ValueRange operands, DictionaryAttr attrs,
                                      mlir::OpaqueProperties properties,
                                      mlir::RegionRange regions,
                                      SmallVectorImpl<Type> &results) {
  auto lhs = operands[0].getType().cast<IntegerType>();
  auto rhs = operands[1].getType().cast<IntegerType>();
  // the result width stays the same no matter the signedness
  unsigned resultWidth = lhs.getWidth() + rhs.getWidth();
  IntegerType::SignednessSemantics signedness =
      getSignedInheritedSignedness(lhs, rhs);

  results.push_back(IntegerType::get(context, resultWidth, signedness));
  return success();
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

LogicalResult DivOp::inferReturnTypes(MLIRContext *context,
                                      std::optional<Location> loc,
                                      ValueRange operands, DictionaryAttr attrs,
                                      mlir::OpaqueProperties properties,
                                      mlir::RegionRange regions,
                                      SmallVectorImpl<Type> &results) {
  auto lhs = operands[0].getType().cast<IntegerType>();
  auto rhs = operands[1].getType().cast<IntegerType>();
  // The result width is always at least as large as the bit width of lhs
  unsigned resultWidth = lhs.getWidth();

  // if the divisor is signed, then the result width needs to be extended by 1
  if (rhs.isSigned())
    ++resultWidth;

  IntegerType::SignednessSemantics signedness =
      getSignedInheritedSignedness(lhs, rhs);

  results.push_back(IntegerType::get(context, resultWidth, signedness));
  return success();
}

//===----------------------------------------------------------------------===//
// Utility
//===----------------------------------------------------------------------===//

namespace circt {
namespace hwarith {

unsigned inferAddResultType(IntegerType::SignednessSemantics &signedness,
                            IntegerType lhs, IntegerType rhs) {
  // the result width is never less than max(w1, w2) + 1
  unsigned resultWidth = std::max(lhs.getWidth(), rhs.getWidth()) + 1;

  if (lhs.getSignedness() == rhs.getSignedness()) {
    // max(w1, w2) + 1 in case both operands use the same signedness
    // the signedness is also identical to the operands
    signedness = lhs.getSignedness();
  } else {
    // For mixed signedness the result is always signed
    signedness = IntegerType::Signed;

    // Regarding the result width two case need to be considered:
    if ((lhs.isUnsigned() && lhs.getWidth() >= rhs.getWidth()) ||
        (rhs.isUnsigned() && rhs.getWidth() >= lhs.getWidth())) {
      // 1. the unsigned width is >= the signed width,
      // then the width needs to be increased by 1
      ++resultWidth;
    }
    // 2. the unsigned width is < the signed width,
    // then no further adjustment is needed
  }
  return resultWidth;
}

static LogicalResult verifyBinOp(Operation *binOp) {
  auto ops = binOp->getOperands();
  if (ops.size() != 2)
    return binOp->emitError() << "expected 2 operands but got " << ops.size();

  return success();
}

} // namespace hwarith
} // namespace circt

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/HWArith/HWArith.cpp.inc"
