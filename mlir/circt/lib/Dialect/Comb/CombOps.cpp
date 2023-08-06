//===- CombOps.cpp - Implement the Comb operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements combinational ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace comb;

/// Create a sign extension operation from a value of integer type to an equal
/// or larger integer type.
Value comb::createOrFoldSExt(Location loc, Value value, Type destTy,
                             OpBuilder &builder) {
  IntegerType valueType = value.getType().dyn_cast<IntegerType>();
  assert(valueType && destTy.isa<IntegerType>() &&
         valueType.getWidth() <= destTy.getIntOrFloatBitWidth() &&
         valueType.getWidth() != 0 && "invalid sext operands");
  // If already the right size, we are done.
  if (valueType == destTy)
    return value;

  // sext is concat with a replicate of the sign bits and the bottom part.
  auto signBit =
      builder.createOrFold<ExtractOp>(loc, value, valueType.getWidth() - 1, 1);
  auto signBits = builder.createOrFold<ReplicateOp>(
      loc, signBit, destTy.getIntOrFloatBitWidth() - valueType.getWidth());
  return builder.createOrFold<ConcatOp>(loc, signBits, value);
}

Value comb::createOrFoldSExt(Value value, Type destTy,
                             ImplicitLocOpBuilder &builder) {
  return createOrFoldSExt(builder.getLoc(), value, destTy, builder);
}

Value comb::createOrFoldNot(Location loc, Value value, OpBuilder &builder,
                            bool twoState) {
  auto allOnes = builder.create<hw::ConstantOp>(loc, value.getType(), -1);
  return builder.createOrFold<XorOp>(loc, value, allOnes, twoState);
}

Value comb::createOrFoldNot(Value value, ImplicitLocOpBuilder &builder,
                            bool twoState) {
  return createOrFoldNot(builder.getLoc(), value, builder, twoState);
}

//===----------------------------------------------------------------------===//
// ICmpOp
//===----------------------------------------------------------------------===//

ICmpPredicate ICmpOp::getFlippedPredicate(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return ICmpPredicate::eq;
  case ICmpPredicate::ne:
    return ICmpPredicate::ne;
  case ICmpPredicate::slt:
    return ICmpPredicate::sgt;
  case ICmpPredicate::sle:
    return ICmpPredicate::sge;
  case ICmpPredicate::sgt:
    return ICmpPredicate::slt;
  case ICmpPredicate::sge:
    return ICmpPredicate::sle;
  case ICmpPredicate::ult:
    return ICmpPredicate::ugt;
  case ICmpPredicate::ule:
    return ICmpPredicate::uge;
  case ICmpPredicate::ugt:
    return ICmpPredicate::ult;
  case ICmpPredicate::uge:
    return ICmpPredicate::ule;
  case ICmpPredicate::ceq:
    return ICmpPredicate::ceq;
  case ICmpPredicate::cne:
    return ICmpPredicate::cne;
  case ICmpPredicate::weq:
    return ICmpPredicate::weq;
  case ICmpPredicate::wne:
    return ICmpPredicate::wne;
  }
  llvm_unreachable("unknown comparison predicate");
}

bool ICmpOp::isPredicateSigned(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::ult:
  case ICmpPredicate::ugt:
  case ICmpPredicate::ule:
  case ICmpPredicate::uge:
  case ICmpPredicate::ne:
  case ICmpPredicate::eq:
  case ICmpPredicate::cne:
  case ICmpPredicate::ceq:
  case ICmpPredicate::wne:
  case ICmpPredicate::weq:
    return false;
  case ICmpPredicate::slt:
  case ICmpPredicate::sgt:
  case ICmpPredicate::sle:
  case ICmpPredicate::sge:
    return true;
  }
  llvm_unreachable("unknown comparison predicate");
}

/// Returns the predicate for a logically negated comparison, e.g. mapping
/// EQ => NE and SLE => SGT.
ICmpPredicate ICmpOp::getNegatedPredicate(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return ICmpPredicate::ne;
  case ICmpPredicate::ne:
    return ICmpPredicate::eq;
  case ICmpPredicate::slt:
    return ICmpPredicate::sge;
  case ICmpPredicate::sle:
    return ICmpPredicate::sgt;
  case ICmpPredicate::sgt:
    return ICmpPredicate::sle;
  case ICmpPredicate::sge:
    return ICmpPredicate::slt;
  case ICmpPredicate::ult:
    return ICmpPredicate::uge;
  case ICmpPredicate::ule:
    return ICmpPredicate::ugt;
  case ICmpPredicate::ugt:
    return ICmpPredicate::ule;
  case ICmpPredicate::uge:
    return ICmpPredicate::ult;
  case ICmpPredicate::ceq:
    return ICmpPredicate::cne;
  case ICmpPredicate::cne:
    return ICmpPredicate::ceq;
  case ICmpPredicate::weq:
    return ICmpPredicate::wne;
  case ICmpPredicate::wne:
    return ICmpPredicate::weq;
  }
  llvm_unreachable("unknown comparison predicate");
}

/// Return true if this is an equality test with -1, which is a "reduction
/// and" operation in Verilog.
bool ICmpOp::isEqualAllOnes() {
  if (getPredicate() != ICmpPredicate::eq)
    return false;

  if (auto op1 =
          dyn_cast_or_null<hw::ConstantOp>(getOperand(1).getDefiningOp()))
    return op1.getValue().isAllOnes();
  return false;
}

/// Return true if this is a not equal test with 0, which is a "reduction
/// or" operation in Verilog.
bool ICmpOp::isNotEqualZero() {
  if (getPredicate() != ICmpPredicate::ne)
    return false;

  if (auto op1 =
          dyn_cast_or_null<hw::ConstantOp>(getOperand(1).getDefiningOp()))
    return op1.getValue().isZero();
  return false;
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

LogicalResult ReplicateOp::verify() {
  // The source must be equal or smaller than the dest type, and an even
  // multiple of it.  Both are already known to be signless integers.
  auto srcWidth = getOperand().getType().cast<IntegerType>().getWidth();
  auto dstWidth = getType().cast<IntegerType>().getWidth();
  if (srcWidth == 0)
    return emitOpError("replicate does not take zero bit integer");

  if (srcWidth > dstWidth)
    return emitOpError("replicate cannot shrink bitwidth of operand"),
           failure();

  if (dstWidth % srcWidth)
    return emitOpError("replicate must produce integer multiple of operand"),
           failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Variadic operations
//===----------------------------------------------------------------------===//

static LogicalResult verifyUTBinOp(Operation *op) {
  if (op->getOperands().empty())
    return op->emitOpError("requires 1 or more args");
  return success();
}

LogicalResult AddOp::verify() { return verifyUTBinOp(*this); }

LogicalResult MulOp::verify() { return verifyUTBinOp(*this); }

LogicalResult AndOp::verify() { return verifyUTBinOp(*this); }

LogicalResult OrOp::verify() { return verifyUTBinOp(*this); }

LogicalResult XorOp::verify() { return verifyUTBinOp(*this); }

/// Return true if this is a two operand xor with an all ones constant as its
/// RHS operand.
bool XorOp::isBinaryNot() {
  if (getNumOperands() != 2)
    return false;
  if (auto cst = getOperand(1).getDefiningOp<hw::ConstantOp>())
    if (cst.getValue().isAllOnes())
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

static unsigned getTotalWidth(ValueRange inputs) {
  unsigned resultWidth = 0;
  for (auto input : inputs) {
    resultWidth += input.getType().cast<IntegerType>().getWidth();
  }
  return resultWidth;
}

LogicalResult ConcatOp::verify() {
  unsigned tyWidth = getType().cast<IntegerType>().getWidth();
  unsigned operandsTotalWidth = getTotalWidth(getInputs());
  if (tyWidth != operandsTotalWidth)
    return emitOpError("ConcatOp requires operands total width to "
                       "match type width. operands "
                       "totalWidth is")
           << operandsTotalWidth << ", but concatOp type width is " << tyWidth;

  return success();
}

void ConcatOp::build(OpBuilder &builder, OperationState &result, Value hd,
                     ValueRange tl) {
  result.addOperands(ValueRange{hd});
  result.addOperands(tl);
  unsigned hdWidth = hd.getType().cast<IntegerType>().getWidth();
  result.addTypes(builder.getIntegerType(getTotalWidth(tl) + hdWidth));
}

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  unsigned resultWidth = getTotalWidth(operands);
  results.push_back(IntegerType::get(context, resultWidth));
  return success();
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::verify() {
  unsigned srcWidth = getInput().getType().cast<IntegerType>().getWidth();
  unsigned dstWidth = getType().cast<IntegerType>().getWidth();
  if (getLowBit() >= srcWidth || srcWidth - getLowBit() < dstWidth)
    return emitOpError("from bit too large for input"), failure();

  return success();
}

LogicalResult TruthTableOp::verify() {
  size_t numInputs = getInputs().size();
  if (numInputs >= sizeof(size_t) * 8)
    return emitOpError("Truth tables support a maximum of ")
           << sizeof(size_t) * 8 - 1 << " inputs on your platform";

  ArrayAttr table = getLookupTable();
  if (table.size() != (1ull << numInputs))
    return emitOpError("Expected lookup table of 2^n length");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Comb/Comb.cpp.inc"
