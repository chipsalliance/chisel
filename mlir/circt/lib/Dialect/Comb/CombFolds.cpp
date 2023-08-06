//===- CombFolds.cpp - Folds + Canonicalization for Comb operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/KnownBits.h"

using namespace mlir;
using namespace circt;
using namespace comb;
using namespace matchers;

/// Create a new instance of a generic operation that only has value operands,
/// and has a single result value whose type matches the first operand.
///
/// This should not be used to create instances of ops with attributes or with
/// more complicated type signatures.
static Value createGenericOp(Location loc, OperationName name,
                             ArrayRef<Value> operands, OpBuilder &builder) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(operands[0].getType());
  return builder.create(state)->getResult(0);
}

static TypedAttr getIntAttr(const APInt &value, MLIRContext *context) {
  return IntegerAttr::get(IntegerType::get(context, value.getBitWidth()),
                          value);
}

/// Flatten concat and mux operands into a vector.
static void getConcatOperands(Value v, SmallVectorImpl<Value> &result) {
  if (auto concat = v.getDefiningOp<ConcatOp>()) {
    for (auto op : concat.getOperands())
      getConcatOperands(op, result);
  } else if (auto repl = v.getDefiningOp<ReplicateOp>()) {
    for (size_t i = 0, e = repl.getMultiple(); i != e; ++i)
      getConcatOperands(repl.getOperand(), result);
  } else {
    result.push_back(v);
  }
}

/// A wrapper of `PatternRewriter::replaceOp` to propagate "sv.namehint"
/// attribute. If a replaced op has a "sv.namehint" attribute, this function
/// propagates the name to the new value.
static void replaceOpAndCopyName(PatternRewriter &rewriter, Operation *op,
                                 Value newValue) {
  if (auto *newOp = newValue.getDefiningOp()) {
    auto name = op->getAttrOfType<StringAttr>("sv.namehint");
    if (name && !newOp->hasAttr("sv.namehint"))
      rewriter.updateRootInPlace(newOp,
                                 [&] { newOp->setAttr("sv.namehint", name); });
  }
  rewriter.replaceOp(op, newValue);
}

/// A wrapper of `PatternRewriter::replaceOpWithNewOp` to propagate
/// "sv.namehint" attribute. If a replaced op has a "sv.namehint" attribute,
/// this function propagates the name to the new value.
template <typename OpTy, typename... Args>
static OpTy replaceOpWithNewOpAndCopyName(PatternRewriter &rewriter,
                                          Operation *op, Args &&...args) {
  auto name = op->getAttrOfType<StringAttr>("sv.namehint");
  auto newOp =
      rewriter.replaceOpWithNewOp<OpTy>(op, std::forward<Args>(args)...);
  if (name && !newOp->hasAttr("sv.namehint"))
    rewriter.updateRootInPlace(newOp,
                               [&] { newOp->setAttr("sv.namehint", name); });

  return newOp;
}

// Return true if the op has SV attributes. Note that we cannot use a helper
// function `hasSVAttributes` defined under SV dialect because of a cyclic
// dependency.
static bool hasSVAttributes(Operation *op) {
  return op->hasAttr("sv.attributes");
}

namespace {
template <typename SubType>
struct ComplementMatcher {
  SubType lhs;
  ComplementMatcher(SubType lhs) : lhs(std::move(lhs)) {}
  bool match(Operation *op) {
    auto xorOp = dyn_cast<XorOp>(op);
    return xorOp && xorOp.isBinaryNot() && lhs.match(op->getOperand(0));
  }
};
} // end anonymous namespace

template <typename SubType>
static inline ComplementMatcher<SubType> m_Complement(const SubType &subExpr) {
  return ComplementMatcher<SubType>(subExpr);
}

/// Flattens a single input in `op` if `hasOneUse` is true and it can be defined
/// as an Op. Returns true if successful, and false otherwise.
///
/// Example: op(1, 2, op(3, 4), 5) -> op(1, 2, 3, 4, 5)  // returns true
///
static bool tryFlatteningOperands(Operation *op, PatternRewriter &rewriter) {
  auto inputs = op->getOperands();

  for (size_t i = 0, size = inputs.size(); i != size; ++i) {
    Operation *flattenOp = inputs[i].getDefiningOp();
    if (!flattenOp || flattenOp->getName() != op->getName())
      continue;

    // Check for loops
    if (flattenOp == op)
      continue;

    // Don't duplicate logic when it has multiple uses.
    if (!inputs[i].hasOneUse()) {
      // We can fold a multi-use binary operation into this one if this allows a
      // constant to fold though.  For example, fold
      //    (or a, b, c, (or d, cst1), cst2) --> (or a, b, c, d, cst1, cst2)
      // since the constants will both fold and we end up with the equiv cost.
      //
      // We don't do this for add/mul because the hardware won't be shared
      // between the two ops if duplicated.
      if (flattenOp->getNumOperands() != 2 || !isa<AndOp, OrOp, XorOp>(op) ||
          !flattenOp->getOperand(1).getDefiningOp<hw::ConstantOp>() ||
          !inputs.back().getDefiningOp<hw::ConstantOp>())
        continue;
    }

    // Otherwise, flatten away.
    auto flattenOpInputs = flattenOp->getOperands();

    SmallVector<Value, 4> newOperands;
    newOperands.reserve(size + flattenOpInputs.size());

    auto flattenOpIndex = inputs.begin() + i;
    newOperands.append(inputs.begin(), flattenOpIndex);
    newOperands.append(flattenOpInputs.begin(), flattenOpInputs.end());
    newOperands.append(flattenOpIndex + 1, inputs.end());

    Value result =
        createGenericOp(op->getLoc(), op->getName(), newOperands, rewriter);

    // If the original operation and flatten operand have bin flags, propagte
    // the flag to new one.
    if (op->hasAttrOfType<UnitAttr>("twoState") &&
        flattenOp->hasAttrOfType<UnitAttr>("twoState"))
      result.getDefiningOp()->setAttr("twoState", rewriter.getUnitAttr());

    replaceOpAndCopyName(rewriter, op, result);
    return true;
  }
  return false;
}

// Given a range of uses of an operation, find the lowest and highest bits
// inclusive that are ever referenced. The range of uses must not be empty.
static std::pair<size_t, size_t>
getLowestBitAndHighestBitRequired(Operation *op, bool narrowTrailingBits,
                                  size_t originalOpWidth) {
  auto users = op->getUsers();
  assert(!users.empty() &&
         "getLowestBitAndHighestBitRequired cannot operate on "
         "a empty list of uses.");

  // when we don't want to narrowTrailingBits (namely in arithmetic
  // operations), forcing lowestBitRequired = 0
  size_t lowestBitRequired = narrowTrailingBits ? originalOpWidth - 1 : 0;
  size_t highestBitRequired = 0;

  for (auto *user : users) {
    if (auto extractOp = dyn_cast<ExtractOp>(user)) {
      size_t lowBit = extractOp.getLowBit();
      size_t highBit =
          extractOp.getType().cast<IntegerType>().getWidth() + lowBit - 1;
      highestBitRequired = std::max(highestBitRequired, highBit);
      lowestBitRequired = std::min(lowestBitRequired, lowBit);
      continue;
    }

    highestBitRequired = originalOpWidth - 1;
    lowestBitRequired = 0;
    break;
  }

  return {lowestBitRequired, highestBitRequired};
}

template <class OpTy>
static bool narrowOperationWidth(OpTy op, bool narrowTrailingBits,
                                 PatternRewriter &rewriter) {
  IntegerType opType =
      op.getResult().getType().template dyn_cast<IntegerType>();
  if (!opType)
    return false;

  auto range = getLowestBitAndHighestBitRequired(op, narrowTrailingBits,
                                                 opType.getWidth());
  if (range.second + 1 == opType.getWidth() && range.first == 0)
    return false;

  SmallVector<Value> args;
  auto newType = rewriter.getIntegerType(range.second - range.first + 1);
  for (auto inop : op.getOperands()) {
    // deal with muxes here
    if (inop.getType() != op.getType())
      args.push_back(inop);
    else
      args.push_back(rewriter.createOrFold<ExtractOp>(inop.getLoc(), newType,
                                                      inop, range.first));
  }
  Value newop = rewriter.createOrFold<OpTy>(op.getLoc(), newType, args);
  newop.getDefiningOp()->setDialectAttrs(op->getDialectAttrs());
  if (range.first)
    newop = rewriter.createOrFold<ConcatOp>(
        op.getLoc(), newop,
        rewriter.create<hw::ConstantOp>(op.getLoc(),
                                        APInt::getZero(range.first)));
  if (range.second + 1 < opType.getWidth())
    newop = rewriter.createOrFold<ConcatOp>(
        op.getLoc(),
        rewriter.create<hw::ConstantOp>(
            op.getLoc(), APInt::getZero(opType.getWidth() - range.second - 1)),
        newop);
  rewriter.replaceOp(op, newop);
  return true;
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

OpFoldResult ReplicateOp::fold(FoldAdaptor adaptor) {
  // Replicate one time -> noop.
  if (getType().cast<IntegerType>().getWidth() ==
      getInput().getType().getIntOrFloatBitWidth())
    return getInput();

  // Constant fold.
  if (auto input = adaptor.getInput().dyn_cast_or_null<IntegerAttr>()) {
    if (input.getValue().getBitWidth() == 1) {
      if (input.getValue().isZero())
        return getIntAttr(
            APInt::getZero(getType().cast<IntegerType>().getWidth()),
            getContext());
      return getIntAttr(
          APInt::getAllOnes(getType().cast<IntegerType>().getWidth()),
          getContext());
    }

    APInt result = APInt::getZeroWidth();
    for (auto i = getMultiple(); i != 0; --i)
      result = result.concat(input.getValue());
    return getIntAttr(result, getContext());
  }

  return {};
}

OpFoldResult ParityOp::fold(FoldAdaptor adaptor) {
  // Constant fold.
  if (auto input = adaptor.getInput().dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(APInt(1, input.getValue().popcount() & 1), getContext());

  return {};
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

/// Performs constant folding `calculate` with element-wise behavior on the two
/// attributes in `operands` and returns the result if possible.
static Attribute constFoldBinaryOp(ArrayRef<Attribute> operands,
                                   hw::PEO paramOpcode) {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (!operands[0] || !operands[1])
    return {};

  // Fold constants with ParamExprAttr::get which handles simple constants as
  // well as parameter expressions.
  return hw::ParamExprAttr::get(paramOpcode, operands[0].cast<TypedAttr>(),
                                operands[1].cast<TypedAttr>());
}

OpFoldResult ShlOp::fold(FoldAdaptor adaptor) {
  if (auto rhs = adaptor.getRhs().dyn_cast_or_null<IntegerAttr>()) {
    unsigned shift = rhs.getValue().getZExtValue();
    unsigned width = getType().getIntOrFloatBitWidth();
    if (shift == 0)
      return getOperand(0);
    if (width <= shift)
      return getIntAttr(APInt::getZero(width), getContext());
  }

  return constFoldBinaryOp(adaptor.getOperands(), hw::PEO::Shl);
}

LogicalResult ShlOp::canonicalize(ShlOp op, PatternRewriter &rewriter) {
  // ShlOp(x, cst) -> Concat(Extract(x), zeros)
  APInt value;
  if (!matchPattern(op.getRhs(), m_ConstantInt(&value)))
    return failure();

  unsigned width = op.getLhs().getType().cast<IntegerType>().getWidth();
  unsigned shift = value.getZExtValue();

  // This case is handled by fold.
  if (width <= shift || shift == 0)
    return failure();

  auto zeros =
      rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(shift));

  // Remove the high bits which would be removed by the Shl.
  auto extract =
      rewriter.create<ExtractOp>(op.getLoc(), op.getLhs(), 0, width - shift);

  replaceOpWithNewOpAndCopyName<ConcatOp>(rewriter, op, extract, zeros);
  return success();
}

OpFoldResult ShrUOp::fold(FoldAdaptor adaptor) {
  if (auto rhs = adaptor.getRhs().dyn_cast_or_null<IntegerAttr>()) {
    unsigned shift = rhs.getValue().getZExtValue();
    if (shift == 0)
      return getOperand(0);

    unsigned width = getType().getIntOrFloatBitWidth();
    if (width <= shift)
      return getIntAttr(APInt::getZero(width), getContext());
  }
  return constFoldBinaryOp(adaptor.getOperands(), hw::PEO::ShrU);
}

LogicalResult ShrUOp::canonicalize(ShrUOp op, PatternRewriter &rewriter) {
  // ShrUOp(x, cst) -> Concat(zeros, Extract(x))
  APInt value;
  if (!matchPattern(op.getRhs(), m_ConstantInt(&value)))
    return failure();

  unsigned width = op.getLhs().getType().cast<IntegerType>().getWidth();
  unsigned shift = value.getZExtValue();

  // This case is handled by fold.
  if (width <= shift || shift == 0)
    return failure();

  auto zeros =
      rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(shift));

  // Remove the low bits which would be removed by the Shr.
  auto extract = rewriter.create<ExtractOp>(op.getLoc(), op.getLhs(), shift,
                                            width - shift);

  replaceOpWithNewOpAndCopyName<ConcatOp>(rewriter, op, zeros, extract);
  return success();
}

OpFoldResult ShrSOp::fold(FoldAdaptor adaptor) {
  if (auto rhs = adaptor.getRhs().dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue().getZExtValue() == 0)
      return getOperand(0);
  }
  return constFoldBinaryOp(adaptor.getOperands(), hw::PEO::ShrS);
}

LogicalResult ShrSOp::canonicalize(ShrSOp op, PatternRewriter &rewriter) {
  // ShrSOp(x, cst) -> Concat(replicate(extract(x, topbit)),extract(x))
  APInt value;
  if (!matchPattern(op.getRhs(), m_ConstantInt(&value)))
    return failure();

  unsigned width = op.getLhs().getType().cast<IntegerType>().getWidth();
  unsigned shift = value.getZExtValue();

  auto topbit =
      rewriter.createOrFold<ExtractOp>(op.getLoc(), op.getLhs(), width - 1, 1);
  auto sext = rewriter.createOrFold<ReplicateOp>(op.getLoc(), topbit, shift);

  if (width <= shift) {
    replaceOpAndCopyName(rewriter, op, {sext});
    return success();
  }

  auto extract = rewriter.create<ExtractOp>(op.getLoc(), op.getLhs(), shift,
                                            width - shift);

  replaceOpWithNewOpAndCopyName<ConcatOp>(rewriter, op, sext, extract);
  return success();
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

OpFoldResult ExtractOp::fold(FoldAdaptor adaptor) {
  // If we are extracting the entire input, then return it.
  if (getInput().getType() == getType())
    return getInput();

  // Constant fold.
  if (auto input = adaptor.getInput().dyn_cast_or_null<IntegerAttr>()) {
    unsigned dstWidth = getType().cast<IntegerType>().getWidth();
    return getIntAttr(input.getValue().lshr(getLowBit()).trunc(dstWidth),
                      getContext());
  }
  return {};
}

// Transforms extract(lo, cat(a, b, c, d, e)) into
// cat(extract(lo1, b), c, extract(lo2, d)).
// innerCat must be the argument of the provided ExtractOp.
static LogicalResult extractConcatToConcatExtract(ExtractOp op,
                                                  ConcatOp innerCat,
                                                  PatternRewriter &rewriter) {
  auto reversedConcatArgs = llvm::reverse(innerCat.getInputs());
  size_t beginOfFirstRelevantElement = 0;
  auto it = reversedConcatArgs.begin();
  size_t lowBit = op.getLowBit();

  // This loop finds the first concatArg that is covered by the ExtractOp
  for (; it != reversedConcatArgs.end(); it++) {
    assert(beginOfFirstRelevantElement <= lowBit &&
           "incorrectly moved past an element that lowBit has coverage over");
    auto operand = *it;

    size_t operandWidth = operand.getType().getIntOrFloatBitWidth();
    if (lowBit < beginOfFirstRelevantElement + operandWidth) {
      // A bit other than the first bit will be used in this element.
      // ...... ........ ...
      //           ^---lowBit
      //        ^---beginOfFirstRelevantElement
      //
      // Edge-case close to the end of the range.
      // ...... ........ ...
      //                 ^---(position + operandWidth)
      //               ^---lowBit
      //        ^---beginOfFirstRelevantElement
      //
      // Edge-case close to the beginning of the rang
      // ...... ........ ...
      //        ^---lowBit
      //        ^---beginOfFirstRelevantElement
      //
      break;
    }

    // extraction discards this element.
    // ...... ........  ...
    // |      ^---lowBit
    // ^---beginOfFirstRelevantElement
    beginOfFirstRelevantElement += operandWidth;
  }
  assert(it != reversedConcatArgs.end() &&
         "incorrectly failed to find an element which contains coverage of "
         "lowBit");

  SmallVector<Value> reverseConcatArgs;
  size_t widthRemaining = op.getType().cast<IntegerType>().getWidth();
  size_t extractLo = lowBit - beginOfFirstRelevantElement;

  // Transform individual arguments of innerCat(..., a, b, c,) into
  // [ extract(a), b, extract(c) ], skipping an extract operation where
  // possible.
  for (; widthRemaining != 0 && it != reversedConcatArgs.end(); it++) {
    auto concatArg = *it;
    size_t operandWidth = concatArg.getType().getIntOrFloatBitWidth();
    size_t widthToConsume = std::min(widthRemaining, operandWidth - extractLo);

    if (widthToConsume == operandWidth && extractLo == 0) {
      reverseConcatArgs.push_back(concatArg);
    } else {
      auto resultType = IntegerType::get(rewriter.getContext(), widthToConsume);
      reverseConcatArgs.push_back(
          rewriter.create<ExtractOp>(op.getLoc(), resultType, *it, extractLo));
    }

    widthRemaining -= widthToConsume;

    // Beyond the first element, all elements are extracted from position 0.
    extractLo = 0;
  }

  if (reverseConcatArgs.size() == 1) {
    replaceOpAndCopyName(rewriter, op, reverseConcatArgs[0]);
  } else {
    replaceOpWithNewOpAndCopyName<ConcatOp>(
        rewriter, op, SmallVector<Value>(llvm::reverse(reverseConcatArgs)));
  }
  return success();
}

// Transforms extract(lo, replicate(a, N)) into replicate(a, N-c).
static bool extractFromReplicate(ExtractOp op, ReplicateOp replicate,
                                 PatternRewriter &rewriter) {
  auto extractResultWidth = op.getType().cast<IntegerType>().getWidth();
  auto replicateEltWidth =
      replicate.getOperand().getType().getIntOrFloatBitWidth();

  // If the extract starts at the base of an element and is an even multiple,
  // we can replace the extract with a smaller replicate.
  if (op.getLowBit() % replicateEltWidth == 0 &&
      extractResultWidth % replicateEltWidth == 0) {
    replaceOpWithNewOpAndCopyName<ReplicateOp>(rewriter, op, op.getType(),
                                               replicate.getOperand());
    return true;
  }

  // If the extract is completely contained in one element, extract from the
  // element.
  if (op.getLowBit() % replicateEltWidth + extractResultWidth <=
      replicateEltWidth) {
    replaceOpWithNewOpAndCopyName<ExtractOp>(
        rewriter, op, op.getType(), replicate.getOperand(),
        op.getLowBit() % replicateEltWidth);
    return true;
  }

  // We don't currently handle the case of extracting from non-whole elements,
  // e.g. `extract (replicate 2-bit-thing, N), 1`.
  return false;
}

LogicalResult ExtractOp::canonicalize(ExtractOp op, PatternRewriter &rewriter) {
  auto *inputOp = op.getInput().getDefiningOp();

  // This turns out to be incredibly expensive.  Disable until performance is
  // addressed.
#if 0
  // If the extracted bits are all known, then return the result.
  auto knownBits = computeKnownBits(op.getInput())
                       .extractBits(op.getType().cast<IntegerType>().getWidth(),
                                    op.getLowBit());
  if (knownBits.isConstant()) {
    replaceOpWithNewOpAndCopyName<hw::ConstantOp>(rewriter, op,
                                                  knownBits.getConstant());
    return success();
  }
#endif

  // extract(olo, extract(ilo, x)) = extract(olo + ilo, x)
  if (auto innerExtract = dyn_cast_or_null<ExtractOp>(inputOp)) {
    replaceOpWithNewOpAndCopyName<ExtractOp>(
        rewriter, op, op.getType(), innerExtract.getInput(),
        innerExtract.getLowBit() + op.getLowBit());
    return success();
  }

  // extract(lo, cat(a, b, c, d, e)) = cat(extract(lo1, b), c, extract(lo2, d))
  if (auto innerCat = dyn_cast_or_null<ConcatOp>(inputOp))
    return extractConcatToConcatExtract(op, innerCat, rewriter);

  // extract(lo, replicate(a))
  if (auto replicate = dyn_cast_or_null<ReplicateOp>(inputOp))
    if (extractFromReplicate(op, replicate, rewriter))
      return success();

  // `extract(and(a, cst))` -> `extract(a)` when the relevant bits of the
  // and/or/xor are not modifying the extracted bits.
  if (inputOp && inputOp->getNumOperands() == 2 &&
      isa<AndOp, OrOp, XorOp>(inputOp)) {
    if (auto cstRHS = inputOp->getOperand(1).getDefiningOp<hw::ConstantOp>()) {
      auto extractedCst = cstRHS.getValue().extractBits(
          op.getType().cast<IntegerType>().getWidth(), op.getLowBit());
      if (isa<OrOp, XorOp>(inputOp) && extractedCst.isZero()) {
        replaceOpWithNewOpAndCopyName<ExtractOp>(
            rewriter, op, op.getType(), inputOp->getOperand(0), op.getLowBit());
        return success();
      }

      // `extract(and(a, cst))` -> `concat(extract(a), 0)` if we only need one
      // extract to represent the result.  Turning it into a pile of extracts is
      // always fine by our cost model, but we don't want to explode things into
      // a ton of bits because it will bloat the IR and generated Verilog.
      if (isa<AndOp>(inputOp)) {
        // For our cost model, we only do this if the bit pattern is a
        // contiguous series of ones.
        unsigned lz = extractedCst.countLeadingZeros();
        unsigned tz = extractedCst.countTrailingZeros();
        unsigned pop = extractedCst.popcount();
        if (extractedCst.getBitWidth() - lz - tz == pop) {
          auto resultTy = rewriter.getIntegerType(pop);
          SmallVector<Value> resultElts;
          if (lz)
            resultElts.push_back(rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt::getZero(lz)));
          resultElts.push_back(rewriter.createOrFold<ExtractOp>(
              op.getLoc(), resultTy, inputOp->getOperand(0),
              op.getLowBit() + tz));
          if (tz)
            resultElts.push_back(rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt::getZero(tz)));
          replaceOpWithNewOpAndCopyName<ConcatOp>(rewriter, op, resultElts);
          return success();
        }
      }
    }
  }

  // `extract(lowBit, shl(1, x))` -> `x == lowBit` when a single bit is
  // extracted.
  if (op.getType().cast<IntegerType>().getWidth() == 1 && inputOp)
    if (auto shlOp = dyn_cast<ShlOp>(inputOp))
      if (auto lhsCst = shlOp.getOperand(0).getDefiningOp<hw::ConstantOp>())
        if (lhsCst.getValue().isOne()) {
          auto newCst = rewriter.create<hw::ConstantOp>(
              shlOp.getLoc(),
              APInt(lhsCst.getValue().getBitWidth(), op.getLowBit()));
          replaceOpWithNewOpAndCopyName<ICmpOp>(rewriter, op, ICmpPredicate::eq,
                                                shlOp->getOperand(1), newCst,
                                                false);
          return success();
        }

  return failure();
}

//===----------------------------------------------------------------------===//
// Associative Variadic operations
//===----------------------------------------------------------------------===//

// Reduce all operands to a single value (either integer constant or parameter
// expression) if all the operands are constants.
static Attribute constFoldAssociativeOp(ArrayRef<Attribute> operands,
                                        hw::PEO paramOpcode) {
  assert(operands.size() > 1 && "caller should handle one-operand case");
  // We can only fold anything in the case where all operands are known to be
  // constants.  Check the least common one first for an early out.
  if (!operands[1] || !operands[0])
    return {};

  // This will fold to a simple constant if all operands are constant.
  if (llvm::all_of(operands.drop_front(2),
                   [&](Attribute in) { return !!in; })) {
    SmallVector<mlir::TypedAttr> typedOperands;
    typedOperands.reserve(operands.size());
    for (auto operand : operands) {
      if (auto typedOperand = operand.dyn_cast<mlir::TypedAttr>())
        typedOperands.push_back(typedOperand);
      else
        break;
    }
    if (typedOperands.size() == operands.size())
      return hw::ParamExprAttr::get(paramOpcode, typedOperands);
  }

  return {};
}

/// When we find a logical operation (and, or, xor) with a constant e.g.
/// `X & 42`, we want to push the constant into the computation of X if it leads
/// to simplification.
///
/// This function handles the case where the logical operation has a concat
/// operand.  We check to see if we can simplify the concat, e.g. when it has
/// constant operands.
///
/// This returns true when a simplification happens.
static bool canonicalizeLogicalCstWithConcat(Operation *logicalOp,
                                             size_t concatIdx, const APInt &cst,
                                             PatternRewriter &rewriter) {
  auto concatOp = logicalOp->getOperand(concatIdx).getDefiningOp<ConcatOp>();
  assert((isa<AndOp, OrOp, XorOp>(logicalOp) && concatOp));

  // Check to see if any operands can be simplified by pushing the logical op
  // into all parts of the concat.
  bool canSimplify =
      llvm::any_of(concatOp->getOperands(), [&](Value operand) -> bool {
        auto *operandOp = operand.getDefiningOp();
        if (!operandOp)
          return false;

        // If the concat has a constant operand then we can transform this.
        if (isa<hw::ConstantOp>(operandOp))
          return true;
        // If the concat has the same logical operation and that operation has
        // a constant operation than we can fold it into that suboperation.
        return operandOp->getName() == logicalOp->getName() &&
               operandOp->hasOneUse() && operandOp->getNumOperands() != 0 &&
               operandOp->getOperands().back().getDefiningOp<hw::ConstantOp>();
      });

  if (!canSimplify)
    return false;

  // Create a new instance of the logical operation.  We have to do this the
  // hard way since we're generic across a family of different ops.
  auto createLogicalOp = [&](ArrayRef<Value> operands) -> Value {
    return createGenericOp(logicalOp->getLoc(), logicalOp->getName(), operands,
                           rewriter);
  };

  // Ok, let's do the transformation.  We do this by slicing up the constant
  // for each unit of the concat and duplicate the operation into the
  // sub-operand.
  SmallVector<Value> newConcatOperands;
  newConcatOperands.reserve(concatOp->getNumOperands());

  // Work from MSB to LSB.
  size_t nextOperandBit = concatOp.getType().getIntOrFloatBitWidth();
  for (Value operand : concatOp->getOperands()) {
    size_t operandWidth = operand.getType().getIntOrFloatBitWidth();
    nextOperandBit -= operandWidth;
    // Take a slice of the constant.
    auto eltCst = rewriter.create<hw::ConstantOp>(
        logicalOp->getLoc(), cst.lshr(nextOperandBit).trunc(operandWidth));

    newConcatOperands.push_back(createLogicalOp({operand, eltCst}));
  }

  // Create the concat, and the rest of the logical op if we need it.
  Value newResult =
      rewriter.create<ConcatOp>(concatOp.getLoc(), newConcatOperands);

  // If we had a variadic logical op on the top level, then recreate it with the
  // new concat and without the constant operand.
  if (logicalOp->getNumOperands() > 2) {
    auto origOperands = logicalOp->getOperands();
    SmallVector<Value> operands;
    // Take any stuff before the concat.
    operands.append(origOperands.begin(), origOperands.begin() + concatIdx);
    // Take any stuff after the concat but before the constant.
    operands.append(origOperands.begin() + concatIdx + 1,
                    origOperands.begin() + (origOperands.size() - 1));
    // Include the new concat.
    operands.push_back(newResult);
    newResult = createLogicalOp(operands);
  }

  replaceOpAndCopyName(rewriter, logicalOp, newResult);
  return true;
}

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  APInt value = APInt::getAllOnes(getType().cast<IntegerType>().getWidth());

  auto inputs = adaptor.getInputs();

  // and(x, 01, 10) -> 00 -- annulment.
  for (auto operand : inputs) {
    if (!operand)
      continue;
    value &= operand.cast<IntegerAttr>().getValue();
    if (value.isZero())
      return getIntAttr(value, getContext());
  }

  // and(x, -1) -> x.
  if (inputs.size() == 2 && inputs[1] &&
      inputs[1].cast<IntegerAttr>().getValue().isAllOnes())
    return getInputs()[0];

  // and(x, x, x) -> x.  This also handles and(x) -> x.
  if (llvm::all_of(getInputs(),
                   [&](auto in) { return in == this->getInputs()[0]; }))
    return getInputs()[0];

  // and(..., x, ..., ~x, ...) -> 0
  for (Value arg : getInputs()) {
    Value subExpr;
    if (matchPattern(arg, m_Complement(m_Any(&subExpr)))) {
      for (Value arg2 : getInputs())
        if (arg2 == subExpr)
          return getIntAttr(
              APInt::getZero(getType().cast<IntegerType>().getWidth()),
              getContext());
    }
  }

  // Constant fold
  return constFoldAssociativeOp(inputs, hw::PEO::And);
}

/// Returns a single common operand that all inputs of the operation `op` can
/// be traced back to, or an empty `Value` if no such operand exists.
///
/// For example for `or(a[0], a[1], ..., a[n-1])` this function returns `a`
/// (assuming the bit-width of `a` is `n`).
template <typename Op>
static Value getCommonOperand(Op op) {
  if (!op.getType().isInteger(1))
    return Value();

  auto inputs = op.getInputs();
  size_t size = inputs.size();

  auto sourceOp = inputs[0].template getDefiningOp<ExtractOp>();
  if (!sourceOp)
    return Value();
  Value source = sourceOp.getOperand();

  // Fast path: the input size is not equal to the width of the source.
  if (size != source.getType().getIntOrFloatBitWidth())
    return Value();

  // Tracks the bits that were encountered.
  llvm::BitVector bits(size);
  bits.set(sourceOp.getLowBit());

  for (size_t i = 1; i != size; ++i) {
    auto extractOp = inputs[i].template getDefiningOp<ExtractOp>();
    if (!extractOp || extractOp.getOperand() != source)
      return Value();
    bits.set(extractOp.getLowBit());
  }

  return bits.all() ? source : Value();
}

/// Canonicalize an idempotent operation `op` so that only one input of any kind
/// occurs.
///
/// Example: `and(x, y, x, z)` -> `and(x, y, z)`
template <typename Op>
static bool canonicalizeIdempotentInputs(Op op, PatternRewriter &rewriter) {
  auto inputs = op.getInputs();
  llvm::SmallSetVector<Value, 8> uniqueInputs;

  for (const auto input : inputs)
    uniqueInputs.insert(input);

  if (uniqueInputs.size() < inputs.size()) {
    replaceOpWithNewOpAndCopyName<Op>(rewriter, op, op.getType(),
                                      uniqueInputs.getArrayRef());
    return true;
  }

  return false;
}

LogicalResult AndOp::canonicalize(AndOp op, PatternRewriter &rewriter) {
  auto inputs = op.getInputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands, `fold` should handle this");

  // and(..., x, ..., x) -> and(..., x, ...) -- idempotent
  // Trivial and(x), and(x, x) cases are handled by [AndOp::fold] above.
  if (size > 2 && canonicalizeIdempotentInputs(op, rewriter))
    return success();

  // Patterns for and with a constant on RHS.
  APInt value;
  if (matchPattern(inputs.back(), m_ConstantInt(&value))) {
    // and(..., '1) -> and(...) -- identity
    if (value.isAllOnes()) {
      replaceOpWithNewOpAndCopyName<AndOp>(rewriter, op, op.getType(),
                                           inputs.drop_back(), false);
      return success();
    }

    // TODO: Combine multiple constants together even if they aren't at the
    // end. and(..., c1, c2) -> and(..., c3) where c3 = c1 & c2 -- constant
    // folding
    APInt value2;
    if (matchPattern(inputs[size - 2], m_ConstantInt(&value2))) {
      auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value & value2);
      SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
      newOperands.push_back(cst);
      replaceOpWithNewOpAndCopyName<AndOp>(rewriter, op, op.getType(),
                                           newOperands, false);
      return success();
    }

    // Handle 'and' with a single bit constant on the RHS.
    if (size == 2 && value.isPowerOf2()) {
      // If the LHS is a replicate from a single bit, we can 'concat' it
      // into place.  e.g.:
      //   `replicate(x) & 4` -> `concat(zeros, x, zeros)`
      // TODO: Generalize this for non-single-bit operands.
      if (auto replicate = inputs[0].getDefiningOp<ReplicateOp>()) {
        auto replicateOperand = replicate.getOperand();
        if (replicateOperand.getType().isInteger(1)) {
          unsigned resultWidth = op.getType().getIntOrFloatBitWidth();
          auto trailingZeros = value.countTrailingZeros();

          // Don't add zero bit constants unnecessarily.
          SmallVector<Value, 3> concatOperands;
          if (trailingZeros != resultWidth - 1) {
            auto highZeros = rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt::getZero(resultWidth - trailingZeros - 1));
            concatOperands.push_back(highZeros);
          }
          concatOperands.push_back(replicateOperand);
          if (trailingZeros != 0) {
            auto lowZeros = rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt::getZero(trailingZeros));
            concatOperands.push_back(lowZeros);
          }
          replaceOpWithNewOpAndCopyName<ConcatOp>(rewriter, op, op.getType(),
                                                  concatOperands);
          return success();
        }
      }
    }

    // If this is an and from an extract op, try shrinking the extract.
    if (auto extractOp = inputs[0].getDefiningOp<ExtractOp>()) {
      if (size == 2 &&
          // We can shrink it if the mask has leading or trailing zeros.
          (value.countLeadingZeros() || value.countTrailingZeros())) {
        unsigned lz = value.countLeadingZeros();
        unsigned tz = value.countTrailingZeros();

        // Start by extracting the smaller number of bits.
        auto smallTy = rewriter.getIntegerType(value.getBitWidth() - lz - tz);
        Value smallElt = rewriter.createOrFold<ExtractOp>(
            extractOp.getLoc(), smallTy, extractOp->getOperand(0),
            extractOp.getLowBit() + tz);
        // Apply the 'and' mask if needed.
        APInt smallMask = value.extractBits(smallTy.getWidth(), tz);
        if (!smallMask.isAllOnes()) {
          auto loc = inputs.back().getLoc();
          smallElt = rewriter.createOrFold<AndOp>(
              loc, smallElt, rewriter.create<hw::ConstantOp>(loc, smallMask),
              false);
        }

        // The final replacement will be a concat of the leading/trailing zeros
        // along with the smaller extracted value.
        SmallVector<Value> resultElts;
        if (lz)
          resultElts.push_back(
              rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(lz)));
        resultElts.push_back(smallElt);
        if (tz)
          resultElts.push_back(
              rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(tz)));
        replaceOpWithNewOpAndCopyName<ConcatOp>(rewriter, op, resultElts);
        return success();
      }
    }

    // and(concat(x, cst1), a, b, c, cst2)
    //    ==> and(a, b, c, concat(and(x,cst2'), and(cst1,cst2'')).
    // We do this for even more multi-use concats since they are "just wiring".
    for (size_t i = 0; i < size - 1; ++i) {
      if (auto concat = inputs[i].getDefiningOp<ConcatOp>())
        if (canonicalizeLogicalCstWithConcat(op, i, value, rewriter))
          return success();
    }
  }

  // and(x, and(...)) -> and(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  // extracts only of and(...) -> and(extract()...)
  if (narrowOperationWidth(op, true, rewriter))
    return success();

  // and(a[0], a[1], ..., a[n]) -> icmp eq(a, -1)
  if (auto source = getCommonOperand(op)) {
    auto cmpAgainst =
        rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getAllOnes(size));
    replaceOpWithNewOpAndCopyName<ICmpOp>(rewriter, op, ICmpPredicate::eq,
                                          source, cmpAgainst);
    return success();
  }

  /// TODO: and(..., x, not(x)) -> and(..., 0) -- complement
  return failure();
}

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  auto value = APInt::getZero(getType().cast<IntegerType>().getWidth());
  auto inputs = adaptor.getInputs();
  // or(x, 10, 01) -> 11
  for (auto operand : inputs) {
    if (!operand)
      continue;
    value |= operand.cast<IntegerAttr>().getValue();
    if (value.isAllOnes())
      return getIntAttr(value, getContext());
  }

  // or(x, 0) -> x
  if (inputs.size() == 2 && inputs[1] &&
      inputs[1].cast<IntegerAttr>().getValue().isZero())
    return getInputs()[0];

  // or(x, x, x) -> x.  This also handles or(x) -> x
  if (llvm::all_of(getInputs(),
                   [&](auto in) { return in == this->getInputs()[0]; }))
    return getInputs()[0];

  // or(..., x, ..., ~x, ...) -> -1
  for (Value arg : getInputs()) {
    Value subExpr;
    if (matchPattern(arg, m_Complement(m_Any(&subExpr)))) {
      for (Value arg2 : getInputs())
        if (arg2 == subExpr)
          return getIntAttr(
              APInt::getAllOnes(getType().cast<IntegerType>().getWidth()),
              getContext());
    }
  }

  // Constant fold
  return constFoldAssociativeOp(inputs, hw::PEO::Or);
}

/// Simplify concat ops in an or op when a constant operand is present in either
/// concat.
///
/// This will invert an or(concat, concat) into concat(or, or, ...), which can
/// often be further simplified due to the smaller or ops being easier to fold.
///
/// For example:
///
/// or(..., concat(x, 0), concat(0, y))
///    ==> or(..., concat(x, 0, y)), when x and y don't overlap.
///
/// or(..., concat(x: i2, cst1: i4), concat(cst2: i5, y: i1))
///    ==> or(..., concat(or(x: i2,               extract(cst2, 4..3)),
///                       or(extract(cst1, 3..1), extract(cst2, 2..0)),
///                       or(extract(cst1, 0..0), y: i1))
static bool canonicalizeOrOfConcatsWithCstOperands(OrOp op, size_t concatIdx1,
                                                   size_t concatIdx2,
                                                   PatternRewriter &rewriter) {
  assert(concatIdx1 < concatIdx2 && "concatIdx1 must be < concatIdx2");

  auto inputs = op.getInputs();
  auto concat1 = inputs[concatIdx1].getDefiningOp<ConcatOp>();
  auto concat2 = inputs[concatIdx2].getDefiningOp<ConcatOp>();

  assert(concat1 && concat2 && "expected indexes to point to ConcatOps");

  // We can simplify as long as a constant is present in either concat.
  bool hasConstantOp1 =
      llvm::any_of(concat1->getOperands(), [&](Value operand) -> bool {
        return operand.getDefiningOp<hw::ConstantOp>();
      });
  if (!hasConstantOp1) {
    bool hasConstantOp2 =
        llvm::any_of(concat2->getOperands(), [&](Value operand) -> bool {
          return operand.getDefiningOp<hw::ConstantOp>();
        });
    if (!hasConstantOp2)
      return false;
  }

  SmallVector<Value> newConcatOperands;

  // Simultaneously iterate over the operands of both concat ops, from MSB to
  // LSB, pushing out or's of overlapping ranges of the operands. When operands
  // span different bit ranges, we extract only the maximum overlap.
  auto operands1 = concat1->getOperands();
  auto operands2 = concat2->getOperands();
  // Number of bits already consumed from operands 1 and 2, respectively.
  unsigned consumedWidth1 = 0;
  unsigned consumedWidth2 = 0;
  for (auto it1 = operands1.begin(), end1 = operands1.end(),
            it2 = operands2.begin(), end2 = operands2.end();
       it1 != end1 && it2 != end2;) {
    auto operand1 = *it1;
    auto operand2 = *it2;

    unsigned remainingWidth1 =
        hw::getBitWidth(operand1.getType()) - consumedWidth1;
    unsigned remainingWidth2 =
        hw::getBitWidth(operand2.getType()) - consumedWidth2;
    unsigned widthToConsume = std::min(remainingWidth1, remainingWidth2);
    auto narrowedType = rewriter.getIntegerType(widthToConsume);

    auto extract1 = rewriter.createOrFold<ExtractOp>(
        op.getLoc(), narrowedType, operand1, remainingWidth1 - widthToConsume);
    auto extract2 = rewriter.createOrFold<ExtractOp>(
        op.getLoc(), narrowedType, operand2, remainingWidth2 - widthToConsume);

    newConcatOperands.push_back(
        rewriter.createOrFold<OrOp>(op.getLoc(), extract1, extract2, false));

    consumedWidth1 += widthToConsume;
    consumedWidth2 += widthToConsume;

    if (widthToConsume == remainingWidth1) {
      ++it1;
      consumedWidth1 = 0;
    }
    if (widthToConsume == remainingWidth2) {
      ++it2;
      consumedWidth2 = 0;
    }
  }

  ConcatOp newOp = rewriter.create<ConcatOp>(op.getLoc(), newConcatOperands);

  // Copy the old operands except for concatIdx1 and concatIdx2, and append the
  // new ConcatOp to the end.
  SmallVector<Value> newOrOperands;
  newOrOperands.append(inputs.begin(), inputs.begin() + concatIdx1);
  newOrOperands.append(inputs.begin() + concatIdx1 + 1,
                       inputs.begin() + concatIdx2);
  newOrOperands.append(inputs.begin() + concatIdx2 + 1,
                       inputs.begin() + inputs.size());
  newOrOperands.push_back(newOp);

  replaceOpWithNewOpAndCopyName<OrOp>(rewriter, op, op.getType(),
                                      newOrOperands);
  return true;
}

LogicalResult OrOp::canonicalize(OrOp op, PatternRewriter &rewriter) {
  auto inputs = op.getInputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // or(..., x, ..., x, ...) -> or(..., x) -- idempotent
  // Trivial or(x), or(x, x) cases are handled by [OrOp::fold].
  if (size > 2 && canonicalizeIdempotentInputs(op, rewriter))
    return success();

  // Patterns for and with a constant on RHS.
  APInt value;
  if (matchPattern(inputs.back(), m_ConstantInt(&value))) {
    // or(..., '0) -> or(...) -- identity
    if (value.isZero()) {
      replaceOpWithNewOpAndCopyName<OrOp>(rewriter, op, op.getType(),
                                          inputs.drop_back());
      return success();
    }

    // or(..., c1, c2) -> or(..., c3) where c3 = c1 | c2 -- constant folding
    APInt value2;
    if (matchPattern(inputs[size - 2], m_ConstantInt(&value2))) {
      auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value | value2);
      SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
      newOperands.push_back(cst);
      replaceOpWithNewOpAndCopyName<OrOp>(rewriter, op, op.getType(),
                                          newOperands);
      return success();
    }

    // or(concat(x, cst1), a, b, c, cst2)
    //    ==> or(a, b, c, concat(or(x,cst2'), or(cst1,cst2'')).
    // We do this for even more multi-use concats since they are "just wiring".
    for (size_t i = 0; i < size - 1; ++i) {
      if (auto concat = inputs[i].getDefiningOp<ConcatOp>())
        if (canonicalizeLogicalCstWithConcat(op, i, value, rewriter))
          return success();
    }
  }

  // or(x, or(...)) -> or(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  // or(..., concat(x, cst1), concat(cst2, y)
  //    ==> or(..., concat(x, cst3, y)), when x and y don't overlap.
  for (size_t i = 0; i < size - 1; ++i) {
    if (auto concat = inputs[i].getDefiningOp<ConcatOp>())
      for (size_t j = i + 1; j < size; ++j)
        if (auto concat = inputs[j].getDefiningOp<ConcatOp>())
          if (canonicalizeOrOfConcatsWithCstOperands(op, i, j, rewriter))
            return success();
  }

  // extracts only of or(...) -> or(extract()...)
  if (narrowOperationWidth(op, true, rewriter))
    return success();

  // or(a[0], a[1], ..., a[n]) -> icmp ne(a, 0)
  if (auto source = getCommonOperand(op)) {
    auto cmpAgainst =
        rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getZero(size));
    replaceOpWithNewOpAndCopyName<ICmpOp>(rewriter, op, ICmpPredicate::ne,
                                          source, cmpAgainst);
    return success();
  }

  // or(mux(c_1, a, 0), mux(c_2, a, 0), ..., mux(c_n, a, 0)) -> mux(or(c_1, c_2,
  // .., c_n), a, 0)
  if (auto firstMux = op.getOperand(0).getDefiningOp<comb::MuxOp>()) {
    APInt value;
    if (op.getTwoState() && firstMux.getTwoState() &&
        matchPattern(firstMux.getFalseValue(), m_ConstantInt(&value)) &&
        value.isZero()) {
      SmallVector<Value> conditions{firstMux.getCond()};
      auto check = [&](Value v) {
        auto mux = v.getDefiningOp<comb::MuxOp>();
        if (!mux)
          return false;
        conditions.push_back(mux.getCond());
        return mux.getTwoState() &&
               firstMux.getTrueValue() == mux.getTrueValue() &&
               firstMux.getFalseValue() == mux.getFalseValue();
      };
      if (llvm::all_of(op.getOperands().drop_front(), check)) {
        auto cond = rewriter.create<comb::OrOp>(op.getLoc(), conditions, true);
        replaceOpWithNewOpAndCopyName<comb::MuxOp>(
            rewriter, op, cond, firstMux.getTrueValue(),
            firstMux.getFalseValue(), true);
        return success();
      }
    }
  }

  /// TODO: or(..., x, not(x)) -> or(..., '1) -- complement
  return failure();
}

OpFoldResult XorOp::fold(FoldAdaptor adaptor) {
  auto size = getInputs().size();
  auto inputs = adaptor.getInputs();

  // xor(x) -> x -- noop
  if (size == 1)
    return getInputs()[0];

  // xor(x, x) -> 0 -- idempotent
  if (size == 2 && getInputs()[0] == getInputs()[1])
    return IntegerAttr::get(getType(), 0);

  // xor(x, 0) -> x
  if (inputs.size() == 2 && inputs[1] &&
      inputs[1].cast<IntegerAttr>().getValue().isZero())
    return getInputs()[0];

  // xor(xor(x,1),1) -> x
  // but not self loop
  if (isBinaryNot()) {
    Value subExpr;
    if (matchPattern(getOperand(0), m_Complement(m_Any(&subExpr))) &&
        subExpr != getResult())
      return subExpr;
  }

  // Constant fold
  return constFoldAssociativeOp(inputs, hw::PEO::Xor);
}

// xor(icmp, a, b, 1) -> xor(icmp, a, b) if icmp has one user.
static void canonicalizeXorIcmpTrue(XorOp op, unsigned icmpOperand,
                                    PatternRewriter &rewriter) {
  auto icmp = op.getOperand(icmpOperand).getDefiningOp<ICmpOp>();
  auto negatedPred = ICmpOp::getNegatedPredicate(icmp.getPredicate());

  Value result =
      rewriter.create<ICmpOp>(icmp.getLoc(), negatedPred, icmp.getOperand(0),
                              icmp.getOperand(1), icmp.getTwoState());

  // If the xor had other operands, rebuild it.
  if (op.getNumOperands() > 2) {
    SmallVector<Value, 4> newOperands(op.getOperands());
    newOperands.pop_back();
    newOperands.erase(newOperands.begin() + icmpOperand);
    newOperands.push_back(result);
    result = rewriter.create<XorOp>(op.getLoc(), newOperands, op.getTwoState());
  }

  replaceOpAndCopyName(rewriter, op, result);
}

LogicalResult XorOp::canonicalize(XorOp op, PatternRewriter &rewriter) {
  auto inputs = op.getInputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // xor(..., x, x) -> xor (...) -- idempotent
  if (inputs[size - 1] == inputs[size - 2]) {
    assert(size > 2 &&
           "expected idempotent case for 2 elements handled already.");
    replaceOpWithNewOpAndCopyName<XorOp>(rewriter, op, op.getType(),
                                         inputs.drop_back(/*n=*/2), false);
    return success();
  }

  // Patterns for xor with a constant on RHS.
  APInt value;
  if (matchPattern(inputs.back(), m_ConstantInt(&value))) {
    // xor(..., 0) -> xor(...) -- identity
    if (value.isZero()) {
      replaceOpWithNewOpAndCopyName<XorOp>(rewriter, op, op.getType(),
                                           inputs.drop_back(), false);
      return success();
    }

    // xor(..., c1, c2) -> xor(..., c3) where c3 = c1 ^ c2.
    APInt value2;
    if (matchPattern(inputs[size - 2], m_ConstantInt(&value2))) {
      auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value ^ value2);
      SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
      newOperands.push_back(cst);
      replaceOpWithNewOpAndCopyName<XorOp>(rewriter, op, op.getType(),
                                           newOperands, false);
      return success();
    }

    bool isSingleBit = value.getBitWidth() == 1;

    // Check for subexpressions that we can simplify.
    for (size_t i = 0; i < size - 1; ++i) {
      Value operand = inputs[i];

      // xor(concat(x, cst1), a, b, c, cst2)
      //    ==> xor(a, b, c, concat(xor(x,cst2'), xor(cst1,cst2'')).
      // We do this for even more multi-use concats since they are "just
      // wiring".
      if (auto concat = operand.getDefiningOp<ConcatOp>())
        if (canonicalizeLogicalCstWithConcat(op, i, value, rewriter))
          return success();

      // xor(icmp, a, b, 1) -> xor(icmp, a, b) if icmp has one user.
      if (isSingleBit && operand.hasOneUse()) {
        assert(value == 1 && "single bit constant has to be one if not zero");
        if (auto icmp = operand.getDefiningOp<ICmpOp>())
          return canonicalizeXorIcmpTrue(op, i, rewriter), success();
      }
    }
  }

  // xor(x, xor(...)) -> xor(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  // extracts only of xor(...) -> xor(extract()...)
  if (narrowOperationWidth(op, true, rewriter))
    return success();

  // xor(a[0], a[1], ..., a[n]) -> parity(a)
  if (auto source = getCommonOperand(op)) {
    replaceOpWithNewOpAndCopyName<ParityOp>(rewriter, op, source);
    return success();
  }

  return failure();
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  // sub(x - x) -> 0
  if (getRhs() == getLhs())
    return getIntAttr(
        APInt::getZero(getLhs().getType().getIntOrFloatBitWidth()),
        getContext());

  if (adaptor.getRhs()) {
    // If both are constants, we can unconditionally fold.
    if (adaptor.getLhs()) {
      // Constant fold (c1 - c2) => (c1 + -1*c2).
      auto negOne = getIntAttr(
          APInt::getAllOnes(getLhs().getType().getIntOrFloatBitWidth()),
          getContext());
      auto rhsNeg = hw::ParamExprAttr::get(
          hw::PEO::Mul, adaptor.getRhs().cast<TypedAttr>(), negOne);
      return hw::ParamExprAttr::get(hw::PEO::Add,
                                    adaptor.getLhs().cast<TypedAttr>(), rhsNeg);
    }

    // sub(x - 0) -> x
    if (auto rhsC = adaptor.getRhs().dyn_cast<IntegerAttr>()) {
      if (rhsC.getValue().isZero())
        return getLhs();
    }
  }

  return {};
}

LogicalResult SubOp::canonicalize(SubOp op, PatternRewriter &rewriter) {
  // sub(x, cst) -> add(x, -cst)
  APInt value;
  if (matchPattern(op.getRhs(), m_ConstantInt(&value))) {
    auto negCst = rewriter.create<hw::ConstantOp>(op.getLoc(), -value);
    replaceOpWithNewOpAndCopyName<AddOp>(rewriter, op, op.getLhs(), negCst,
                                         false);
    return success();
  }

  // extracts only of sub(...) -> sub(extract()...)
  if (narrowOperationWidth(op, false, rewriter))
    return success();

  return failure();
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto size = getInputs().size();

  // add(x) -> x -- noop
  if (size == 1u)
    return getInputs()[0];

  // Constant fold constant operands.
  return constFoldAssociativeOp(adaptor.getOperands(), hw::PEO::Add);
}

LogicalResult AddOp::canonicalize(AddOp op, PatternRewriter &rewriter) {
  auto inputs = op.getInputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  APInt value, value2;

  // add(..., 0) -> add(...) -- identity
  if (matchPattern(inputs.back(), m_ConstantInt(&value)) && value.isZero()) {
    replaceOpWithNewOpAndCopyName<AddOp>(rewriter, op, op.getType(),
                                         inputs.drop_back(), false);
    return success();
  }

  // add(..., c1, c2) -> add(..., c3) where c3 = c1 + c2 -- constant folding
  if (matchPattern(inputs[size - 1], m_ConstantInt(&value)) &&
      matchPattern(inputs[size - 2], m_ConstantInt(&value2))) {
    auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value + value2);
    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
    newOperands.push_back(cst);
    replaceOpWithNewOpAndCopyName<AddOp>(rewriter, op, op.getType(),
                                         newOperands, false);
    return success();
  }

  // add(..., x, x) -> add(..., shl(x, 1))
  if (inputs[size - 1] == inputs[size - 2]) {
    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));

    auto one = rewriter.create<hw::ConstantOp>(op.getLoc(), op.getType(), 1);
    auto shiftLeftOp =
        rewriter.create<comb::ShlOp>(op.getLoc(), inputs.back(), one, false);

    newOperands.push_back(shiftLeftOp);
    replaceOpWithNewOpAndCopyName<AddOp>(rewriter, op, op.getType(),
                                         newOperands, false);
    return success();
  }

  auto shlOp = inputs[size - 1].getDefiningOp<comb::ShlOp>();
  // add(..., x, shl(x, c)) -> add(..., mul(x, (1 << c) + 1))
  if (shlOp && shlOp.getLhs() == inputs[size - 2] &&
      matchPattern(shlOp.getRhs(), m_ConstantInt(&value))) {

    APInt one(/*numBits=*/value.getBitWidth(), 1, /*isSigned=*/false);
    auto rhs =
        rewriter.create<hw::ConstantOp>(op.getLoc(), (one << value) + one);

    std::array<Value, 2> factors = {shlOp.getLhs(), rhs};
    auto mulOp = rewriter.create<comb::MulOp>(op.getLoc(), factors, false);

    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
    newOperands.push_back(mulOp);
    replaceOpWithNewOpAndCopyName<AddOp>(rewriter, op, op.getType(),
                                         newOperands, false);
    return success();
  }

  auto mulOp = inputs[size - 1].getDefiningOp<comb::MulOp>();
  // add(..., x, mul(x, c)) -> add(..., mul(x, c + 1))
  if (mulOp && mulOp.getInputs().size() == 2 &&
      mulOp.getInputs()[0] == inputs[size - 2] &&
      matchPattern(mulOp.getInputs()[1], m_ConstantInt(&value))) {

    APInt one(/*numBits=*/value.getBitWidth(), 1, /*isSigned=*/false);
    auto rhs = rewriter.create<hw::ConstantOp>(op.getLoc(), value + one);
    std::array<Value, 2> factors = {mulOp.getInputs()[0], rhs};
    auto newMulOp = rewriter.create<comb::MulOp>(op.getLoc(), factors, false);

    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
    newOperands.push_back(newMulOp);
    replaceOpWithNewOpAndCopyName<AddOp>(rewriter, op, op.getType(),
                                         newOperands, false);
    return success();
  }

  // add(x, add(...)) -> add(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  // extracts only of add(...) -> add(extract()...)
  if (narrowOperationWidth(op, false, rewriter))
    return success();

  // add(add(x, c1), c2) -> add(x, c1 + c2)
  auto addOp = inputs[0].getDefiningOp<comb::AddOp>();
  if (addOp && addOp.getInputs().size() == 2 &&
      matchPattern(addOp.getInputs()[1], m_ConstantInt(&value2)) &&
      inputs.size() == 2 && matchPattern(inputs[1], m_ConstantInt(&value))) {

    auto rhs = rewriter.create<hw::ConstantOp>(op.getLoc(), value + value2);
    replaceOpWithNewOpAndCopyName<AddOp>(
        rewriter, op, op.getType(), ArrayRef<Value>{addOp.getInputs()[0], rhs},
        /*twoState=*/op.getTwoState() && addOp.getTwoState());
    return success();
  }

  return failure();
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto size = getInputs().size();
  auto inputs = adaptor.getInputs();

  // mul(x) -> x -- noop
  if (size == 1u)
    return getInputs()[0];

  auto width = getType().cast<IntegerType>().getWidth();
  APInt value(/*numBits=*/width, 1, /*isSigned=*/false);

  // mul(x, 0, 1) -> 0 -- annulment
  for (auto operand : inputs) {
    if (!operand)
      continue;
    value *= operand.cast<IntegerAttr>().getValue();
    if (value.isZero())
      return getIntAttr(value, getContext());
  }

  // Constant fold
  return constFoldAssociativeOp(inputs, hw::PEO::Mul);
}

LogicalResult MulOp::canonicalize(MulOp op, PatternRewriter &rewriter) {
  auto inputs = op.getInputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  APInt value, value2;

  // mul(x, c) -> shl(x, log2(c)), where c is a power of two.
  if (size == 2 && matchPattern(inputs.back(), m_ConstantInt(&value)) &&
      value.isPowerOf2()) {
    auto shift = rewriter.create<hw::ConstantOp>(op.getLoc(), op.getType(),
                                                 value.exactLogBase2());
    auto shlOp =
        rewriter.create<comb::ShlOp>(op.getLoc(), inputs[0], shift, false);

    replaceOpWithNewOpAndCopyName<MulOp>(rewriter, op, op.getType(),
                                         ArrayRef<Value>(shlOp), false);
    return success();
  }

  // mul(..., 1) -> mul(...) -- identity
  if (matchPattern(inputs.back(), m_ConstantInt(&value)) && value.isOne()) {
    replaceOpWithNewOpAndCopyName<MulOp>(rewriter, op, op.getType(),
                                         inputs.drop_back());
    return success();
  }

  // mul(..., c1, c2) -> mul(..., c3) where c3 = c1 * c2 -- constant folding
  if (matchPattern(inputs[size - 1], m_ConstantInt(&value)) &&
      matchPattern(inputs[size - 2], m_ConstantInt(&value2))) {
    auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value * value2);
    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
    newOperands.push_back(cst);
    replaceOpWithNewOpAndCopyName<MulOp>(rewriter, op, op.getType(),
                                         newOperands);
    return success();
  }

  // mul(a, mul(...)) -> mul(a, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  // extracts only of mul(...) -> mul(extract()...)
  if (narrowOperationWidth(op, false, rewriter))
    return success();

  return failure();
}

template <class Op, bool isSigned>
static OpFoldResult foldDiv(Op op, ArrayRef<Attribute> constants) {
  if (auto rhsValue = constants[1].dyn_cast_or_null<IntegerAttr>()) {
    // divu(x, 1) -> x, divs(x, 1) -> x
    if (rhsValue.getValue() == 1)
      return op.getLhs();

    // If the divisor is zero, do not fold for now.
    if (rhsValue.getValue().isZero())
      return {};
  }

  return constFoldBinaryOp(constants, isSigned ? hw::PEO::DivS : hw::PEO::DivU);
}

OpFoldResult DivUOp::fold(FoldAdaptor adaptor) {
  return foldDiv<DivUOp, /*isSigned=*/false>(*this, adaptor.getOperands());
}

OpFoldResult DivSOp::fold(FoldAdaptor adaptor) {
  return foldDiv<DivSOp, /*isSigned=*/true>(*this, adaptor.getOperands());
}

template <class Op, bool isSigned>
static OpFoldResult foldMod(Op op, ArrayRef<Attribute> constants) {
  if (auto rhsValue = constants[1].dyn_cast_or_null<IntegerAttr>()) {
    // modu(x, 1) -> 0, mods(x, 1) -> 0
    if (rhsValue.getValue() == 1)
      return getIntAttr(APInt::getZero(op.getType().getIntOrFloatBitWidth()),
                        op.getContext());

    // If the divisor is zero, do not fold for now.
    if (rhsValue.getValue().isZero())
      return {};
  }

  if (auto lhsValue = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    // modu(0, x) -> 0, mods(0, x) -> 0
    if (lhsValue.getValue().isZero())
      return getIntAttr(APInt::getZero(op.getType().getIntOrFloatBitWidth()),
                        op.getContext());
  }

  return constFoldBinaryOp(constants, isSigned ? hw::PEO::ModS : hw::PEO::ModU);
}

OpFoldResult ModUOp::fold(FoldAdaptor adaptor) {
  return foldMod<ModUOp, /*isSigned=*/false>(*this, adaptor.getOperands());
}

OpFoldResult ModSOp::fold(FoldAdaptor adaptor) {
  return foldMod<ModSOp, /*isSigned=*/true>(*this, adaptor.getOperands());
}
//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// Constant folding
OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  if (getNumOperands() == 1)
    return getOperand(0);

  // If all the operands are constant, we can fold.
  for (auto attr : adaptor.getInputs())
    if (!attr || !attr.isa<IntegerAttr>())
      return {};

  // If we got here, we can constant fold.
  unsigned resultWidth = getType().getIntOrFloatBitWidth();
  APInt result(resultWidth, 0);

  unsigned nextInsertion = resultWidth;
  // Insert each chunk into the result.
  for (auto attr : adaptor.getInputs()) {
    auto chunk = attr.cast<IntegerAttr>().getValue();
    nextInsertion -= chunk.getBitWidth();
    result.insertBits(chunk, nextInsertion);
  }

  return getIntAttr(result, getContext());
}

LogicalResult ConcatOp::canonicalize(ConcatOp op, PatternRewriter &rewriter) {
  auto inputs = op.getInputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // This function is used when we flatten neighboring operands of a
  // (variadic) concat into a new vesion of the concat.  first/last indices
  // are inclusive.
  auto flattenConcat = [&](size_t firstOpIndex, size_t lastOpIndex,
                           ValueRange replacements) -> LogicalResult {
    SmallVector<Value, 4> newOperands;
    newOperands.append(inputs.begin(), inputs.begin() + firstOpIndex);
    newOperands.append(replacements.begin(), replacements.end());
    newOperands.append(inputs.begin() + lastOpIndex + 1, inputs.end());
    if (newOperands.size() == 1)
      replaceOpAndCopyName(rewriter, op, newOperands[0]);
    else
      replaceOpWithNewOpAndCopyName<ConcatOp>(rewriter, op, op.getType(),
                                              newOperands);
    return success();
  };

  Value commonOperand = inputs[0];
  for (size_t i = 0; i != size; ++i) {
    // Check to see if all operands are the same.
    if (inputs[i] != commonOperand)
      commonOperand = Value();

    // If an operand to the concat is itself a concat, then we can fold them
    // together.
    if (auto subConcat = inputs[i].getDefiningOp<ConcatOp>())
      return flattenConcat(i, i, subConcat->getOperands());

    // Check for canonicalization due to neighboring operands.
    if (i != 0) {
      // Merge neighboring constants.
      if (auto cst = inputs[i].getDefiningOp<hw::ConstantOp>()) {
        if (auto prevCst = inputs[i - 1].getDefiningOp<hw::ConstantOp>()) {
          unsigned prevWidth = prevCst.getValue().getBitWidth();
          unsigned thisWidth = cst.getValue().getBitWidth();
          auto resultCst = cst.getValue().zext(prevWidth + thisWidth);
          resultCst |= prevCst.getValue().zext(prevWidth + thisWidth)
                       << thisWidth;
          Value replacement =
              rewriter.create<hw::ConstantOp>(op.getLoc(), resultCst);
          return flattenConcat(i - 1, i, replacement);
        }
      }

      // If the two operands are the same, turn them into a replicate.
      if (inputs[i] == inputs[i - 1]) {
        Value replacement =
            rewriter.createOrFold<ReplicateOp>(op.getLoc(), inputs[i], 2);
        return flattenConcat(i - 1, i, replacement);
      }

      // If this input is a replicate, see if we can fold it with the previous
      // one.
      if (auto repl = inputs[i].getDefiningOp<ReplicateOp>()) {
        // ... x, repl(x, n), ...  ==> ..., repl(x, n+1), ...
        if (repl.getOperand() == inputs[i - 1]) {
          Value replacement = rewriter.createOrFold<ReplicateOp>(
              op.getLoc(), repl.getOperand(), repl.getMultiple() + 1);
          return flattenConcat(i - 1, i, replacement);
        }
        // ... repl(x, n), repl(x, m), ...  ==> ..., repl(x, n+m), ...
        if (auto prevRepl = inputs[i - 1].getDefiningOp<ReplicateOp>()) {
          if (prevRepl.getOperand() == repl.getOperand()) {
            Value replacement = rewriter.createOrFold<ReplicateOp>(
                op.getLoc(), repl.getOperand(),
                repl.getMultiple() + prevRepl.getMultiple());
            return flattenConcat(i - 1, i, replacement);
          }
        }
      }

      // ... repl(x, n), x, ...  ==> ..., repl(x, n+1), ...
      if (auto repl = inputs[i - 1].getDefiningOp<ReplicateOp>()) {
        if (repl.getOperand() == inputs[i]) {
          Value replacement = rewriter.createOrFold<ReplicateOp>(
              op.getLoc(), inputs[i], repl.getMultiple() + 1);
          return flattenConcat(i - 1, i, replacement);
        }
      }

      // Merge neighboring extracts of neighboring inputs, e.g.
      // {A[3], A[2]} -> A[3:2]
      if (auto extract = inputs[i].getDefiningOp<ExtractOp>()) {
        if (auto prevExtract = inputs[i - 1].getDefiningOp<ExtractOp>()) {
          if (extract.getInput() == prevExtract.getInput()) {
            auto thisWidth = extract.getType().cast<IntegerType>().getWidth();
            if (prevExtract.getLowBit() == extract.getLowBit() + thisWidth) {
              auto prevWidth = prevExtract.getType().getIntOrFloatBitWidth();
              auto resType = rewriter.getIntegerType(thisWidth + prevWidth);
              Value replacement = rewriter.create<ExtractOp>(
                  op.getLoc(), resType, extract.getInput(),
                  extract.getLowBit());
              return flattenConcat(i - 1, i, replacement);
            }
          }
        }
      }
      // Merge neighboring array extracts of neighboring inputs, e.g.
      // {Array[4], bitcast(Array[3:2])} -> bitcast(A[4:2])

      // This represents a slice of an array.
      struct ArraySlice {
        Value input;
        Value index;
        size_t width;
        static std::optional<ArraySlice> get(Value value) {
          assert(value.getType().isa<IntegerType>() && "expected integer type");
          if (auto arrayGet = value.getDefiningOp<hw::ArrayGetOp>())
            return ArraySlice{arrayGet.getInput(), arrayGet.getIndex(), 1};
          // array slice op is wrapped with bitcast.
          if (auto bitcast = value.getDefiningOp<hw::BitcastOp>())
            if (auto arraySlice =
                    bitcast.getInput().getDefiningOp<hw::ArraySliceOp>())
              return ArraySlice{
                  arraySlice.getInput(), arraySlice.getLowIndex(),
                  hw::type_cast<hw::ArrayType>(arraySlice.getType()).getSize()};
          return std::nullopt;
        }
      };
      if (auto extractOpt = ArraySlice::get(inputs[i])) {
        if (auto prevExtractOpt = ArraySlice::get(inputs[i - 1])) {
          // Check that two array slices are mergable.
          if (prevExtractOpt->index.getType() == extractOpt->index.getType() &&
              prevExtractOpt->input == extractOpt->input &&
              hw::isOffset(extractOpt->index, prevExtractOpt->index,
                           extractOpt->width)) {
            auto resType = hw::ArrayType::get(
                hw::type_cast<hw::ArrayType>(prevExtractOpt->input.getType())
                    .getElementType(),
                extractOpt->width + prevExtractOpt->width);
            auto resIntType = rewriter.getIntegerType(hw::getBitWidth(resType));
            Value replacement = rewriter.create<hw::BitcastOp>(
                op.getLoc(), resIntType,
                rewriter.create<hw::ArraySliceOp>(op.getLoc(), resType,
                                                  prevExtractOpt->input,
                                                  extractOpt->index));
            return flattenConcat(i - 1, i, replacement);
          }
        }
      }
    }
  }

  // If all operands were the same, then this is a replicate.
  if (commonOperand) {
    replaceOpWithNewOpAndCopyName<ReplicateOp>(rewriter, op, op.getType(),
                                               commonOperand);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// MuxOp
//===----------------------------------------------------------------------===//

OpFoldResult MuxOp::fold(FoldAdaptor adaptor) {
  // mux (c, b, b) -> b
  if (getTrueValue() == getFalseValue())
    return getTrueValue();

  // mux(0, a, b) -> b
  // mux(1, a, b) -> a
  if (auto pred = adaptor.getCond().dyn_cast_or_null<IntegerAttr>()) {
    if (pred.getValue().isZero())
      return getFalseValue();
    return getTrueValue();
  }

  // mux(cond, 1, 0) -> cond
  if (auto tv = adaptor.getTrueValue().dyn_cast_or_null<IntegerAttr>())
    if (auto fv = adaptor.getFalseValue().dyn_cast_or_null<IntegerAttr>())
      if (tv.getValue().isOne() && fv.getValue().isZero() &&
          hw::getBitWidth(getType()) == 1)
        return getCond();

  return {};
}

/// Check to see if the condition to the specified mux is an equality
/// comparison `indexValue` and one or more constants.  If so, put the
/// constants in the constants vector and return true, otherwise return false.
///
/// This is part of foldMuxChain.
///
static bool
getMuxChainCondConstant(Value cond, Value indexValue, bool isInverted,
                        std::function<void(hw::ConstantOp)> constantFn) {
  // Handle `idx == 42` and `idx != 42`.
  if (auto cmp = cond.getDefiningOp<ICmpOp>()) {
    // TODO: We could handle things like "x < 2" as two entries.
    auto requiredPredicate =
        (isInverted ? ICmpPredicate::eq : ICmpPredicate::ne);
    if (cmp.getLhs() == indexValue && cmp.getPredicate() == requiredPredicate) {
      if (auto cst = cmp.getRhs().getDefiningOp<hw::ConstantOp>()) {
        constantFn(cst);
        return true;
      }
    }
    return false;
  }

  // Handle mux(`idx == 1 || idx == 3`, value, muxchain).
  if (auto orOp = cond.getDefiningOp<OrOp>()) {
    if (!isInverted)
      return false;
    for (auto operand : orOp.getOperands())
      if (!getMuxChainCondConstant(operand, indexValue, isInverted, constantFn))
        return false;
    return true;
  }

  // Handle mux(`idx != 1 && idx != 3`, muxchain, value).
  if (auto andOp = cond.getDefiningOp<AndOp>()) {
    if (isInverted)
      return false;
    for (auto operand : andOp.getOperands())
      if (!getMuxChainCondConstant(operand, indexValue, isInverted, constantFn))
        return false;
    return true;
  }

  return false;
}

/// Given a mux, check to see if the "on true" value (or "on false" value if
/// isFalseSide=true) is a mux tree with the same condition.  This allows us
/// to turn things like `mux(VAL == 0, A, (mux (VAL == 1), B, C))` into
/// `array_get (array_create(A, B, C), VAL)` which is far more compact and
/// allows synthesis tools to do more interesting optimizations.
///
/// This returns false if we cannot form the mux tree (or do not want to) and
/// returns true if the mux was replaced.
static bool foldMuxChain(MuxOp rootMux, bool isFalseSide,
                         PatternRewriter &rewriter) {
  // Get the index value being compared.  Later we check to see if it is
  // compared to a constant with the right predicate.
  auto rootCmp = rootMux.getCond().getDefiningOp<ICmpOp>();
  if (!rootCmp)
    return false;
  Value indexValue = rootCmp.getLhs();

  // Return the value to use if the equality match succeeds.
  auto getCaseValue = [&](MuxOp mux) -> Value {
    return mux.getOperand(1 + unsigned(!isFalseSide));
  };

  // Return the value to use if the equality match fails.  This is the next
  // mux in the sequence or the "otherwise" value.
  auto getTreeValue = [&](MuxOp mux) -> Value {
    return mux.getOperand(1 + unsigned(isFalseSide));
  };

  // Start scanning the mux tree to see what we've got.  Keep track of the
  // constant comparison value and the SSA value to use when equal to it.
  SmallVector<Location> locationsFound;
  SmallVector<std::pair<hw::ConstantOp, Value>, 4> valuesFound;

  /// Extract constants and values into `valuesFound` and return true if this is
  /// part of the mux tree, otherwise return false.
  auto collectConstantValues = [&](MuxOp mux) -> bool {
    return getMuxChainCondConstant(
        mux.getCond(), indexValue, isFalseSide, [&](hw::ConstantOp cst) {
          valuesFound.push_back({cst, getCaseValue(mux)});
          locationsFound.push_back(mux.getCond().getLoc());
          locationsFound.push_back(mux->getLoc());
        });
  };

  // Make sure the root is a correct comparison with a constant.
  if (!collectConstantValues(rootMux))
    return false;

  // Make sure that we're not looking at the intermediate node in a mux tree.
  if (rootMux->hasOneUse()) {
    if (auto userMux = dyn_cast<MuxOp>(*rootMux->user_begin())) {
      if (getTreeValue(userMux) == rootMux.getResult() &&
          getMuxChainCondConstant(userMux.getCond(), indexValue, isFalseSide,
                                  [&](hw::ConstantOp cst) {}))
        return false;
    }
  }

  // Scan up the tree linearly.
  auto nextTreeValue = getTreeValue(rootMux);
  while (1) {
    auto nextMux = nextTreeValue.getDefiningOp<MuxOp>();
    if (!nextMux || !nextMux->hasOneUse())
      break;
    if (!collectConstantValues(nextMux))
      break;
    nextTreeValue = getTreeValue(nextMux);
  }

  // We need to have more than three values to create an array.  This is an
  // arbitrary threshold which is saying that one or two muxes together is ok,
  // but three should be folded.
  if (valuesFound.size() < 3)
    return false;

  // If the array is greater that 9 bits, it will take over 512 elements and
  // it will be too large for a single expression.
  auto indexWidth = indexValue.getType().cast<IntegerType>().getWidth();
  if (indexWidth >= 9)
    return false;

  // Next we need to see if the values are dense-ish.  We don't want to have
  // a tremendous number of replicated entries in the array.  Some sparsity is
  // ok though, so we require the table to be at least 5/8 utilized.
  uint64_t tableSize = 1ULL << indexWidth;
  if (valuesFound.size() < (tableSize * 5) / 8)
    return false; // Not dense enough.

  // Ok, we're going to do the transformation, start by building the table
  // filled with the "otherwise" value.
  SmallVector<Value, 8> table(tableSize, nextTreeValue);

  // Fill in entries in the table from the leaf to the root of the expression.
  // This ensures that any duplicate matches end up with the ultimate value,
  // which is the one closer to the root.
  for (auto &elt : llvm::reverse(valuesFound)) {
    uint64_t idx = elt.first.getValue().getZExtValue();
    assert(idx < table.size() && "constant should be same bitwidth as index");
    table[idx] = elt.second;
  }

  // The hw.array_create operation has the operand list in unintuitive order
  // with a[0] stored as the last element, not the first.
  std::reverse(table.begin(), table.end());

  // Build the array_create and the array_get.
  auto fusedLoc = rewriter.getFusedLoc(locationsFound);
  auto array = rewriter.create<hw::ArrayCreateOp>(fusedLoc, table);
  replaceOpWithNewOpAndCopyName<hw::ArrayGetOp>(rewriter, rootMux, array,
                                                indexValue);
  return true;
}

/// Given a fully associative variadic operation like (a+b+c+d), break the
/// expression into two parts, one without the specified operand (e.g.
/// `tmp = a+b+d`) and one that combines that into the full expression (e.g.
/// `tmp+c`), and return the inner expression.
///
/// NOTE: This mutates the operation in place if it only has a single user,
/// which assumes that user will be removed.
///
static Value extractOperandFromFullyAssociative(Operation *fullyAssoc,
                                                size_t operandNo,
                                                PatternRewriter &rewriter) {
  assert(fullyAssoc->getNumOperands() >= 2 && "cannot split up unary ops");
  assert(operandNo < fullyAssoc->getNumOperands() && "Invalid operand #");

  // If this expression already has two operands (the common case) no splitting
  // is necessary.
  if (fullyAssoc->getNumOperands() == 2)
    return fullyAssoc->getOperand(operandNo ^ 1);

  // If the operation has a single use, mutate it in place.
  if (fullyAssoc->hasOneUse()) {
    fullyAssoc->eraseOperand(operandNo);
    return fullyAssoc->getResult(0);
  }

  // Form the new operation with the operands that remain.
  SmallVector<Value> operands;
  operands.append(fullyAssoc->getOperands().begin(),
                  fullyAssoc->getOperands().begin() + operandNo);
  operands.append(fullyAssoc->getOperands().begin() + operandNo + 1,
                  fullyAssoc->getOperands().end());
  Value opWithoutExcluded = createGenericOp(
      fullyAssoc->getLoc(), fullyAssoc->getName(), operands, rewriter);
  Value excluded = fullyAssoc->getOperand(operandNo);

  Value fullResult =
      createGenericOp(fullyAssoc->getLoc(), fullyAssoc->getName(),
                      ArrayRef<Value>{opWithoutExcluded, excluded}, rewriter);
  replaceOpAndCopyName(rewriter, fullyAssoc, fullResult);
  return opWithoutExcluded;
}

/// Fold things like `mux(cond, x|y|z|a, a)` -> `(x|y|z)&replicate(cond)|a` and
/// `mux(cond, a, x|y|z|a) -> `(x|y|z)&replicate(~cond) | a` (when isTrueOperand
/// is true.  Return true on successful transformation, false if not.
///
/// These are various forms of "predicated ops" that can be handled with a
/// replicate/and combination.
static bool foldCommonMuxValue(MuxOp op, bool isTrueOperand,
                               PatternRewriter &rewriter) {
  // Check to see the operand in question is an operation.  If it is a port,
  // we can't simplify it.
  Operation *subExpr =
      (isTrueOperand ? op.getFalseValue() : op.getTrueValue()).getDefiningOp();
  if (!subExpr || subExpr->getNumOperands() < 2)
    return false;

  // If this isn't an operation we can handle, don't spend energy on it.
  if (!isa<AndOp, XorOp, OrOp, MuxOp>(subExpr))
    return false;

  // Check to see if the common value occurs in the operand list for the
  // subexpression op.  If so, then we can simplify it.
  Value commonValue = isTrueOperand ? op.getTrueValue() : op.getFalseValue();
  size_t opNo = 0, e = subExpr->getNumOperands();
  while (opNo != e && subExpr->getOperand(opNo) != commonValue)
    ++opNo;
  if (opNo == e)
    return false;

  // If we got a hit, then go ahead and simplify it!
  Value cond = op.getCond();

  // `mux(cond, a, mux(cond2, a, b))` -> `mux(cond|cond2, a, b)`
  // `mux(cond, a, mux(cond2, b, a))` -> `mux(cond|~cond2, a, b)`
  // `mux(cond, mux(cond2, a, b), a)` -> `mux(~cond|cond2, a, b)`
  // `mux(cond, mux(cond2, b, a), a)` -> `mux(~cond|~cond2, a, b)`
  if (auto subMux = dyn_cast<MuxOp>(subExpr)) {
    Value otherValue;
    Value subCond = subMux.getCond();

    // Invert th subCond if needed and dig out the 'b' value.
    if (subMux.getTrueValue() == commonValue)
      otherValue = subMux.getFalseValue();
    else if (subMux.getFalseValue() == commonValue) {
      otherValue = subMux.getTrueValue();
      subCond = createOrFoldNot(op.getLoc(), subCond, rewriter);
    } else {
      // We can't fold `mux(cond, a, mux(a, x, y))`.
      return false;
    }

    // Invert the outer cond if needed, and combine the mux conditions.
    if (!isTrueOperand)
      cond = createOrFoldNot(op.getLoc(), cond, rewriter);
    cond = rewriter.createOrFold<OrOp>(op.getLoc(), cond, subCond, false);
    replaceOpWithNewOpAndCopyName<MuxOp>(rewriter, op, cond, commonValue,
                                         otherValue, op.getTwoState());
    return true;
  }

  // Invert the condition if needed.  Or/Xor invert when dealing with
  // TrueOperand, And inverts for False operand.
  bool isaAndOp = isa<AndOp>(subExpr);
  if (isTrueOperand ^ isaAndOp)
    cond = createOrFoldNot(op.getLoc(), cond, rewriter);

  auto extendedCond =
      rewriter.createOrFold<ReplicateOp>(op.getLoc(), op.getType(), cond);

  // Cache this information before subExpr is erased by extraction below.
  bool isaXorOp = isa<XorOp>(subExpr);
  bool isaOrOp = isa<OrOp>(subExpr);

  // Handle the fully associative ops, start by pulling out the subexpression
  // from a many operand version of the op.
  auto restOfAssoc =
      extractOperandFromFullyAssociative(subExpr, opNo, rewriter);

  // `mux(cond, x|y|z|a, a)` -> `(x|y|z)&replicate(cond) | a`
  // `mux(cond, x^y^z^a, a)` -> `(x^y^z)&replicate(cond) ^ a`
  if (isaOrOp || isaXorOp) {
    auto masked = rewriter.createOrFold<AndOp>(op.getLoc(), extendedCond,
                                               restOfAssoc, false);
    if (isaXorOp)
      replaceOpWithNewOpAndCopyName<XorOp>(rewriter, op, masked, commonValue,
                                           false);
    else
      replaceOpWithNewOpAndCopyName<OrOp>(rewriter, op, masked, commonValue,
                                          false);
    return true;
  }

  // `mux(cond, a, x&y&z&a)` -> `((x&y&z)|replicate(cond)) & a`
  assert(isaAndOp && "unexpected operation here");
  auto masked = rewriter.createOrFold<OrOp>(op.getLoc(), extendedCond,
                                            restOfAssoc, false);
  replaceOpWithNewOpAndCopyName<AndOp>(rewriter, op, masked, commonValue,
                                       false);
  return true;
}

/// This function is invoke when we find a mux with true/false operations that
/// have the same opcode.  Check to see if we can strength reduce the mux by
/// applying it to less data by applying this transformation:
///   `mux(cond, op(a, b), op(a, c))` -> `op(a, mux(cond, b, c))`
static bool foldCommonMuxOperation(MuxOp mux, Operation *trueOp,
                                   Operation *falseOp,
                                   PatternRewriter &rewriter) {
  // Right now we only apply to concat.
  // TODO: Generalize this to and, or, xor, icmp(!), which all occur in practice
  if (!isa<ConcatOp>(trueOp))
    return false;

  // Decode the operands, looking through recursive concats and replicates.
  SmallVector<Value> trueOperands, falseOperands;
  getConcatOperands(trueOp->getResult(0), trueOperands);
  getConcatOperands(falseOp->getResult(0), falseOperands);

  size_t numTrueOperands = trueOperands.size();
  size_t numFalseOperands = falseOperands.size();

  if (!numTrueOperands || !numFalseOperands ||
      (trueOperands.front() != falseOperands.front() &&
       trueOperands.back() != falseOperands.back()))
    return false;

  // Pull all leading shared operands out into their own op if any are common.
  if (trueOperands.front() == falseOperands.front()) {
    SmallVector<Value> operands;
    size_t i;
    for (i = 0; i < numTrueOperands; ++i) {
      Value trueOperand = trueOperands[i];
      if (trueOperand == falseOperands[i])
        operands.push_back(trueOperand);
      else
        break;
    }
    if (i == numTrueOperands) {
      // Selecting between distinct, but lexically identical, concats.
      replaceOpAndCopyName(rewriter, mux, trueOp->getResult(0));
      return true;
    }

    Value sharedMSB;
    if (llvm::all_of(operands, [&](Value v) { return v == operands.front(); }))
      sharedMSB = rewriter.createOrFold<ReplicateOp>(
          mux->getLoc(), operands.front(), operands.size());
    else
      sharedMSB = rewriter.createOrFold<ConcatOp>(mux->getLoc(), operands);
    operands.clear();

    // Get a concat of the LSB's on each side.
    operands.append(trueOperands.begin() + i, trueOperands.end());
    Value trueLSB = rewriter.createOrFold<ConcatOp>(trueOp->getLoc(), operands);
    operands.clear();
    operands.append(falseOperands.begin() + i, falseOperands.end());
    Value falseLSB =
        rewriter.createOrFold<ConcatOp>(falseOp->getLoc(), operands);
    // Merge the LSBs with a new mux and concat the MSB with the LSB to be
    // done.
    Value lsb = rewriter.createOrFold<MuxOp>(
        mux->getLoc(), mux.getCond(), trueLSB, falseLSB, mux.getTwoState());
    replaceOpWithNewOpAndCopyName<ConcatOp>(rewriter, mux, sharedMSB, lsb);
    return true;
  }

  // If trailing operands match, try to commonize them.
  if (trueOperands.back() == falseOperands.back()) {
    SmallVector<Value> operands;
    size_t i;
    for (i = 0;; ++i) {
      Value trueOperand = trueOperands[numTrueOperands - i - 1];
      if (trueOperand == falseOperands[numFalseOperands - i - 1])
        operands.push_back(trueOperand);
      else
        break;
    }
    std::reverse(operands.begin(), operands.end());
    Value sharedLSB = rewriter.createOrFold<ConcatOp>(mux->getLoc(), operands);
    operands.clear();

    // Get a concat of the MSB's on each side.
    operands.append(trueOperands.begin(), trueOperands.end() - i);
    Value trueMSB = rewriter.createOrFold<ConcatOp>(trueOp->getLoc(), operands);
    operands.clear();
    operands.append(falseOperands.begin(), falseOperands.end() - i);
    Value falseMSB =
        rewriter.createOrFold<ConcatOp>(falseOp->getLoc(), operands);
    // Merge the MSBs with a new mux and concat the MSB with the LSB to be done.
    Value msb = rewriter.createOrFold<MuxOp>(
        mux->getLoc(), mux.getCond(), trueMSB, falseMSB, mux.getTwoState());
    replaceOpWithNewOpAndCopyName<ConcatOp>(rewriter, mux, msb, sharedLSB);
    return true;
  }

  return false;
}

// If both arguments of the mux are arrays with the same elements, sink the
// mux and return a uniform array initializing all elements to it.
static bool foldMuxOfUniformArrays(MuxOp op, PatternRewriter &rewriter) {
  auto trueVec = op.getTrueValue().getDefiningOp<hw::ArrayCreateOp>();
  auto falseVec = op.getFalseValue().getDefiningOp<hw::ArrayCreateOp>();
  if (!trueVec || !falseVec)
    return false;
  if (!trueVec.isUniform() || !falseVec.isUniform())
    return false;

  auto mux = rewriter.create<MuxOp>(
      op.getLoc(), op.getCond(), trueVec.getUniformElement(),
      falseVec.getUniformElement(), op.getTwoState());

  SmallVector<Value> values(trueVec.getInputs().size(), mux);
  rewriter.replaceOpWithNewOp<hw::ArrayCreateOp>(op, values);
  return true;
}

namespace {
struct MuxRewriter : public mlir::OpRewritePattern<MuxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MuxOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult MuxRewriter::matchAndRewrite(MuxOp op,
                                           PatternRewriter &rewriter) const {
  // If the op has a SV attribute, don't optimize it.
  if (hasSVAttributes(op))
    return failure();
  APInt value;

  if (matchPattern(op.getTrueValue(), m_ConstantInt(&value))) {
    if (value.getBitWidth() == 1) {
      // mux(a, 0, b) -> and(~a, b) for single-bit values.
      if (value.isZero()) {
        auto notCond = createOrFoldNot(op.getLoc(), op.getCond(), rewriter);
        replaceOpWithNewOpAndCopyName<AndOp>(rewriter, op, notCond,
                                             op.getFalseValue(), false);
        return success();
      }

      // mux(a, 1, b) -> or(a, b) for single-bit values.
      replaceOpWithNewOpAndCopyName<OrOp>(rewriter, op, op.getCond(),
                                          op.getFalseValue(), false);
      return success();
    }

    // Check for mux of two constants.  There are many ways to simplify them.
    APInt value2;
    if (matchPattern(op.getFalseValue(), m_ConstantInt(&value2))) {
      // When both inputs are constants and differ by only one bit, we can
      // simplify by splitting the mux into up to three contiguous chunks: one
      // for the differing bit and up to two for the bits that are the same.
      // E.g. mux(a, 3'h2, 0) -> concat(0, mux(a, 1, 0), 0) -> concat(0, a, 0)
      APInt xorValue = value ^ value2;
      if (xorValue.isPowerOf2()) {
        unsigned leadingZeros = xorValue.countLeadingZeros();
        unsigned trailingZeros = value.getBitWidth() - leadingZeros - 1;
        SmallVector<Value, 3> operands;

        // Concat operands go from MSB to LSB, so we handle chunks in reverse
        // order of bit indexes.
        // For the chunks that are identical (i.e. correspond to 0s in
        // xorValue), we can extract directly from either input value, and we
        // arbitrarily pick the trueValue().

        if (leadingZeros > 0)
          operands.push_back(rewriter.createOrFold<ExtractOp>(
              op.getLoc(), op.getTrueValue(), trailingZeros + 1, leadingZeros));

        // Handle the differing bit, which should simplify into either cond or
        // ~cond.
        auto v1 = rewriter.createOrFold<ExtractOp>(
            op.getLoc(), op.getTrueValue(), trailingZeros, 1);
        auto v2 = rewriter.createOrFold<ExtractOp>(
            op.getLoc(), op.getFalseValue(), trailingZeros, 1);
        operands.push_back(rewriter.createOrFold<MuxOp>(
            op.getLoc(), op.getCond(), v1, v2, false));

        if (trailingZeros > 0)
          operands.push_back(rewriter.createOrFold<ExtractOp>(
              op.getLoc(), op.getTrueValue(), 0, trailingZeros));

        replaceOpWithNewOpAndCopyName<ConcatOp>(rewriter, op, op.getType(),
                                                operands);
        return success();
      }

      // If the true value is all ones and the false is all zeros then we have a
      // replicate pattern.
      if (value.isAllOnes() && value2.isZero()) {
        replaceOpWithNewOpAndCopyName<ReplicateOp>(rewriter, op, op.getType(),
                                                   op.getCond());
        return success();
      }
    }
  }

  if (matchPattern(op.getFalseValue(), m_ConstantInt(&value)) &&
      value.getBitWidth() == 1) {
    // mux(a, b, 0) -> and(a, b) for single-bit values.
    if (value.isZero()) {
      replaceOpWithNewOpAndCopyName<AndOp>(rewriter, op, op.getCond(),
                                           op.getTrueValue(), false);
      return success();
    }

    // mux(a, b, 1) -> or(~a, b) for single-bit values.
    // falseValue() is known to be a single-bit 1, which we can use for
    // the 1 in the representation of ~ using xor.
    auto notCond = rewriter.createOrFold<XorOp>(op.getLoc(), op.getCond(),
                                                op.getFalseValue(), false);
    replaceOpWithNewOpAndCopyName<OrOp>(rewriter, op, notCond,
                                        op.getTrueValue(), false);
    return success();
  }

  // mux(!a, b, c) -> mux(a, c, b)
  Value subExpr;
  Operation *condOp = op.getCond().getDefiningOp();
  if (condOp && matchPattern(condOp, m_Complement(m_Any(&subExpr))) &&
      op.getTwoState()) {
    replaceOpWithNewOpAndCopyName<MuxOp>(rewriter, op, op.getType(), subExpr,
                                         op.getFalseValue(), op.getTrueValue(),
                                         true);
    return success();
  }

  // Same but with Demorgan's law.
  // mux(and(~a, ~b, ~c), x, y) -> mux(or(a, b, c), y, x)
  // mux(or(~a, ~b, ~c), x, y) -> mux(and(a, b, c), y, x)
  if (condOp && condOp->hasOneUse()) {
    SmallVector<Value> invertedOperands;

    /// Scan all the operands to see if they are complemented.  If so, build a
    /// vector of them and return true, otherwise return false.
    auto getInvertedOperands = [&]() -> bool {
      for (Value operand : condOp->getOperands()) {
        if (matchPattern(operand, m_Complement(m_Any(&subExpr))))
          invertedOperands.push_back(subExpr);
        else
          return false;
      }
      return true;
    };

    if (isa<AndOp>(condOp) && getInvertedOperands()) {
      auto newOr =
          rewriter.createOrFold<OrOp>(op.getLoc(), invertedOperands, false);
      replaceOpWithNewOpAndCopyName<MuxOp>(rewriter, op, newOr,
                                           op.getFalseValue(),
                                           op.getTrueValue(), op.getTwoState());
      return success();
    }
    if (isa<OrOp>(condOp) && getInvertedOperands()) {
      auto newAnd =
          rewriter.createOrFold<AndOp>(op.getLoc(), invertedOperands, false);
      replaceOpWithNewOpAndCopyName<MuxOp>(rewriter, op, newAnd,
                                           op.getFalseValue(),
                                           op.getTrueValue(), op.getTwoState());
      return success();
    }
  }

  if (auto falseMux =
          dyn_cast_or_null<MuxOp>(op.getFalseValue().getDefiningOp())) {
    // mux(selector, x, mux(selector, y, z) = mux(selector, x, z)
    if (op.getCond() == falseMux.getCond()) {
      replaceOpWithNewOpAndCopyName<MuxOp>(
          rewriter, op, op.getCond(), op.getTrueValue(),
          falseMux.getFalseValue(), op.getTwoStateAttr());
      return success();
    }

    // Check to see if we can fold a mux tree into an array_create/get pair.
    if (foldMuxChain(op, /*isFalse*/ true, rewriter))
      return success();
  }

  if (auto trueMux =
          dyn_cast_or_null<MuxOp>(op.getTrueValue().getDefiningOp())) {
    // mux(selector, mux(selector, a, b), c) = mux(selector, a, c)
    if (op.getCond() == trueMux.getCond()) {
      replaceOpWithNewOpAndCopyName<MuxOp>(
          rewriter, op, op.getCond(), trueMux.getTrueValue(),
          op.getFalseValue(), op.getTwoStateAttr());
      return success();
    }

    // Check to see if we can fold a mux tree into an array_create/get pair.
    if (foldMuxChain(op, /*isFalseSide*/ false, rewriter))
      return success();
  }

  // mux(c1, mux(c2, a, b), mux(c2, a, c)) -> mux(c2, a, mux(c1, b, c))
  if (auto trueMux = dyn_cast_or_null<MuxOp>(op.getTrueValue().getDefiningOp()),
      falseMux = dyn_cast_or_null<MuxOp>(op.getFalseValue().getDefiningOp());
      trueMux && falseMux && trueMux.getCond() == falseMux.getCond() &&
      trueMux.getTrueValue() == falseMux.getTrueValue()) {
    auto subMux = rewriter.create<MuxOp>(
        rewriter.getFusedLoc({trueMux.getLoc(), falseMux.getLoc()}),
        op.getCond(), trueMux.getFalseValue(), falseMux.getFalseValue());
    replaceOpWithNewOpAndCopyName<MuxOp>(rewriter, op, trueMux.getCond(),
                                         trueMux.getTrueValue(), subMux,
                                         op.getTwoStateAttr());
    return success();
  }

  // mux(c1, mux(c2, a, b), mux(c2, c, b)) -> mux(c2, mux(c1, a, c), b)
  if (auto trueMux = dyn_cast_or_null<MuxOp>(op.getTrueValue().getDefiningOp()),
      falseMux = dyn_cast_or_null<MuxOp>(op.getFalseValue().getDefiningOp());
      trueMux && falseMux && trueMux.getCond() == falseMux.getCond() &&
      trueMux.getFalseValue() == falseMux.getFalseValue()) {
    auto subMux = rewriter.create<MuxOp>(
        rewriter.getFusedLoc({trueMux.getLoc(), falseMux.getLoc()}),
        op.getCond(), trueMux.getTrueValue(), falseMux.getTrueValue());
    replaceOpWithNewOpAndCopyName<MuxOp>(rewriter, op, trueMux.getCond(),
                                         subMux, trueMux.getFalseValue(),
                                         op.getTwoStateAttr());
    return success();
  }

  // mux(c1, mux(c2, a, b), mux(c3, a, b)) -> mux(mux(c1, c2, c3), a, b)
  if (auto trueMux = dyn_cast_or_null<MuxOp>(op.getTrueValue().getDefiningOp()),
      falseMux = dyn_cast_or_null<MuxOp>(op.getFalseValue().getDefiningOp());
      trueMux && falseMux &&
      trueMux.getTrueValue() == falseMux.getTrueValue() &&
      trueMux.getFalseValue() == falseMux.getFalseValue()) {
    auto subMux = rewriter.create<MuxOp>(
        rewriter.getFusedLoc(
            {op.getLoc(), trueMux.getLoc(), falseMux.getLoc()}),
        op.getCond(), trueMux.getCond(), falseMux.getCond());
    replaceOpWithNewOpAndCopyName<MuxOp>(
        rewriter, op, subMux, trueMux.getTrueValue(), trueMux.getFalseValue(),
        op.getTwoStateAttr());
    return success();
  }

  // mux(cond, x|y|z|a, a) -> (x|y|z)&replicate(cond) | a
  if (foldCommonMuxValue(op, false, rewriter))
    return success();
  // mux(cond, a, x|y|z|a) -> (x|y|z)&replicate(~cond) | a
  if (foldCommonMuxValue(op, true, rewriter))
    return success();

  // `mux(cond, op(a, b), op(a, c))` -> `op(a, mux(cond, b, c))`
  if (Operation *trueOp = op.getTrueValue().getDefiningOp())
    if (Operation *falseOp = op.getFalseValue().getDefiningOp())
      if (trueOp->getName() == falseOp->getName())
        if (foldCommonMuxOperation(op, trueOp, falseOp, rewriter))
          return success();

  // extracts only of mux(...) -> mux(extract()...)
  if (narrowOperationWidth(op, true, rewriter))
    return success();

  // mux(cond, repl(n, a1), repl(n, a2)) -> repl(n, mux(cond, a1, a2))
  if (foldMuxOfUniformArrays(op, rewriter))
    return success();

  return failure();
}

static bool foldArrayOfMuxes(hw::ArrayCreateOp op, PatternRewriter &rewriter) {
  // Do not fold uniform or singleton arrays to avoid duplicating muxes.
  if (op.getInputs().empty() || op.isUniform())
    return false;
  auto inputs = op.getInputs();
  if (inputs.size() <= 1)
    return false;

  // Check the operands to the array create.  Ensure all of them are the
  // same op with the same number of operands.
  auto first = inputs[0].getDefiningOp<comb::MuxOp>();
  if (!first || hasSVAttributes(first))
    return false;

  // Check whether all operands are muxes with the same condition.
  for (size_t i = 1, n = inputs.size(); i < n; ++i) {
    auto input = inputs[i].getDefiningOp<comb::MuxOp>();
    if (!input || first.getCond() != input.getCond())
      return false;
  }

  // Collect the true and the false branches into arrays.
  SmallVector<Value> trues{first.getTrueValue()};
  SmallVector<Value> falses{first.getFalseValue()};
  SmallVector<Location> locs{first->getLoc()};
  bool isTwoState = true;
  for (size_t i = 1, n = inputs.size(); i < n; ++i) {
    auto input = inputs[i].getDefiningOp<comb::MuxOp>();
    trues.push_back(input.getTrueValue());
    falses.push_back(input.getFalseValue());
    locs.push_back(input->getLoc());
    if (!input.getTwoState())
      isTwoState = false;
  }

  // Define the location of the array create as the aggregate of all muxes.
  auto loc = FusedLoc::get(op.getContext(), locs);

  // Replace the create with an aggregate operation.  Push the create op
  // into the operands of the aggregate operation.
  auto arrayTy = op.getType();
  auto trueValues = rewriter.create<hw::ArrayCreateOp>(loc, arrayTy, trues);
  auto falseValues = rewriter.create<hw::ArrayCreateOp>(loc, arrayTy, falses);
  rewriter.replaceOpWithNewOp<comb::MuxOp>(op, arrayTy, first.getCond(),
                                           trueValues, falseValues, isTwoState);
  return true;
}

struct ArrayRewriter : public mlir::OpRewritePattern<hw::ArrayCreateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(hw::ArrayCreateOp op,
                                PatternRewriter &rewriter) const override {
    if (foldArrayOfMuxes(op, rewriter))
      return success();
    return failure();
  }
};

} // namespace

void MuxOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<MuxRewriter, ArrayRewriter>(context);
}

//===----------------------------------------------------------------------===//
// ICmpOp
//===----------------------------------------------------------------------===//

// Calculate the result of a comparison when the LHS and RHS are both
// constants.
static bool applyCmpPredicate(ICmpPredicate predicate, const APInt &lhs,
                              const APInt &rhs) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return lhs.eq(rhs);
  case ICmpPredicate::ne:
    return lhs.ne(rhs);
  case ICmpPredicate::slt:
    return lhs.slt(rhs);
  case ICmpPredicate::sle:
    return lhs.sle(rhs);
  case ICmpPredicate::sgt:
    return lhs.sgt(rhs);
  case ICmpPredicate::sge:
    return lhs.sge(rhs);
  case ICmpPredicate::ult:
    return lhs.ult(rhs);
  case ICmpPredicate::ule:
    return lhs.ule(rhs);
  case ICmpPredicate::ugt:
    return lhs.ugt(rhs);
  case ICmpPredicate::uge:
    return lhs.uge(rhs);
  case ICmpPredicate::ceq:
    return lhs.eq(rhs);
  case ICmpPredicate::cne:
    return lhs.ne(rhs);
  case ICmpPredicate::weq:
    return lhs.eq(rhs);
  case ICmpPredicate::wne:
    return lhs.ne(rhs);
  }
  llvm_unreachable("unknown comparison predicate");
}

// Returns the result of applying the predicate when the LHS and RHS are the
// exact same value.
static bool applyCmpPredicateToEqualOperands(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
  case ICmpPredicate::sle:
  case ICmpPredicate::sge:
  case ICmpPredicate::ule:
  case ICmpPredicate::uge:
  case ICmpPredicate::ceq:
  case ICmpPredicate::weq:
    return true;
  case ICmpPredicate::ne:
  case ICmpPredicate::slt:
  case ICmpPredicate::sgt:
  case ICmpPredicate::ult:
  case ICmpPredicate::ugt:
  case ICmpPredicate::cne:
  case ICmpPredicate::wne:
    return false;
  }
  llvm_unreachable("unknown comparison predicate");
}

OpFoldResult ICmpOp::fold(FoldAdaptor adaptor) {
  // gt a, a -> false
  // gte a, a -> true
  if (getLhs() == getRhs()) {
    auto val = applyCmpPredicateToEqualOperands(getPredicate());
    return IntegerAttr::get(getType(), val);
  }

  // gt 1, 2 -> false
  if (auto lhs = adaptor.getLhs().dyn_cast_or_null<IntegerAttr>()) {
    if (auto rhs = adaptor.getRhs().dyn_cast_or_null<IntegerAttr>()) {
      auto val =
          applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
      return IntegerAttr::get(getType(), val);
    }
  }
  return {};
}

// Given a range of operands, computes the number of matching prefix and
// suffix elements. This does not perform cross-element matching.
template <typename Range>
static size_t computeCommonPrefixLength(const Range &a, const Range &b) {
  size_t commonPrefixLength = 0;
  auto ia = a.begin();
  auto ib = b.begin();

  for (; ia != a.end() && ib != b.end(); ia++, ib++, commonPrefixLength++) {
    if (*ia != *ib) {
      break;
    }
  }

  return commonPrefixLength;
}

static size_t getTotalWidth(ArrayRef<Value> operands) {
  size_t totalWidth = 0;
  for (auto operand : operands) {
    // getIntOrFloatBitWidth should never raise, since all arguments to
    // ConcatOp are integers.
    ssize_t width = operand.getType().getIntOrFloatBitWidth();
    assert(width >= 0);
    totalWidth += width;
  }
  return totalWidth;
}

/// Reduce the strength icmp(concat(...), concat(...)) by doing a element-wise
/// comparison on common prefix and suffixes. Returns success() if a rewriting
/// happens.  This handles both concat and replicate.
static LogicalResult matchAndRewriteCompareConcat(ICmpOp op, Operation *lhs,
                                                  Operation *rhs,
                                                  PatternRewriter &rewriter) {
  // It is safe to assume that [{lhsOperands, rhsOperands}.size() > 0] and
  // all elements have non-zero length. Both these invariants are verified
  // by the ConcatOp verifier.
  SmallVector<Value> lhsOperands, rhsOperands;
  getConcatOperands(lhs->getResult(0), lhsOperands);
  getConcatOperands(rhs->getResult(0), rhsOperands);
  ArrayRef<Value> lhsOperandsRef = lhsOperands, rhsOperandsRef = rhsOperands;

  auto formCatOrReplicate = [&](Location loc,
                                ArrayRef<Value> operands) -> Value {
    assert(!operands.empty());
    Value sameElement = operands[0];
    for (size_t i = 1, e = operands.size(); i != e && sameElement; ++i)
      if (sameElement != operands[i])
        sameElement = Value();
    if (sameElement)
      return rewriter.createOrFold<ReplicateOp>(loc, sameElement,
                                                operands.size());
    return rewriter.createOrFold<ConcatOp>(loc, operands);
  };

  auto replaceWith = [&](ICmpPredicate predicate, Value lhs,
                         Value rhs) -> LogicalResult {
    replaceOpWithNewOpAndCopyName<ICmpOp>(rewriter, op, predicate, lhs, rhs,
                                          op.getTwoState());
    return success();
  };

  size_t commonPrefixLength =
      computeCommonPrefixLength(lhsOperands, rhsOperands);
  if (commonPrefixLength == lhsOperands.size()) {
    // cat(a, b, c) == cat(a, b, c) -> 1
    bool result = applyCmpPredicateToEqualOperands(op.getPredicate());
    replaceOpWithNewOpAndCopyName<hw::ConstantOp>(rewriter, op,
                                                  APInt(1, result));
    return success();
  }

  size_t commonSuffixLength = computeCommonPrefixLength(
      llvm::reverse(lhsOperandsRef), llvm::reverse(rhsOperandsRef));

  size_t commonPrefixTotalWidth =
      getTotalWidth(lhsOperandsRef.take_front(commonPrefixLength));
  size_t commonSuffixTotalWidth =
      getTotalWidth(lhsOperandsRef.take_back(commonSuffixLength));
  auto lhsOnly = lhsOperandsRef.drop_front(commonPrefixLength)
                     .drop_back(commonSuffixLength);
  auto rhsOnly = rhsOperandsRef.drop_front(commonPrefixLength)
                     .drop_back(commonSuffixLength);

  auto replaceWithoutReplicatingSignBit = [&]() {
    auto newLhs = formCatOrReplicate(lhs->getLoc(), lhsOnly);
    auto newRhs = formCatOrReplicate(rhs->getLoc(), rhsOnly);
    return replaceWith(op.getPredicate(), newLhs, newRhs);
  };

  auto replaceWithReplicatingSignBit = [&]() {
    auto firstNonEmptyValue = lhsOperands[0];
    auto firstNonEmptyElemWidth =
        firstNonEmptyValue.getType().getIntOrFloatBitWidth();
    Value signBit = rewriter.createOrFold<ExtractOp>(
        op.getLoc(), firstNonEmptyValue, firstNonEmptyElemWidth - 1, 1);

    auto newLhs = rewriter.create<ConcatOp>(lhs->getLoc(), signBit, lhsOnly);
    auto newRhs = rewriter.create<ConcatOp>(rhs->getLoc(), signBit, rhsOnly);
    return replaceWith(op.getPredicate(), newLhs, newRhs);
  };

  if (ICmpOp::isPredicateSigned(op.getPredicate())) {
    // scmp(cat(..x, b), cat(..y, b)) == scmp(cat(..x), cat(..y))
    if (commonPrefixTotalWidth == 0 && commonSuffixTotalWidth > 0)
      return replaceWithoutReplicatingSignBit();

    // scmp(cat(a, ..x, b), cat(a, ..y, b)) == scmp(cat(sgn(a), ..x),
    // cat(sgn(b), ..y)) Note that we cannot perform this optimization if
    // [width(b) = 0 && width(a) <= 1]. since that common prefix is the sign
    // bit. Doing the rewrite can result in an infinite loop.
    if (commonPrefixTotalWidth > 1 || commonSuffixTotalWidth > 0)
      return replaceWithReplicatingSignBit();

  } else if (commonPrefixTotalWidth > 0 || commonSuffixTotalWidth > 0) {
    // ucmp(cat(a, ..x, b), cat(a, ..y, b)) = ucmp(cat(..x), cat(..y))
    return replaceWithoutReplicatingSignBit();
  }

  return failure();
}

/// Given an equality comparison with a constant value and some operand that has
/// known bits, simplify the comparison to check only the unknown bits of the
/// input.
///
/// One simple example of this is that `concat(0, stuff) == 0` can be simplified
/// to `stuff == 0`, or `and(x, 3) == 0` can be simplified to
/// `extract x[1:0] == 0`
static void combineEqualityICmpWithKnownBitsAndConstant(
    ICmpOp cmpOp, const KnownBits &bitAnalysis, const APInt &rhsCst,
    PatternRewriter &rewriter) {

  // If any of the known bits disagree with any of the comparison bits, then
  // we can constant fold this comparison right away.
  APInt bitsKnown = bitAnalysis.Zero | bitAnalysis.One;
  if ((bitsKnown & rhsCst) != bitAnalysis.One) {
    // If we discover a mismatch then we know an "eq" comparison is false
    // and a "ne" comparison is true!
    bool result = cmpOp.getPredicate() == ICmpPredicate::ne;
    replaceOpWithNewOpAndCopyName<hw::ConstantOp>(rewriter, cmpOp,
                                                  APInt(1, result));
    return;
  }

  // Check to see if we can prove the result entirely of the comparison (in
  // which we bail out early), otherwise build a list of values to concat and a
  // smaller constant to compare against.
  SmallVector<Value> newConcatOperands;
  auto newConstant = APInt::getZeroWidth();

  // Ok, some (maybe all) bits are known and some others may be unknown.
  // Extract out segments of the operand and compare against the
  // corresponding bits.
  unsigned knownMSB = bitsKnown.countLeadingOnes();

  Value operand = cmpOp.getLhs();

  // Ok, some bits are known but others are not.  Extract out sequences of
  // bits that are unknown and compare just those bits.  We work from MSB to
  // LSB.
  while (knownMSB != bitsKnown.getBitWidth()) {
    // Drop any high bits that are known.
    if (knownMSB)
      bitsKnown = bitsKnown.trunc(bitsKnown.getBitWidth() - knownMSB);

    // Find the span of unknown bits, and extract it.
    unsigned unknownBits = bitsKnown.countLeadingZeros();
    unsigned lowBit = bitsKnown.getBitWidth() - unknownBits;
    auto spanOperand = rewriter.createOrFold<ExtractOp>(
        operand.getLoc(), operand, /*lowBit=*/lowBit,
        /*bitWidth=*/unknownBits);
    auto spanConstant = rhsCst.lshr(lowBit).trunc(unknownBits);

    // Add this info to the concat we're generating.
    newConcatOperands.push_back(spanOperand);
    // FIXME(llvm merge, cc697fc292b0): concat doesn't work with zero bit values
    // newConstant = newConstant.concat(spanConstant);
    if (newConstant.getBitWidth() != 0)
      newConstant = newConstant.concat(spanConstant);
    else
      newConstant = spanConstant;

    // Drop the unknown bits in prep for the next chunk.
    unsigned newWidth = bitsKnown.getBitWidth() - unknownBits;
    bitsKnown = bitsKnown.trunc(newWidth);
    knownMSB = bitsKnown.countLeadingOnes();
  }

  // If all the operands to the concat are foldable then we have an identity
  // situation where all the sub-elements equal each other.  This implies that
  // the overall result is foldable.
  if (newConcatOperands.empty()) {
    bool result = cmpOp.getPredicate() == ICmpPredicate::eq;
    replaceOpWithNewOpAndCopyName<hw::ConstantOp>(rewriter, cmpOp,
                                                  APInt(1, result));
    return;
  }

  // If we have a single operand remaining, use it, otherwise form a concat.
  Value concatResult =
      rewriter.createOrFold<ConcatOp>(operand.getLoc(), newConcatOperands);

  // Form the comparison against the smaller constant.
  auto newConstantOp = rewriter.create<hw::ConstantOp>(
      cmpOp.getOperand(1).getLoc(), newConstant);

  replaceOpWithNewOpAndCopyName<ICmpOp>(rewriter, cmpOp, cmpOp.getPredicate(),
                                        concatResult, newConstantOp,
                                        cmpOp.getTwoState());
}

// Simplify icmp eq(xor(a,b,cst1), cst2) -> icmp eq(xor(a,b), cst1^cst2).
static void combineEqualityICmpWithXorOfConstant(ICmpOp cmpOp, XorOp xorOp,
                                                 const APInt &rhs,
                                                 PatternRewriter &rewriter) {
  auto ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(xorOp);

  auto xorRHS = xorOp.getOperands().back().getDefiningOp<hw::ConstantOp>();
  auto newRHS = rewriter.create<hw::ConstantOp>(xorRHS->getLoc(),
                                                xorRHS.getValue() ^ rhs);
  Value newLHS;
  switch (xorOp.getNumOperands()) {
  case 1:
    // This isn't common but is defined so we need to handle it.
    newLHS = rewriter.create<hw::ConstantOp>(xorOp.getLoc(),
                                             APInt::getZero(rhs.getBitWidth()));
    break;
  case 2:
    // The binary case is the most common.
    newLHS = xorOp.getOperand(0);
    break;
  default:
    // The general case forces us to form a new xor with the remaining operands.
    SmallVector<Value> newOperands(xorOp.getOperands());
    newOperands.pop_back();
    newLHS = rewriter.create<XorOp>(xorOp.getLoc(), newOperands, false);
    break;
  }

  bool xorMultipleUses = !xorOp->hasOneUse();

  // If the xor has multiple uses (not just the compare, then we need/want to
  // replace them as well.
  if (xorMultipleUses)
    replaceOpWithNewOpAndCopyName<XorOp>(rewriter, xorOp, newLHS, xorRHS,
                                         false);

  // Replace the comparison.
  rewriter.restoreInsertionPoint(ip);
  replaceOpWithNewOpAndCopyName<ICmpOp>(rewriter, cmpOp, cmpOp.getPredicate(),
                                        newLHS, newRHS, false);
}

LogicalResult ICmpOp::canonicalize(ICmpOp op, PatternRewriter &rewriter) {
  APInt lhs, rhs;

  // icmp 1, x -> icmp x, 1
  if (matchPattern(op.getLhs(), m_ConstantInt(&lhs))) {
    assert(!matchPattern(op.getRhs(), m_ConstantInt(&rhs)) &&
           "Should be folded");
    replaceOpWithNewOpAndCopyName<ICmpOp>(
        rewriter, op, ICmpOp::getFlippedPredicate(op.getPredicate()),
        op.getRhs(), op.getLhs(), op.getTwoState());
    return success();
  }

  // Canonicalize with RHS constant
  if (matchPattern(op.getRhs(), m_ConstantInt(&rhs))) {
    auto getConstant = [&](APInt constant) -> Value {
      return rewriter.create<hw::ConstantOp>(op.getLoc(), std::move(constant));
    };

    auto replaceWith = [&](ICmpPredicate predicate, Value lhs,
                           Value rhs) -> LogicalResult {
      replaceOpWithNewOpAndCopyName<ICmpOp>(rewriter, op, predicate, lhs, rhs,
                                            op.getTwoState());
      return success();
    };

    auto replaceWithConstantI1 = [&](bool constant) -> LogicalResult {
      replaceOpWithNewOpAndCopyName<hw::ConstantOp>(rewriter, op,
                                                    APInt(1, constant));
      return success();
    };

    switch (op.getPredicate()) {
    case ICmpPredicate::slt:
      // x < max -> x != max
      if (rhs.isMaxSignedValue())
        return replaceWith(ICmpPredicate::ne, op.getLhs(), op.getRhs());
      // x < min -> false
      if (rhs.isMinSignedValue())
        return replaceWithConstantI1(0);
      // x < min+1 -> x == min
      if ((rhs - 1).isMinSignedValue())
        return replaceWith(ICmpPredicate::eq, op.getLhs(),
                           getConstant(rhs - 1));
      break;
    case ICmpPredicate::sgt:
      // x > min -> x != min
      if (rhs.isMinSignedValue())
        return replaceWith(ICmpPredicate::ne, op.getLhs(), op.getRhs());
      // x > max -> false
      if (rhs.isMaxSignedValue())
        return replaceWithConstantI1(0);
      // x > max-1 -> x == max
      if ((rhs + 1).isMaxSignedValue())
        return replaceWith(ICmpPredicate::eq, op.getLhs(),
                           getConstant(rhs + 1));
      break;
    case ICmpPredicate::ult:
      // x < max -> x != max
      if (rhs.isAllOnes())
        return replaceWith(ICmpPredicate::ne, op.getLhs(), op.getRhs());
      // x < min -> false
      if (rhs.isZero())
        return replaceWithConstantI1(0);
      // x < min+1 -> x == min
      if ((rhs - 1).isZero())
        return replaceWith(ICmpPredicate::eq, op.getLhs(),
                           getConstant(rhs - 1));

      // x < 0xE0 -> extract(x, 5..7) != 0b111
      if (rhs.countLeadingOnes() + rhs.countTrailingZeros() ==
          rhs.getBitWidth()) {
        auto numOnes = rhs.countLeadingOnes();
        auto smaller = rewriter.create<ExtractOp>(
            op.getLoc(), op.getLhs(), rhs.getBitWidth() - numOnes, numOnes);
        return replaceWith(ICmpPredicate::ne, smaller,
                           getConstant(APInt::getAllOnes(numOnes)));
      }

      break;
    case ICmpPredicate::ugt:
      // x > min -> x != min
      if (rhs.isZero())
        return replaceWith(ICmpPredicate::ne, op.getLhs(), op.getRhs());
      // x > max -> false
      if (rhs.isAllOnes())
        return replaceWithConstantI1(0);
      // x > max-1 -> x == max
      if ((rhs + 1).isAllOnes())
        return replaceWith(ICmpPredicate::eq, op.getLhs(),
                           getConstant(rhs + 1));

      // x > 0x07 -> extract(x, 3..7) != 0b00000
      if ((rhs + 1).isPowerOf2()) {
        auto numOnes = rhs.countTrailingOnes();
        auto newWidth = rhs.getBitWidth() - numOnes;
        auto smaller = rewriter.create<ExtractOp>(op.getLoc(), op.getLhs(),
                                                  numOnes, newWidth);
        return replaceWith(ICmpPredicate::ne, smaller,
                           getConstant(APInt::getZero(newWidth)));
      }

      break;
    case ICmpPredicate::sle:
      // x <= max -> true
      if (rhs.isMaxSignedValue())
        return replaceWithConstantI1(1);
      // x <= c -> x < (c+1)
      return replaceWith(ICmpPredicate::slt, op.getLhs(), getConstant(rhs + 1));
    case ICmpPredicate::sge:
      // x >= min -> true
      if (rhs.isMinSignedValue())
        return replaceWithConstantI1(1);
      // x >= c -> x > (c-1)
      return replaceWith(ICmpPredicate::sgt, op.getLhs(), getConstant(rhs - 1));
    case ICmpPredicate::ule:
      // x <= max -> true
      if (rhs.isAllOnes())
        return replaceWithConstantI1(1);
      // x <= c -> x < (c+1)
      return replaceWith(ICmpPredicate::ult, op.getLhs(), getConstant(rhs + 1));
    case ICmpPredicate::uge:
      // x >= min -> true
      if (rhs.isZero())
        return replaceWithConstantI1(1);
      // x >= c -> x > (c-1)
      return replaceWith(ICmpPredicate::ugt, op.getLhs(), getConstant(rhs - 1));
    case ICmpPredicate::eq:
      if (rhs.getBitWidth() == 1) {
        if (rhs.isZero()) {
          // x == 0 -> x ^ 1
          replaceOpWithNewOpAndCopyName<XorOp>(rewriter, op, op.getLhs(),
                                               getConstant(APInt(1, 1)),
                                               op.getTwoState());
          return success();
        }
        if (rhs.isAllOnes()) {
          // x == 1 -> x
          replaceOpAndCopyName(rewriter, op, op.getLhs());
          return success();
        }
      }
      break;
    case ICmpPredicate::ne:
      if (rhs.getBitWidth() == 1) {
        if (rhs.isZero()) {
          // x != 0 -> x
          replaceOpAndCopyName(rewriter, op, op.getLhs());
          return success();
        }
        if (rhs.isAllOnes()) {
          // x != 1 -> x ^ 1
          replaceOpWithNewOpAndCopyName<XorOp>(rewriter, op, op.getLhs(),
                                               getConstant(APInt(1, 1)),
                                               op.getTwoState());
          return success();
        }
      }
      break;
    case ICmpPredicate::ceq:
    case ICmpPredicate::cne:
    case ICmpPredicate::weq:
    case ICmpPredicate::wne:
      break;
    }

    // We have some specific optimizations for comparison with a constant that
    // are only supported for equality comparisons.
    if (op.getPredicate() == ICmpPredicate::eq ||
        op.getPredicate() == ICmpPredicate::ne) {
      // Simplify `icmp(value_with_known_bits, rhscst)` into some extracts
      // with a smaller constant.  We only support equality comparisons for
      // this.
      auto knownBits = computeKnownBits(op.getLhs());
      if (!knownBits.isUnknown())
        return combineEqualityICmpWithKnownBitsAndConstant(op, knownBits, rhs,
                                                           rewriter),
               success();

      // Simplify icmp eq(xor(a,b,cst1), cst2) -> icmp eq(xor(a,b),
      // cst1^cst2).
      if (auto xorOp = op.getLhs().getDefiningOp<XorOp>())
        if (xorOp.getOperands().back().getDefiningOp<hw::ConstantOp>())
          return combineEqualityICmpWithXorOfConstant(op, xorOp, rhs, rewriter),
                 success();

      // Simplify icmp eq(replicate(v, n), c) -> icmp eq(v, c) if c is zero or
      // all one.
      if (auto replicateOp = op.getLhs().getDefiningOp<ReplicateOp>())
        if (rhs.isAllOnes() || rhs.isZero()) {
          auto width = replicateOp.getInput().getType().getIntOrFloatBitWidth();
          auto cst = rewriter.create<hw::ConstantOp>(
              op.getLoc(), rhs.isAllOnes() ? APInt::getAllOnes(width)
                                           : APInt::getZero(width));
          replaceOpWithNewOpAndCopyName<ICmpOp>(rewriter, op, op.getPredicate(),
                                                replicateOp.getInput(), cst,
                                                op.getTwoState());
          return success();
        }
    }
  }

  // icmp(cat(prefix, a, b, suffix), cat(prefix, c, d, suffix)) => icmp(cat(a,
  // b), cat(c, d)). contains special handling for sign bit in signed
  // compressions.
  if (Operation *opLHS = op.getLhs().getDefiningOp())
    if (Operation *opRHS = op.getRhs().getDefiningOp())
      if (isa<ConcatOp, ReplicateOp>(opLHS) &&
          isa<ConcatOp, ReplicateOp>(opRHS)) {
        if (succeeded(matchAndRewriteCompareConcat(op, opLHS, opRHS, rewriter)))
          return success();
      }

  return failure();
}
