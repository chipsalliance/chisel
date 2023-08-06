//===- DCOps.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/DC/DCOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace circt;
using namespace dc;
using namespace mlir;

bool circt::dc::isI1ValueType(Type t) {
  auto vt = t.dyn_cast<ValueType>();
  if (!vt)
    return false;
  auto innerWidth = vt.getInnerType().getIntOrFloatBitWidth();
  return innerWidth == 1;
}

namespace circt {
namespace dc {

// =============================================================================
// JoinOp
// =============================================================================

OpFoldResult JoinOp::fold(FoldAdaptor adaptor) {
  // Fold simple joins (joins with 1 input).
  if (auto tokens = getTokens(); tokens.size() == 1)
    return tokens.front();

  // These folders are disabled to work around MLIR bugs when changing
  // the number of operands.  https://github.com/llvm/llvm-project/issues/64280
  return {};

  // Remove operands which originate from a dc.source op (redundant).
  auto *op = getOperation();
  for (OpOperand &operand : llvm::make_early_inc_range(op->getOpOperands())) {
    if (auto source = operand.get().getDefiningOp<dc::SourceOp>()) {
      op->eraseOperand(operand.getOperandNumber());
      return getOutput();
    }
  }

  // Remove duplicate operands.
  llvm::DenseSet<Value> uniqueOperands;
  for (OpOperand &operand : llvm::make_early_inc_range(op->getOpOperands())) {
    if (!uniqueOperands.insert(operand.get()).second) {
      op->eraseOperand(operand.getOperandNumber());
      return getOutput();
    }
  }

  // Canonicalization staggered joins where the sink join contains inputs also
  // found in the source join.
  for (OpOperand &operand : llvm::make_early_inc_range(op->getOpOperands())) {
    auto otherJoin = operand.get().getDefiningOp<dc::JoinOp>();
    if (!otherJoin) {
      // Operand does not originate from a join so it's a valid join input.
      continue;
    }

    // Operand originates from a join. Erase the current join operand and add
    // all of the otherJoin op's inputs to this join.
    // DCE will take care of otherJoin in case it's no longer used.
    op->eraseOperand(operand.getOperandNumber());
    op->insertOperands(getNumOperands(), otherJoin.getTokens());
    return getOutput();
  }

  return {};
}

// =============================================================================
// ForkOp
// =============================================================================

template <typename TInt>
static ParseResult parseIntInSquareBrackets(OpAsmParser &parser, TInt &v) {
  if (parser.parseLSquare() || parser.parseInteger(v) || parser.parseRSquare())
    return failure();
  return success();
}

ParseResult ForkOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  size_t size = 0;
  if (parseIntInSquareBrackets(parser, size))
    return failure();

  if (size == 0)
    return parser.emitError(parser.getNameLoc(),
                            "fork size must be greater than 0");

  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  auto tt = dc::TokenType::get(parser.getContext());
  llvm::SmallVector<Type> operandTypes{tt};
  SmallVector<Type> resultTypes{size, tt};
  result.addTypes(resultTypes);
  if (parser.resolveOperand(operand, tt, result.operands))
    return failure();
  return success();
}

void ForkOp::print(OpAsmPrinter &p) {
  p << " [" << getNumResults() << "] ";
  p << getOperand() << " ";
  auto attrs = (*this)->getAttrs();
  if (!attrs.empty()) {
    p << " ";
    p.printOptionalAttrDict(attrs);
  }
}

class EliminateForkToForkPattern : public OpRewritePattern<ForkOp> {
  // Canonicalization of forks where the output is fed into another fork.
public:
  using OpRewritePattern<ForkOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ForkOp fork,
                                PatternRewriter &rewriter) const override {
    for (auto output : fork.getOutputs()) {
      for (auto *user : output.getUsers()) {
        auto userFork = dyn_cast<ForkOp>(user);
        if (!userFork)
          continue;

        // We have a fork feeding into another fork. Replace the output fork by
        // adding more outputs to the current fork.
        size_t totalForks = fork.getNumResults() + userFork.getNumResults() - 1;

        auto newFork = rewriter.create<dc::ForkOp>(fork.getLoc(),
                                                   fork.getToken(), totalForks);
        rewriter.replaceOp(
            fork, newFork.getResults().take_front(fork.getNumResults()));
        rewriter.replaceOp(
            userFork, newFork.getResults().take_back(userFork.getNumResults()));

        // Just stop the pattern here instead of trying to do more - let the
        // canonicalizer recurse if another run of the canonicalization applies.
        return success();
      }
    }
    return failure();
  }
};

class EliminateForkOfSourcePattern : public OpRewritePattern<ForkOp> {
  // Canonicalizes away forks on source ops, in favor of individual source
  // operations. Having standalone sources are a better alternative, since other
  // operations can canonicalize on it (e.g. joins) as well as being very cheap
  // to implement in hardware, if they do remain.
public:
  using OpRewritePattern<ForkOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ForkOp fork,
                                PatternRewriter &rewriter) const override {
    auto source = fork.getToken().getDefiningOp<SourceOp>();
    if (!source)
      return failure();

    // We have a source feeding into a fork. Replace the fork by a source for
    // each output.
    llvm::SmallVector<Value> sources;
    for (size_t i = 0; i < fork.getNumResults(); ++i)
      sources.push_back(rewriter.create<dc::SourceOp>(fork.getLoc()));

    rewriter.replaceOp(fork, sources);
    return success();
  }
};

void ForkOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<EliminateForkToForkPattern, EliminateForkOfSourcePattern>(
      context);
}

LogicalResult ForkOp::fold(FoldAdaptor adaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  // Fold simple forks (forks with 1 output).
  if (getOutputs().size() == 1) {
    results.push_back(getToken());
    return success();
  }

  return failure();
}

// =============================================================================
// UnpackOp
// =============================================================================

struct EliminateRedundantUnpackPattern : public OpRewritePattern<UnpackOp> {
  // Eliminates unpacks where only the token is used.
  using OpRewritePattern<UnpackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnpackOp unpack,
                                PatternRewriter &rewriter) const override {
    // Is the value-side of the unpack used?
    if (!unpack.getOutput().use_empty())
      return failure();

    auto pack = unpack.getInput().getDefiningOp<PackOp>();
    if (!pack)
      return failure();

    // Replace all uses of the unpack token with the packed token.
    rewriter.replaceAllUsesWith(unpack.getToken(), pack.getToken());
    rewriter.eraseOp(unpack);
    return success();
  }
};

void UnpackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<EliminateRedundantUnpackPattern>(context);
}

LogicalResult UnpackOp::fold(FoldAdaptor adaptor,
                             SmallVectorImpl<OpFoldResult> &results) {
  // Unpack of a pack is a no-op.
  if (auto pack = getInput().getDefiningOp<PackOp>()) {
    results.push_back(pack.getToken());
    results.push_back(pack.getInput());
    return success();
  }

  return failure();
}

LogicalResult UnpackOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  auto inputType = operands.front().getType().cast<ValueType>();
  results.push_back(TokenType::get(context));
  results.push_back(inputType.getInnerType());
  return success();
}

// =============================================================================
// PackOp
// =============================================================================

OpFoldResult PackOp::fold(FoldAdaptor adaptor) {
  auto token = getToken();

  // Pack of an unpack is a no-op.
  if (auto unpack = token.getDefiningOp<UnpackOp>()) {
    if (unpack.getOutput() == getInput())
      return unpack.getInput();
  }
  return {};
}

LogicalResult PackOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  llvm::SmallVector<Type> inputTypes;
  Type inputType = operands.back().getType();
  auto valueType = dc::ValueType::get(context, inputType);
  results.push_back(valueType);
  return success();
}

// =============================================================================
// SelectOp
// =============================================================================

class EliminateBranchToSelectPattern : public OpRewritePattern<SelectOp> {
  // Canonicalize away a select that is fed only by a single branch
  // example:
  //   %true, %false = dc.branch %sel1 %token
  //   %0 = dc.select %sel2, %true, %false
  // ->
  //   %0 = dc.join %sel1, %sel2, %token

public:
  using OpRewritePattern<SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SelectOp select,
                                PatternRewriter &rewriter) const override {
    // Do all the inputs come from a branch?
    BranchOp branchInput;
    for (auto operand : {select.getTrueToken(), select.getFalseToken()}) {
      auto br = operand.getDefiningOp<BranchOp>();
      if (!br)
        return failure();

      if (!branchInput)
        branchInput = br;
      else if (branchInput != br)
        return failure();
    }

    // Replace the select with a join (unpack the select conditions).
    rewriter.replaceOpWithNewOp<JoinOp>(
        select,
        llvm::SmallVector<Value>{
            rewriter.create<UnpackOp>(select.getLoc(), select.getCondition())
                .getToken(),
            rewriter
                .create<UnpackOp>(branchInput.getLoc(),
                                  branchInput.getCondition())
                .getToken()});

    return success();
  }
};

void SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<EliminateBranchToSelectPattern>(context);
}

// =============================================================================
// BufferOp
// =============================================================================

FailureOr<SmallVector<int64_t>> BufferOp::getInitValueArray() {
  assert(getInitValues() && "initValues attribute not set");
  SmallVector<int64_t> values;
  for (auto value : getInitValuesAttr()) {
    if (auto iValue = value.dyn_cast<IntegerAttr>()) {
      values.push_back(iValue.getValue().getSExtValue());
    } else {
      return emitError() << "initValues attribute must be an array of integers";
    }
  }
  return values;
}

LogicalResult BufferOp::verify() {
  // Verify that exactly 'size' number of initial values have been provided, if
  // an initializer list have been provided.
  if (auto initVals = getInitValuesAttr()) {
    auto nInits = initVals.size();
    if (nInits != getSize())
      return emitOpError() << "expected " << getSize()
                           << " init values but got " << nInits << ".";
  }

  return success();
}

// =============================================================================
// ToESIOp
// =============================================================================

LogicalResult ToESIOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Type channelEltType;
  if (auto valueType = operands.front().getType().dyn_cast<ValueType>())
    channelEltType = valueType.getInnerType();
  else {
    // dc.token => esi.channel<i0>
    channelEltType = IntegerType::get(context, 0);
  }

  results.push_back(esi::ChannelType::get(context, channelEltType));
  return success();
}

// =============================================================================
// FromESIOp
// =============================================================================

LogicalResult FromESIOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  auto innerType =
      operands.front().getType().cast<esi::ChannelType>().getInner();
  if (auto intType = innerType.dyn_cast<IntegerType>(); intType.getWidth() == 0)
    results.push_back(dc::TokenType::get(context));
  else
    results.push_back(dc::ValueType::get(context, innerType));

  return success();
}

} // namespace dc
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/DC/DC.cpp.inc"
