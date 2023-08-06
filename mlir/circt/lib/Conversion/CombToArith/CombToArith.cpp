//===- CombToArith.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToArith.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace hw;
using namespace comb;
using namespace mlir;
using namespace arith;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a comb::ReplicateOp operation to a comb::ConcatOp
struct CombReplicateOpConversion : OpConversionPattern<ReplicateOp> {
  using OpConversionPattern<ReplicateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type inputType = op.getInput().getType();
    if (inputType.isa<IntegerType>() &&
        inputType.getIntOrFloatBitWidth() == 1) {
      Type outType = rewriter.getIntegerType(op.getMultiple());
      rewriter.replaceOpWithNewOp<ExtSIOp>(op, outType, adaptor.getInput());
      return success();
    }

    SmallVector<Value> inputs(op.getMultiple(), adaptor.getInput());
    rewriter.replaceOpWithNewOp<ConcatOp>(op, inputs);
    return success();
  }
};

/// Lower a hw::ConstantOp operation to a arith::ConstantOp
struct HWConstantOpConversion : OpConversionPattern<hw::ConstantOp> {
  using OpConversionPattern<hw::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, adaptor.getValueAttr());
    return success();
  }
};

/// Lower a comb::ICmpOp operation to a arith::CmpIOp
struct IcmpOpConversion : OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    CmpIPredicate pred;
    switch (adaptor.getPredicate()) {
    case ICmpPredicate::cne:
    case ICmpPredicate::wne:
    case ICmpPredicate::ne:
      pred = CmpIPredicate::ne;
      break;
    case ICmpPredicate::ceq:
    case ICmpPredicate::weq:
    case ICmpPredicate::eq:
      pred = CmpIPredicate::eq;
      break;
    case ICmpPredicate::sge:
      pred = CmpIPredicate::sge;
      break;
    case ICmpPredicate::sgt:
      pred = CmpIPredicate::sgt;
      break;
    case ICmpPredicate::sle:
      pred = CmpIPredicate::sle;
      break;
    case ICmpPredicate::slt:
      pred = CmpIPredicate::slt;
      break;
    case ICmpPredicate::uge:
      pred = CmpIPredicate::uge;
      break;
    case ICmpPredicate::ugt:
      pred = CmpIPredicate::ugt;
      break;
    case ICmpPredicate::ule:
      pred = CmpIPredicate::ule;
      break;
    case ICmpPredicate::ult:
      pred = CmpIPredicate::ult;
      break;
    }

    rewriter.replaceOpWithNewOp<CmpIOp>(op, pred, adaptor.getLhs(),
                                        adaptor.getRhs());
    return success();
  }
};

/// Lower a comb::ExtractOp operation to the arith dialect
struct ExtractOpConversion : OpConversionPattern<ExtractOp> {
  using OpConversionPattern<ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value lowBit = rewriter.create<arith::ConstantOp>(
        op.getLoc(),
        IntegerAttr::get(adaptor.getInput().getType(), adaptor.getLowBit()));
    Value shifted =
        rewriter.create<ShRUIOp>(op.getLoc(), adaptor.getInput(), lowBit);
    rewriter.replaceOpWithNewOp<TruncIOp>(op, op.getResult().getType(),
                                          shifted);
    return success();
  }
};

/// Lower a comb::ConcatOp operation to the arith dialect
struct ConcatOpConversion : OpConversionPattern<ConcatOp> {
  using OpConversionPattern<ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = op.getResult().getType();
    Location loc = op.getLoc();
    unsigned nextInsertion = type.getIntOrFloatBitWidth();

    Value aggregate =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(type, 0));

    for (unsigned i = 0, e = op.getNumOperands(); i < e; i++) {
      nextInsertion -=
          adaptor.getOperands()[i].getType().getIntOrFloatBitWidth();

      Value nextInsValue = rewriter.create<arith::ConstantOp>(
          loc, IntegerAttr::get(type, nextInsertion));
      Value extended =
          rewriter.create<ExtUIOp>(loc, type, adaptor.getOperands()[i]);
      Value shifted = rewriter.create<ShLIOp>(loc, extended, nextInsValue);
      aggregate = rewriter.create<OrIOp>(loc, aggregate, shifted);
    }

    rewriter.replaceOp(op, aggregate);
    return success();
  }
};

/// Lower the two-operand SourceOp to the two-operand TargetOp
template <typename SourceOp, typename TargetOp>
struct BinaryOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<TargetOp>(op, op.getResult().getType(),
                                          adaptor.getOperands());
    return success();
  }
};

/// Lower a comb::ReplicateOp operation to the LLVM dialect.
template <typename SourceOp, typename TargetOp>
struct VariadicOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO: building a tree would be better here
    ValueRange operands = adaptor.getOperands();
    Value runner = operands[0];
    for (Value operand :
         llvm::make_range(operands.begin() + 1, operands.end())) {
      runner = rewriter.create<TargetOp>(op.getLoc(), runner, operand);
    }
    rewriter.replaceOp(op, runner);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to Arith pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToArithPass
    : public ConvertCombToArithBase<ConvertCombToArithPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateCombToArithConversionPatterns(
    TypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<
      CombReplicateOpConversion, HWConstantOpConversion, IcmpOpConversion,
      ExtractOpConversion, ConcatOpConversion,
      BinaryOpConversion<ShlOp, ShLIOp>, BinaryOpConversion<ShrSOp, ShRSIOp>,
      BinaryOpConversion<ShrUOp, ShRUIOp>, BinaryOpConversion<SubOp, SubIOp>,
      BinaryOpConversion<DivSOp, DivSIOp>, BinaryOpConversion<DivUOp, DivUIOp>,
      BinaryOpConversion<ModSOp, RemSIOp>, BinaryOpConversion<ModUOp, RemUIOp>,
      BinaryOpConversion<MuxOp, SelectOp>, VariadicOpConversion<AddOp, AddIOp>,
      VariadicOpConversion<MulOp, MulIOp>, VariadicOpConversion<AndOp, AndIOp>,
      VariadicOpConversion<OrOp, OrIOp>, VariadicOpConversion<XorOp, XOrIOp>>(
      converter, patterns.getContext());
}

void ConvertCombToArithPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<comb::CombDialect>();
  target.addIllegalOp<hw::ConstantOp>();
  target.addLegalDialect<ArithDialect>();
  // Arith does not have an operation equivalent to comb.parity. A lowering
  // would result in undesirably complex logic, therefore, we mark it legal
  // here.
  target.addLegalOp<comb::ParityOp>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });
  // TODO: a pattern for comb.parity
  populateCombToArithConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertCombToArithPass() {
  return std::make_unique<ConvertCombToArithPass>();
}
