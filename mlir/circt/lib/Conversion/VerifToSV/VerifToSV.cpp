//===- VerifToSV.cpp - HW To SV Conversion Pass ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Verif to SV Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSV.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace sv;
using namespace verif;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

struct PrintOpConversionPattern : public OpConversionPattern<PrintOp> {
  using OpConversionPattern<PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // Printf's will be emitted to stdout (32'h8000_0001 in IEEE Std 1800-2012).
    Value fdStdout = rewriter.create<hw::ConstantOp>(
        op.getLoc(), APInt(32, 0x80000001, false));

    auto fstrOp =
        dyn_cast_or_null<FormatVerilogStringOp>(op.getString().getDefiningOp());
    if (!fstrOp)
      return op->emitOpError() << "expected FormatVerilogStringOp as the "
                                  "source of the formatted string";

    rewriter.replaceOpWithNewOp<sv::FWriteOp>(
        op, fdStdout, fstrOp.getFormatString(), fstrOp.getSubstitutions());
    return success();
  }
};

struct HasBeenResetConversion : public OpConversionPattern<HasBeenResetOp> {
  using OpConversionPattern<HasBeenResetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HasBeenResetOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1 = rewriter.getI1Type();
    auto constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 1);
    auto constZero = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 0);
    auto constX = rewriter.create<sv::ConstantXOp>(op.getLoc(), i1);

    // Declare the register that will track the reset state.
    auto reg = rewriter.create<sv::RegOp>(
        op.getLoc(), i1, rewriter.getStringAttr("hasBeenResetReg"));

    auto clock = operands.getClock();
    auto reset = operands.getReset();

    // Explicitly initialize the register in an `initial` block. In general, the
    // register will come up as X, but this may be overridden by simulator
    // configuration options.
    //
    // In case the reset is async, check if the reset is already active during
    // the `initial` block and immediately set the register to 1. Otherwise
    // initialize to X.
    rewriter.create<sv::InitialOp>(op.getLoc(), [&] {
      auto assignOne = [&] {
        rewriter.create<sv::BPAssignOp>(op.getLoc(), reg, constOne);
      };
      auto assignX = [&] {
        rewriter.create<sv::BPAssignOp>(op.getLoc(), reg, constX);
      };
      if (op.getAsync())
        rewriter.create<sv::IfOp>(op.getLoc(), reset, assignOne, assignX);
      else
        assignX();
    });

    // Create the `always` block that sets the register to 1 as soon as the
    // reset is initiated. For async resets this happens at the reset's posedge;
    // for sync resets this happens on the clock's posedge if the reset is set.
    Value triggerOn = op.getAsync() ? reset : clock;
    rewriter.create<sv::AlwaysOp>(
        op.getLoc(), sv::EventControl::AtPosEdge, triggerOn, [&] {
          auto assignOne = [&] {
            rewriter.create<sv::PAssignOp>(op.getLoc(), reg, constOne);
          };
          if (op.getAsync())
            assignOne();
          else
            rewriter.create<sv::IfOp>(op.getLoc(), reset, assignOne);
        });

    // Derive the actual result value:
    //   hasBeenReset = (hasBeenResetReg === 1) && (reset === 0);
    auto regRead = rewriter.create<sv::ReadInOutOp>(op.getLoc(), reg);
    auto regIsOne = rewriter.createOrFold<comb::ICmpOp>(
        op.getLoc(), comb::ICmpPredicate::ceq, regRead, constOne);
    auto resetIsZero = rewriter.createOrFold<comb::ICmpOp>(
        op.getLoc(), comb::ICmpPredicate::ceq, reset, constZero);
    auto resetStartedAndEnded = rewriter.createOrFold<comb::AndOp>(
        op.getLoc(), regIsOne, resetIsZero, true);
    rewriter.replaceOpWithNewOp<hw::WireOp>(
        op, resetStartedAndEnded, rewriter.getStringAttr("hasBeenReset"));

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct VerifToSVPass : public LowerVerifToSVBase<VerifToSVPass> {
  void runOnOperation() override;
};
} // namespace

void VerifToSVPass::runOnOperation() {
  MLIRContext &context = getContext();
  hw::HWModuleOp module = getOperation();

  ConversionTarget target(context);
  RewritePatternSet patterns(&context);

  target.addIllegalOp<PrintOp, HasBeenResetOp>();
  target.addLegalDialect<sv::SVDialect, hw::HWDialect, comb::CombDialect>();
  patterns.add<PrintOpConversionPattern, HasBeenResetConversion>(&context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<hw::HWModuleOp>>
circt::createLowerVerifToSVPass() {
  return std::make_unique<VerifToSVPass>();
}
