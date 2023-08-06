//===- HWToSV.cpp - HW To SV Conversion Pass ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to SV Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToSV.h"
#include "../PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace sv;

static sv::EventControl hwToSvEventControl(hw::EventControl ec) {
  switch (ec) {
  case hw::EventControl::AtPosEdge:
    return sv::EventControl::AtPosEdge;
  case hw::EventControl::AtNegEdge:
    return sv::EventControl::AtNegEdge;
  case hw::EventControl::AtEdge:
    return sv::EventControl::AtEdge;
  }
  llvm_unreachable("Unknown event control kind");
}

namespace {
struct HWToSVPass : public LowerHWToSVBase<HWToSVPass> {
  void runOnOperation() override;
};

struct TriggeredOpConversionPattern : public OpConversionPattern<TriggeredOp> {
  using OpConversionPattern<TriggeredOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TriggeredOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto alwaysOp = rewriter.create<AlwaysOp>(
        op.getLoc(),
        llvm::SmallVector<sv::EventControl>{hwToSvEventControl(op.getEvent())},
        llvm::SmallVector<Value>{op.getTrigger()});
    rewriter.mergeBlocks(op.getBodyBlock(), alwaysOp.getBodyBlock(),
                         operands.getInputs());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void HWToSVPass::runOnOperation() {
  MLIRContext &context = getContext();
  hw::HWModuleOp module = getOperation();

  ConversionTarget target(context);
  RewritePatternSet patterns(&context);

  target.addIllegalOp<TriggeredOp>();
  target.addLegalDialect<sv::SVDialect>();

  patterns.add<TriggeredOpConversionPattern>(&context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// HW to SV Conversion Pass
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<hw::HWModuleOp>> circt::createLowerHWToSVPass() {
  return std::make_unique<HWToSVPass>();
}
