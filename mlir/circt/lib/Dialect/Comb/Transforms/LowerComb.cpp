//===- LowerComb.cpp - Lower some ops in comb -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::comb;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_LOWERCOMB
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

namespace {
/// Lower truth tables to mux trees.
struct TruthTableToMuxTree : public OpConversionPattern<TruthTableOp> {
  using OpConversionPattern::OpConversionPattern;

private:
  /// Get a mux tree for `inputs` corresponding to the given truth table. Do
  /// this recursively by dividing the table in half for each input.
  // NOLINTNEXTLINE(misc-no-recursion)
  Value getMux(Location loc, OpBuilder &b, Value t, Value f,
               ArrayRef<bool> table, Operation::operand_range inputs) const {
    assert(table.size() == (1ull << inputs.size()));
    if (table.size() == 1)
      return table.front() ? t : f;

    size_t half = table.size() / 2;
    Value if1 =
        getMux(loc, b, t, f, table.drop_front(half), inputs.drop_front());
    Value if0 =
        getMux(loc, b, t, f, table.drop_back(half), inputs.drop_front());
    return b.create<MuxOp>(loc, inputs.front(), if1, if0, false);
  }

public:
  LogicalResult matchAndRewrite(TruthTableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    SmallVector<bool> table(
        llvm::map_range(op.getLookupTableAttr().getAsValueRange<IntegerAttr>(),
                        [](const APInt &a) { return !a.isZero(); }));
    Value t = b.create<hw::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), 1));
    Value f = b.create<hw::ConstantOp>(loc, b.getIntegerAttr(b.getI1Type(), 0));

    Value tree = getMux(loc, b, t, f, table, op.getInputs());
    b.updateRootInPlace(tree.getDefiningOp(), [&]() {
      tree.getDefiningOp()->setDialectAttrs(op->getDialectAttrs());
    });
    b.replaceOp(op, tree);
    return success();
  }
};
} // namespace

namespace {
class LowerCombPass : public impl::LowerCombBase<LowerCombPass> {
public:
  using LowerCombBase::LowerCombBase;

  void runOnOperation() override;
};
} // namespace

void LowerCombPass::runOnOperation() {
  ModuleOp module = getOperation();

  ConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addIllegalOp<TruthTableOp>();

  patterns.add<TruthTableToMuxTree>(patterns.getContext());

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return signalPassFailure();
}
