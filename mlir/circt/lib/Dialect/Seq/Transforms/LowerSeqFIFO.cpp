//===- LowerSeqFIFO.cpp - seq.fifo lowering -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace seq;

namespace {

struct FIFOLowering : public OpConversionPattern<seq::FIFOOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::FIFOOp mem, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type eltType = adaptor.getInput().getType();
    Value clk = adaptor.getClk();
    Value rst = adaptor.getRst();
    Location loc = mem.getLoc();
    BackedgeBuilder bb(rewriter, loc);
    size_t depth = mem.getDepth();
    Type countType = rewriter.getIntegerType(llvm::Log2_64_Ceil(depth + 1));
    Type ptrType = rewriter.getIntegerType(llvm::Log2_64_Ceil(depth));
    Backedge rdAddrNext = bb.get(ptrType);
    Backedge wrAddrNext = bb.get(ptrType);
    Backedge nextCount = bb.get(countType);

    // ====== Some constants ======
    Value countTcFull =
        rewriter.create<hw::ConstantOp>(loc, countType, depth - 1);
    Value countTc1 = rewriter.create<hw::ConstantOp>(loc, countType, 1);
    Value countTc0 = rewriter.create<hw::ConstantOp>(loc, countType, 0);
    Value ptrTc1 = rewriter.create<hw::ConstantOp>(loc, ptrType, 1);

    // ====== Hardware units ======
    Value count = rewriter.create<seq::CompRegOp>(
        loc, nextCount, clk, rst,
        rewriter.create<hw::ConstantOp>(loc, countType, 0), "fifo_count");
    seq::HLMemOp hlmem = rewriter.create<seq::HLMemOp>(
        loc, clk, rst, "fifo_mem",
        llvm::SmallVector<int64_t>{static_cast<int64_t>(depth)}, eltType);
    Value rdAddr = rewriter.create<seq::CompRegOp>(
        loc, rdAddrNext, clk, rst,
        rewriter.create<hw::ConstantOp>(loc, ptrType, 0), "fifo_rd_addr");
    Value wrAddr = rewriter.create<seq::CompRegOp>(
        loc, wrAddrNext, clk, rst,
        rewriter.create<hw::ConstantOp>(loc, ptrType, 0), "fifo_wr_addr");

    Value readData = rewriter.create<seq::ReadPortOp>(
        loc, hlmem, llvm::SmallVector<Value>{rdAddr}, adaptor.getRdEn(),
        /*latency*/ 0);
    rewriter.create<seq::WritePortOp>(loc, hlmem,
                                      llvm::SmallVector<Value>{wrAddr},
                                      adaptor.getInput(), adaptor.getWrEn(),
                                      /*latency*/ 1);

    // ====== some more constants =====
    comb::ICmpOp fifoFull = rewriter.create<comb::ICmpOp>(
        loc, comb::ICmpPredicate::eq, count, countTcFull);
    fifoFull->setAttr("sv.namehint", rewriter.getStringAttr("fifo_full"));
    comb::ICmpOp fifoEmpty = rewriter.create<comb::ICmpOp>(
        loc, comb::ICmpPredicate::eq, count, countTc0);
    fifoEmpty->setAttr("sv.namehint", rewriter.getStringAttr("fifo_empty"));

    // ====== Next-state count ======
    auto notRdEn = comb::createOrFoldNot(loc, adaptor.getRdEn(), rewriter);
    auto notWrEn = comb::createOrFoldNot(loc, adaptor.getWrEn(), rewriter);
    Value rdEnNandWrEn = rewriter.create<comb::AndOp>(loc, notRdEn, notWrEn);
    Value rdEnAndNotWrEn =
        rewriter.create<comb::AndOp>(loc, adaptor.getRdEn(), notWrEn);
    Value wrEnAndNotRdEn =
        rewriter.create<comb::AndOp>(loc, adaptor.getWrEn(), notRdEn);

    auto countEqTcFull = rewriter.create<comb::ICmpOp>(
        loc, comb::ICmpPredicate::eq, count, countTcFull);
    auto addCountTc1 = rewriter.create<comb::AddOp>(loc, count, countTc1);
    Value wrEnNext = rewriter.create<comb::MuxOp>(loc, countEqTcFull,
                                                  // keep value
                                                  count,
                                                  // increment
                                                  addCountTc1);
    auto countEqTc0 = rewriter.create<comb::ICmpOp>(
        loc, comb::ICmpPredicate::eq, count, countTc0);
    auto subCountTc1 = rewriter.create<comb::SubOp>(loc, count, countTc1);

    Value rdEnNext = rewriter.create<comb::MuxOp>(loc, countEqTc0,
                                                  // keep value
                                                  count,
                                                  // decrement
                                                  subCountTc1);

    auto nextInnerMux =
        rewriter.create<comb::MuxOp>(loc, rdEnAndNotWrEn, rdEnNext, count);
    auto nextMux = rewriter.create<comb::MuxOp>(loc, wrEnAndNotRdEn, wrEnNext,
                                                nextInnerMux);
    nextCount.setValue(rewriter.create<comb::MuxOp>(
        loc, rdEnNandWrEn, /*keep value*/ count, nextMux));
    static_cast<Value>(nextCount).getDefiningOp()->setAttr(
        "sv.namehint", rewriter.getStringAttr("fifo_count_next"));

    // ====== Read/write pointers ======
    Value wrAndNotFull = rewriter.create<comb::AndOp>(
        loc, adaptor.getWrEn(), comb::createOrFoldNot(loc, fifoFull, rewriter));
    auto addWrAddrPtrTc1 = rewriter.create<comb::AddOp>(loc, wrAddr, ptrTc1);
    wrAddrNext.setValue(rewriter.create<comb::MuxOp>(loc, wrAndNotFull,
                                                     addWrAddrPtrTc1, wrAddr));
    static_cast<Value>(wrAddrNext)
        .getDefiningOp()
        ->setAttr("sv.namehint", rewriter.getStringAttr("fifo_wr_addr_next"));

    auto notFifoEmpty = comb::createOrFoldNot(loc, fifoEmpty, rewriter);
    Value rdAndNotEmpty =
        rewriter.create<comb::AndOp>(loc, adaptor.getRdEn(), notFifoEmpty);
    auto addRdAddrPtrTc1 = rewriter.create<comb::AddOp>(loc, rdAddr, ptrTc1);
    rdAddrNext.setValue(rewriter.create<comb::MuxOp>(loc, rdAndNotEmpty,
                                                     addRdAddrPtrTc1, rdAddr));
    static_cast<Value>(rdAddrNext)
        .getDefiningOp()
        ->setAttr("sv.namehint", rewriter.getStringAttr("fifo_rd_addr_next"));

    // ====== Result values ======
    llvm::SmallVector<Value> results;

    // Data
    results.push_back(readData);
    // Full
    results.push_back(fifoFull);
    // Empty
    results.push_back(fifoEmpty);

    if (auto almostFull = mem.getAlmostFullThreshold()) {
      results.push_back(rewriter.create<comb::ICmpOp>(
          loc, comb::ICmpPredicate::uge, count,
          rewriter.create<hw::ConstantOp>(loc, countType, almostFull.value())));
      static_cast<Value>(results.back())
          .getDefiningOp()
          ->setAttr("sv.namehint", rewriter.getStringAttr("fifo_almost_full"));
    }

    if (auto almostEmpty = mem.getAlmostEmptyThreshold()) {
      results.push_back(rewriter.create<comb::ICmpOp>(
          loc, comb::ICmpPredicate::ule, count,
          rewriter.create<hw::ConstantOp>(loc, countType,
                                          almostEmpty.value())));
      static_cast<Value>(results.back())
          .getDefiningOp()
          ->setAttr("sv.namehint", rewriter.getStringAttr("fifo_almost_empty"));
    }

    rewriter.replaceOp(mem, results);
    return success();
  }
};

#define GEN_PASS_DEF_LOWERSEQFIFO
#include "circt/Dialect/Seq/SeqPasses.h.inc"

struct LowerSeqFIFOPass : public impl::LowerSeqFIFOBase<LowerSeqFIFOPass> {
  void runOnOperation() override;
};

} // namespace

void LowerSeqFIFOPass::runOnOperation() {
  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);

  // Lowering patterns must lower away all HLMem-related operations.
  target.addIllegalOp<seq::FIFOOp>();
  target.addLegalDialect<seq::SeqDialect, hw::HWDialect, comb::CombDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<FIFOLowering>(&ctxt);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::seq::createLowerSeqFIFOPass() {
  return std::make_unique<LowerSeqFIFOPass>();
}
