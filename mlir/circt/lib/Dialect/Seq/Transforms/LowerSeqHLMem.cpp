//===- LowerSeqHLMem.cpp - seq.hlmem lowering -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass pattern matches lowering patterns on seq.hlmem ops and referencing
// ports.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace seq;

namespace {

struct SimpleBehavioralMemoryLowering
    : public OpConversionPattern<seq::HLMemOp> {
  // A simple behavioral SV implementation of a HLMemOp. This is intended as a
  // fall-back pattern if any other higher benefit/target-specific patterns
  // failed to match.
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::HLMemOp mem, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Only support unidimensional memories.
    auto memType = mem.getMemType();
    if (memType.getShape().size() != 1)
      return rewriter.notifyMatchFailure(
          mem, "only unidimensional memories are supported");
    auto size = memType.getShape()[0];

    // Gather up the referencing ops.
    llvm::SmallVector<seq::ReadPortOp> readOps;
    llvm::SmallVector<seq::WritePortOp> writeOps;
    for (auto *user : mem.getHandle().getUsers()) {
      auto res = llvm::TypeSwitch<Operation *, LogicalResult>(user)
                     .Case([&](seq::ReadPortOp op) {
                       readOps.push_back(op);
                       return success();
                     })
                     .Case([&](seq::WritePortOp op) {
                       writeOps.push_back(op);
                       return success();
                     })
                     .Default([&](Operation *op) { return failure(); });
      if (failed(res))
        return rewriter.notifyMatchFailure(user, "unsupported port type");
    }

    auto clk = mem.getClk();
    auto rst = mem.getRst();
    auto memName = mem.getName();

    // Create the SV memory.
    hw::UnpackedArrayType memArrType =
        hw::UnpackedArrayType::get(memType.getElementType(), size);
    auto svMem =
        rewriter.create<sv::RegOp>(mem.getLoc(), memArrType, mem.getNameAttr())
            .getResult();

    // Create write ports by gathering up the write port inputs and
    // materializing the writes inside a single always ff block.
    struct WriteTuple {
      Location loc;
      Value addr;
      Value data;
      Value en;
    };
    llvm::SmallVector<WriteTuple> writeTuples;
    for (auto writeOp : writeOps) {
      if (writeOp.getLatency() != 1)
        return rewriter.notifyMatchFailure(
            writeOp, "only supports write ports with latency == 1");
      auto addr = writeOp.getAddresses()[0];
      auto data = writeOp.getInData();
      auto en = writeOp.getWrEn();
      writeTuples.push_back({writeOp.getLoc(), addr, data, en});
      rewriter.eraseOp(writeOp);
    }

    rewriter.create<sv::AlwaysFFOp>(
        mem.getLoc(), sv::EventControl::AtPosEdge, clk, ResetType::SyncReset,
        sv::EventControl::AtPosEdge, rst, [&] {
          for (auto [loc, address, data, en] : writeTuples) {
            Value a = address, d = data; // So the lambda can capture.
            Location l = loc;
            // Perform write upon write enable being high.
            rewriter.create<sv::IfOp>(loc, en, [&] {
              Value memLoc =
                  rewriter.create<sv::ArrayIndexInOutOp>(l, svMem, a);
              rewriter.create<sv::PAssignOp>(l, memLoc, d);
            });
          }
        });

    // Create read ports. When latency > 1, we perform the read 1 cycle before
    // the latency deadline, and register the read output. By doing so, we avoid
    // start a critical path right after the combinational read.
    // TODO: When latency > 2, and assuming that the exact read time is flexible
    // within the latency window, we could decide on whether to buffer the read
    // data or the read address more, based on which is narrower (saving area).
    for (auto [ri, readOp] : llvm::enumerate(readOps)) {
      rewriter.setInsertionPointAfter(readOp);
      auto loc = readOp.getLoc();

      auto readAddress = readOp.getAddresses()[0];
      unsigned latency = readOp.getLatency();
      unsigned addressDelayCycles = latency - 1;
      if (latency > 0) {
        // Materialize any delays on the read address.
        for (unsigned i = 0; i < addressDelayCycles; ++i) {
          readAddress = rewriter.create<seq::CompRegOp>(
              loc, readAddress, clk,
              rewriter.getStringAttr(memName + "_rdaddr" + std::to_string(ri) +
                                     "_dly" + std::to_string(i)));
        }
      }

      // Create a combinational read.
      Value memLoc =
          rewriter.create<sv::ArrayIndexInOutOp>(loc, svMem, readAddress);
      Value readData = rewriter.create<sv::ReadInOutOp>(loc, memLoc);
      if (latency > 0) {
        // Register the read data.
        readData = rewriter.create<seq::CompRegOp>(
            loc, readData, clk,
            rewriter.getStringAttr(memName + "_rd" + std::to_string(ri) +
                                   "_reg"));
      }
      rewriter.replaceOp(readOp, {readData});
    }

    rewriter.eraseOp(mem);
    return success();
  }
};

#define GEN_PASS_DEF_LOWERSEQHLMEM
#include "circt/Dialect/Seq/SeqPasses.h.inc"

struct LowerSeqHLMemPass : public impl::LowerSeqHLMemBase<LowerSeqHLMemPass> {
  void runOnOperation() override;
};

} // namespace

void LowerSeqHLMemPass::runOnOperation() {
  hw::HWModuleOp top = getOperation();

  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);

  // Lowering patterns must lower away all HLMem-related operations.
  target.addIllegalOp<seq::HLMemOp, seq::ReadPortOp, seq::WritePortOp>();
  target.addLegalDialect<sv::SVDialect, seq::SeqDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<SimpleBehavioralMemoryLowering>(&ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::seq::createLowerSeqHLMemPass() {
  return std::make_unique<LowerSeqHLMemPass>();
}
