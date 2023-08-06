//===- ESILowerPhysical.cpp - Lower ESI to physical -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower ESI to ESI "physical level" ops conversions and pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;

namespace {
/// Lower `ChannelBufferOp`s, breaking out the various options. For now, just
/// replace with the specified number of pipeline stages (since that's the only
/// option).
struct ChannelBufferLowering : public OpConversionPattern<ChannelBufferOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ChannelBufferOp buffer, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult ChannelBufferLowering::matchAndRewrite(
    ChannelBufferOp buffer, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = buffer.getLoc();

  auto type = buffer.getType();

  // Expand 'abstract' buffer into 'physical' stages.
  auto stages = buffer.getStagesAttr();
  uint64_t numStages = 1;
  if (stages) {
    // Guaranteed positive by the parser.
    numStages = stages.getValue().getLimitedValue();
  }
  Value input = buffer.getInput();
  StringAttr bufferName = buffer.getNameAttr();
  for (uint64_t i = 0; i < numStages; ++i) {
    // Create the stages, connecting them up as we build.
    auto stage = rewriter.create<PipelineStageOp>(loc, type, buffer.getClk(),
                                                  buffer.getRst(), input);
    if (bufferName) {
      SmallString<64> stageName(
          {bufferName.getValue(), "_stage", std::to_string(i)});
      stage->setAttr("name", StringAttr::get(rewriter.getContext(), stageName));
    }
    input = stage;
  }

  // Replace the buffer.
  rewriter.replaceOp(buffer, input);
  return success();
}

namespace {
/// Lower pure modules into hw.modules.
struct PureModuleLowering : public OpConversionPattern<ESIPureModuleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ESIPureModuleOp pureMod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
PureModuleLowering::matchAndRewrite(ESIPureModuleOp pureMod, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto loc = pureMod.getLoc();
  Block *body = &pureMod.getBody().front();

  // Track existing names (so we can de-dup) and get op result when we want to
  // replace it with the block args.
  DenseMap<StringAttr, ESIPureModuleInputOp> inputPortNames;
  // Build the port list for `hw.module` construction.
  SmallVector<hw::PortInfo> ports;
  // List the input and output ops.
  SmallVector<ESIPureModuleInputOp> inputs;
  SmallVector<ESIPureModuleOutputOp> outputs;
  SmallVector<Attribute> params;

  for (Operation &op : llvm::make_early_inc_range(body->getOperations())) {
    if (auto port = dyn_cast<ESIPureModuleInputOp>(op)) {
      // If we already have an input port of the same name, replace the result
      // value with the previous one. Checking that the types match is done in
      // the pure module verifier.
      auto existingPort = inputPortNames.find(port.getNameAttr());
      if (existingPort != inputPortNames.end()) {
        rewriter.replaceAllUsesWith(port.getResult(),
                                    existingPort->getSecond().getResult());
        rewriter.eraseOp(port);
        continue;
      }
      // Normal port construction.
      ports.push_back(
          hw::PortInfo{{port.getNameAttr(), port.getResult().getType(),
                        hw::ModulePort::Direction::Input},
                       inputs.size(),
                       {},
                       {},
                       port.getLoc()});
      inputs.push_back(port);
    } else if (auto port = dyn_cast<ESIPureModuleOutputOp>(op)) {
      ports.push_back(
          hw::PortInfo{{port.getNameAttr(), port.getValue().getType(),
                        hw::ModulePort::Direction::Output},
                       outputs.size(),
                       {},
                       {},
                       port.getLoc()});
      outputs.push_back(port);
    } else if (auto param = dyn_cast<ESIPureModuleParamOp>(op)) {
      params.push_back(
          ParamDeclAttr::get(param.getNameAttr(), param.getType()));
      rewriter.eraseOp(param);
    }
  }

  // Create the replacement `hw.module`.
  auto hwMod = rewriter.create<hw::HWModuleOp>(
      loc, pureMod.getNameAttr(), ports, ArrayAttr::get(getContext(), params));
  hwMod->setDialectAttrs(pureMod->getDialectAttrs());
  rewriter.eraseBlock(hwMod.getBodyBlock());
  rewriter.inlineRegionBefore(*body->getParent(), hwMod.getBodyRegion(),
                              hwMod.getBodyRegion().end());
  body = hwMod.getBodyBlock();

  // Re-wire the inputs and erase them.
  for (auto input : inputs) {
    BlockArgument newArg;
    rewriter.updateRootInPlace(hwMod, [&]() {
      newArg = body->addArgument(input.getResult().getType(), input.getLoc());
    });
    rewriter.replaceAllUsesWith(input.getResult(), newArg);
    rewriter.eraseOp(input);
  }

  // Assemble the output values.
  SmallVector<Value> hwOutputOperands;
  for (auto output : outputs) {
    hwOutputOperands.push_back(output.getValue());
    rewriter.eraseOp(output);
  }
  rewriter.setInsertionPointToEnd(body);
  rewriter.create<hw::OutputOp>(pureMod.getLoc(), hwOutputOperands);

  // Erase the original op.
  rewriter.eraseOp(pureMod);
  return success();
}

namespace {
/// Run all the physical lowerings.
struct ESIToPhysicalPass : public LowerESIToPhysicalBase<ESIToPhysicalPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ESIToPhysicalPass::runOnOperation() {
  // Set up a conversion and give it a set of laws.
  ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addIllegalOp<ChannelBufferOp>();
  target.addIllegalOp<ESIPureModuleOp>();

  // Add all the conversion patterns.
  RewritePatternSet patterns(&getContext());
  patterns.insert<ChannelBufferLowering>(&getContext());
  patterns.insert<PureModuleLowering>(&getContext());

  // Run the conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIPhysicalLoweringPass() {
  return std::make_unique<ESIToPhysicalPass>();
}
