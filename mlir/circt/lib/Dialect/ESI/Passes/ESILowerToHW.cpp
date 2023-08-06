//===- ESILowerToHW.cpp - Lower ESI to HW -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower to HW/SV conversions and pass.
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

#ifdef CAPNP
#include "../capnp/ESICapnp.h"
#endif

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;
using namespace circt::sv;

namespace {
/// Lower PipelineStageOp ops to an HW implementation. Unwrap and re-wrap
/// appropriately. Another conversion will take care merging the resulting
/// adjacent wrap/unwrap ops.
struct PipelineStageLowering : public OpConversionPattern<PipelineStageOp> {
public:
  PipelineStageLowering(ESIHWBuilder &builder, MLIRContext *ctxt)
      : OpConversionPattern(ctxt), builder(builder) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PipelineStageOp stage, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIHWBuilder &builder;
};
} // anonymous namespace

LogicalResult PipelineStageLowering::matchAndRewrite(
    PipelineStageOp stage, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = stage.getLoc();
  auto chPort = stage.getInput().getType().dyn_cast<ChannelType>();
  if (!chPort)
    return rewriter.notifyMatchFailure(stage, "stage had wrong type");
  Operation *symTable = stage->getParentWithTrait<OpTrait::SymbolTable>();
  auto stageModule = builder.declareStage(symTable, stage);

  size_t width = circt::hw::getBitWidth(chPort.getInner());

  ArrayAttr stageParams =
      builder.getStageParameterList(rewriter.getUI32IntegerAttr(width));

  // Unwrap the channel. The ready signal is a Value we haven't created yet,
  // so create a temp value and replace it later. Give this constant an
  // odd-looking type to make debugging easier.
  circt::BackedgeBuilder back(rewriter, loc);
  circt::Backedge wrapReady = back.get(rewriter.getI1Type());
  auto unwrap =
      rewriter.create<UnwrapValidReadyOp>(loc, stage.getInput(), wrapReady);

  StringRef pipeStageName = "pipelineStage";
  if (auto name = stage->getAttrOfType<StringAttr>("name"))
    pipeStageName = name.getValue();

  // Instantiate the "ESI_PipelineStage" external module.
  circt::Backedge stageReady = back.get(rewriter.getI1Type());
  llvm::SmallVector<Value> operands = {stage.getClk(), stage.getRst()};
  operands.push_back(unwrap.getRawOutput());
  operands.push_back(unwrap.getValid());
  operands.push_back(stageReady);
  auto stageInst = rewriter.create<InstanceOp>(loc, stageModule, pipeStageName,
                                               operands, stageParams);
  auto stageInstResults = stageInst.getResults();

  // Set a_ready (from the unwrap) back edge correctly to its output from
  // stage.
  wrapReady.setValue(stageInstResults[0]);
  Value x, xValid;
  x = stageInstResults[1];
  xValid = stageInstResults[2];

  // Wrap up the output of the HW stage module.
  auto wrap = rewriter.create<WrapValidReadyOp>(
      loc, chPort, rewriter.getI1Type(), x, xValid);
  // Set the stages x_ready backedge correctly.
  stageReady.setValue(wrap.getReady());

  rewriter.replaceOp(stage, wrap.getChanOutput());
  return success();
}

namespace {
struct NullSourceOpLowering : public OpConversionPattern<NullSourceOp> {
public:
  NullSourceOpLowering(MLIRContext *ctxt) : OpConversionPattern(ctxt) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NullSourceOp nullop, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult NullSourceOpLowering::matchAndRewrite(
    NullSourceOp nullop, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto innerType = nullop.getOut().getType().cast<ChannelType>().getInner();
  Location loc = nullop.getLoc();
  int64_t width = hw::getBitWidth(innerType);
  if (width == -1)
    return rewriter.notifyMatchFailure(
        nullop, "NullOp lowering only supports hw types");
  auto valid =
      rewriter.create<hw::ConstantOp>(nullop.getLoc(), rewriter.getI1Type(), 0);
  auto zero =
      rewriter.create<hw::ConstantOp>(loc, rewriter.getIntegerType(width), 0);
  auto typedZero = rewriter.create<hw::BitcastOp>(loc, innerType, zero);
  auto wrap = rewriter.create<WrapValidReadyOp>(loc, typedZero, valid);
  wrap->setAttr("name", rewriter.getStringAttr("nullsource"));
  rewriter.replaceOp(nullop, {wrap.getChanOutput()});
  return success();
}

namespace {
/// Eliminate back-to-back wrap-unwraps to reduce the number of ESI channels.
struct RemoveWrapUnwrap : public ConversionPattern {
public:
  RemoveWrapUnwrap(MLIRContext *context)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Value valid, ready, data;
    WrapValidReadyOp wrap = dyn_cast<WrapValidReadyOp>(op);
    UnwrapValidReadyOp unwrap = dyn_cast<UnwrapValidReadyOp>(op);
    if (wrap) {
      if (!wrap.getChanOutput().hasOneUse() ||
          !(unwrap = dyn_cast<UnwrapValidReadyOp>(
                wrap.getChanOutput().use_begin()->getOwner())))
        return rewriter.notifyMatchFailure(
            wrap, "This conversion only supports wrap-unwrap back-to-back. "
                  "Could not find 'unwrap'.");

      data = operands[0];
      valid = operands[1];
      ready = unwrap.getReady();
    } else if (unwrap) {
      wrap = dyn_cast<WrapValidReadyOp>(operands[0].getDefiningOp());
      if (!wrap)
        return rewriter.notifyMatchFailure(
            operands[0].getDefiningOp(),
            "This conversion only supports wrap-unwrap back-to-back. "
            "Could not find 'wrap'.");
      valid = wrap.getValid();
      data = wrap.getRawInput();
      ready = operands[1];
    } else {
      return failure();
    }

    if (!wrap.getChanOutput().hasOneUse())
      return rewriter.notifyMatchFailure(wrap, [](Diagnostic &d) {
        d << "This conversion only supports wrap-unwrap back-to-back. "
             "Wrap didn't have exactly one use.";
      });
    rewriter.replaceOp(wrap, {nullptr, ready});
    rewriter.replaceOp(unwrap, {data, valid});
    return success();
  }
};
} // anonymous namespace

namespace {
/// Use the op canonicalizer to lower away the op. Assumes the canonicalizer
/// deletes the op.
template <typename Op>
struct CanonicalizerOpLowering : public OpConversionPattern<Op> {
public:
  CanonicalizerOpLowering(MLIRContext *ctxt) : OpConversionPattern<Op>(ctxt) {}

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (failed(Op::canonicalize(op, rewriter)))
      return rewriter.notifyMatchFailure(op->getLoc(), "canonicalizer failed");
    return success();
  }
};
} // anonymous namespace

namespace {
struct ESItoHWPass : public LowerESItoHWBase<ESItoHWPass> {
  void runOnOperation() override;
};
} // anonymous namespace

namespace {
/// Lower a `wrap.iface` to `wrap.vr` by extracting the wires then feeding the
/// new `wrap.vr`.
struct WrapInterfaceLower : public OpConversionPattern<WrapSVInterfaceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WrapSVInterfaceOp wrap, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
WrapInterfaceLower::matchAndRewrite(WrapSVInterfaceOp wrap, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  if (operands.size() != 1)
    return rewriter.notifyMatchFailure(wrap, [&operands](Diagnostic &d) {
      d << "wrap.iface has 1 argument. Got " << operands.size() << "operands";
    });
  auto sinkModport = dyn_cast<GetModportOp>(operands[0].getDefiningOp());
  if (!sinkModport)
    return failure();
  auto ifaceInstance =
      dyn_cast<InterfaceInstanceOp>(sinkModport.getIface().getDefiningOp());
  if (!ifaceInstance)
    return failure();

  auto loc = wrap.getLoc();
  auto validSignal = rewriter.create<ReadInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::validStr);
  Value dataSignal;
  dataSignal = rewriter.create<ReadInterfaceSignalOp>(loc, ifaceInstance,
                                                      ESIHWBuilder::dataStr);
  auto wrapVR = rewriter.create<WrapValidReadyOp>(loc, dataSignal, validSignal);
  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::readyStr, wrapVR.getReady());
  rewriter.replaceOp(wrap, {wrapVR.getChanOutput()});
  return success();
}

namespace {
/// Lower an unwrap interface to just extract the wires and feed them into an
/// `unwrap.vr`.
struct UnwrapInterfaceLower : public OpConversionPattern<UnwrapSVInterfaceOp> {
public:
  UnwrapInterfaceLower(MLIRContext *ctxt) : OpConversionPattern(ctxt) {}
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnwrapSVInterfaceOp wrap, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult UnwrapInterfaceLower::matchAndRewrite(
    UnwrapSVInterfaceOp unwrap, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto operands = adaptor.getOperands();
  if (operands.size() != 2)
    return rewriter.notifyMatchFailure(unwrap, [&operands](Diagnostic &d) {
      d << "Unwrap.iface has 2 arguments. Got " << operands.size()
        << "operands";
    });

  auto sourceModport = dyn_cast<GetModportOp>(operands[1].getDefiningOp());
  if (!sourceModport)
    return failure();
  auto ifaceInstance =
      dyn_cast<InterfaceInstanceOp>(sourceModport.getIface().getDefiningOp());
  if (!ifaceInstance)
    return failure();

  auto loc = unwrap.getLoc();
  auto readySignal = rewriter.create<ReadInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::readyStr);
  auto unwrapVR =
      rewriter.create<UnwrapValidReadyOp>(loc, operands[0], readySignal);
  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::validStr, unwrapVR.getValid());

  rewriter.create<AssignInterfaceSignalOp>(
      loc, ifaceInstance, ESIHWBuilder::dataStr, unwrapVR.getRawOutput());
  rewriter.eraseOp(unwrap);
  return success();
}

namespace {
/// Lower `CosimEndpointOp` ops to a SystemVerilog extern module and a Capnp
/// gasket op.
struct CosimLowering : public OpConversionPattern<CosimEndpointOp> {
public:
  CosimLowering(ESIHWBuilder &b)
      : OpConversionPattern(b.getContext(), 1), builder(b) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CosimEndpointOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  ESIHWBuilder &builder;
};
} // anonymous namespace

LogicalResult
CosimLowering::matchAndRewrite(CosimEndpointOp ep, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
#ifndef CAPNP
  (void)builder;
  return rewriter.notifyMatchFailure(
      ep, "Cosim lowering requires the ESI capnp plugin, which was disabled.");
#else
  auto loc = ep.getLoc();
  auto *ctxt = rewriter.getContext();
  auto operands = adaptor.getOperands();
  Value clk = operands[0];
  Value rst = operands[1];
  Value send = operands[2];

  circt::BackedgeBuilder bb(rewriter, loc);
  Type ui64Type =
      IntegerType::get(ctxt, 64, IntegerType::SignednessSemantics::Unsigned);
  capnp::CapnpTypeSchema sendTypeSchema(send.getType());
  if (!sendTypeSchema.isSupported())
    return rewriter.notifyMatchFailure(ep, "Send type not supported yet");
  capnp::CapnpTypeSchema recvTypeSchema(ep.getRecv().getType());
  if (!recvTypeSchema.isSupported())
    return rewriter.notifyMatchFailure(ep, "Recv type not supported yet");

  // Set all the parameters.
  SmallVector<Attribute, 8> params;
  if (auto ext = ep->getAttrOfType<StringAttr>("name_ext"))
    params.push_back(ParamDeclAttr::get("ENDPOINT_ID_EXT", ext));
  else
    params.push_back(
        ParamDeclAttr::get("ENDPOINT_ID_EXT", StringAttr::get(ctxt, "")));
  params.push_back(ParamDeclAttr::get(
      "SEND_TYPE_ID", IntegerAttr::get(ui64Type, sendTypeSchema.typeID())));
  params.push_back(
      ParamDeclAttr::get("SEND_TYPE_SIZE_BITS",
                         rewriter.getI32IntegerAttr(sendTypeSchema.size())));
  params.push_back(ParamDeclAttr::get(
      "RECV_TYPE_ID", IntegerAttr::get(ui64Type, recvTypeSchema.typeID())));
  params.push_back(
      ParamDeclAttr::get("RECV_TYPE_SIZE_BITS",
                         rewriter.getI32IntegerAttr(recvTypeSchema.size())));

  // Set up the egest route to drive the EP's send ports.
  ArrayType egestBitArrayType =
      ArrayType::get(rewriter.getI1Type(), sendTypeSchema.size());
  auto sendReady = bb.get(rewriter.getI1Type());
  UnwrapValidReadyOp unwrapSend =
      rewriter.create<UnwrapValidReadyOp>(loc, send, sendReady);
  auto encodeData = rewriter.create<CapnpEncodeOp>(loc, egestBitArrayType, clk,
                                                   unwrapSend.getValid(),
                                                   unwrapSend.getRawOutput());

  // Get information necessary for injest path.
  auto recvReady = bb.get(rewriter.getI1Type());
  ArrayType ingestBitArrayType =
      ArrayType::get(rewriter.getI1Type(), recvTypeSchema.size());

  // Build or get the cached Cosim Endpoint module parameterization.
  Operation *symTable = ep->getParentWithTrait<OpTrait::SymbolTable>();
  HWModuleExternOp endpoint = builder.declareCosimEndpointOp(
      symTable, egestBitArrayType, ingestBitArrayType);

  // Create replacement Cosim_Endpoint instance.
  StringAttr nameAttr = ep->getAttr("name").dyn_cast_or_null<StringAttr>();
  StringRef name = nameAttr ? nameAttr.getValue() : "CosimEndpointOp";
  Value epInstInputs[] = {
      clk, rst, recvReady, unwrapSend.getValid(), encodeData.getCapnpBits(),
  };

  auto cosimEpModule = rewriter.create<InstanceOp>(
      loc, endpoint, name, epInstInputs, ArrayAttr::get(ctxt, params));
  sendReady.setValue(cosimEpModule.getResult(2));

  // Set up the injest path.
  Value recvDataFromCosim = cosimEpModule.getResult(1);
  Value recvValidFromCosim = cosimEpModule.getResult(0);
  auto decodeData =
      rewriter.create<CapnpDecodeOp>(loc, recvTypeSchema.getType(), clk,
                                     recvValidFromCosim, recvDataFromCosim);
  WrapValidReadyOp wrapRecv = rewriter.create<WrapValidReadyOp>(
      loc, decodeData.getDecodedData(), recvValidFromCosim);
  recvReady.setValue(wrapRecv.getReady());

  // Replace the CosimEndpointOp op.
  rewriter.replaceOp(ep, wrapRecv.getChanOutput());

  return success();
#endif // CAPNP
}

namespace {
/// Lower the encode gasket to SV/HW.
struct EncoderLowering : public OpConversionPattern<CapnpEncodeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CapnpEncodeOp enc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
#ifndef CAPNP
    return rewriter.notifyMatchFailure(enc,
                                       "encode.capnp lowering requires the ESI "
                                       "capnp plugin, which was disabled.");
#else
    capnp::CapnpTypeSchema encodeType(enc.getDataToEncode().getType());
    if (!encodeType.isSupported())
      return rewriter.notifyMatchFailure(enc, "Type not supported yet");
    auto operands = adaptor.getOperands();
    Value encoderOutput = encodeType.buildEncoder(rewriter, operands[0],
                                                  operands[1], operands[2]);
    assert(encoderOutput && "Error in TypeSchema.buildEncoder()");
    rewriter.replaceOp(enc, encoderOutput);
    return success();
#endif
  }
};
} // anonymous namespace

namespace {
/// Lower the decode gasket to SV/HW.
struct DecoderLowering : public OpConversionPattern<CapnpDecodeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CapnpDecodeOp dec, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
#ifndef CAPNP
    return rewriter.notifyMatchFailure(dec,
                                       "decode.capnp lowering requires the ESI "
                                       "capnp plugin, which was disabled.");
#else
    capnp::CapnpTypeSchema decodeType(dec.getDecodedData().getType());
    if (!decodeType.isSupported())
      return rewriter.notifyMatchFailure(dec, "Type not supported yet");
    auto operands = adaptor.getOperands();
    Value decoderOutput = decodeType.buildDecoder(rewriter, operands[0],
                                                  operands[1], operands[2]);
    assert(decoderOutput && "Error in TypeSchema.buildDecoder()");
    rewriter.replaceOp(dec, decoderOutput);
    return success();
#endif
  }
};
} // namespace

void ESItoHWPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();

  // Set up a conversion and give it a set of laws.
  ConversionTarget pass1Target(*ctxt);
  pass1Target.addLegalDialect<comb::CombDialect>();
  pass1Target.addLegalDialect<HWDialect>();
  pass1Target.addLegalDialect<SVDialect>();
  pass1Target.addLegalOp<WrapValidReadyOp, UnwrapValidReadyOp>();
  pass1Target.addLegalOp<CapnpDecodeOp, CapnpEncodeOp>();

  pass1Target.addIllegalOp<WrapSVInterfaceOp, UnwrapSVInterfaceOp>();
  pass1Target.addIllegalOp<PipelineStageOp>();

  // Add all the conversion patterns.
  ESIHWBuilder esiBuilder(top);
  RewritePatternSet pass1Patterns(ctxt);
  pass1Patterns.insert<PipelineStageLowering>(esiBuilder, ctxt);
  pass1Patterns.insert<WrapInterfaceLower>(ctxt);
  pass1Patterns.insert<UnwrapInterfaceLower>(ctxt);
  pass1Patterns.insert<CosimLowering>(esiBuilder);
  pass1Patterns.insert<NullSourceOpLowering>(ctxt);

  // Run the conversion.
  if (failed(
          applyPartialConversion(top, pass1Target, std::move(pass1Patterns))))
    signalPassFailure();

  ConversionTarget pass2Target(*ctxt);
  pass2Target.addLegalDialect<comb::CombDialect>();
  pass2Target.addLegalDialect<HWDialect>();
  pass2Target.addLegalDialect<SVDialect>();
  pass2Target.addIllegalDialect<ESIDialect>();

  RewritePatternSet pass2Patterns(ctxt);
  pass2Patterns.insert<CanonicalizerOpLowering<UnwrapFIFOOp>>(ctxt);
  pass2Patterns.insert<CanonicalizerOpLowering<WrapFIFOOp>>(ctxt);
  pass2Patterns.insert<RemoveWrapUnwrap>(ctxt);
  pass2Patterns.insert<EncoderLowering>(ctxt);
  pass2Patterns.insert<DecoderLowering>(ctxt);
  if (failed(
          applyPartialConversion(top, pass2Target, std::move(pass2Patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::esi::createESItoHWPass() {
  return std::make_unique<ESItoHWPass>();
}
