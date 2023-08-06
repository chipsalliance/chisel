//===- HandshakeToDC.cpp - Translate Handshake into DC --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Handshake to DC Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HandshakeToDC.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCTypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Handshake/Visitor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace mlir;
using namespace circt;
using namespace handshake;
using namespace dc;
using namespace hw;

namespace {

using ConvertedOps = DenseSet<Operation *>;

struct DCTuple {
  DCTuple() = default;
  DCTuple(Value token, Value data) : token(token), data(data) {}
  DCTuple(dc::UnpackOp unpack)
      : token(unpack.getToken()), data(unpack.getOutput()) {}
  Value token;
  Value data;
};

// Unpack a !dc.value<...> into a DCTuple.
static DCTuple unpack(OpBuilder &b, Value v) {
  if (v.getType().isa<dc::ValueType>())
    return DCTuple(b.create<dc::UnpackOp>(v.getLoc(), v));
  assert(v.getType().isa<dc::TokenType>() && "Expected a dc::TokenType");
  return DCTuple(v, {});
}

static Value pack(OpBuilder &b, Value token, Value data = {}) {
  if (!data)
    return token;
  return b.create<dc::PackOp>(token.getLoc(), token, data);
}

class DCTypeConverter : public TypeConverter {
public:
  DCTypeConverter() {
    addConversion([](Type type) -> Type {
      if (type.isa<NoneType>())
        return dc::TokenType::get(type.getContext());
      return dc::ValueType::get(type.getContext(), type);
    });
    addConversion([](ValueType type) { return type; });
    addConversion([](TokenType type) { return type; });

    addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange inputs,
           mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;

          // Materialize !dc.value<> -> !dc.token
          if (resultType.isa<dc::TokenType>() &&
              inputs.front().getType().isa<dc::ValueType>())
            return unpack(builder, inputs.front()).token;

          // Materialize !dc.token -> !dc.value<>
          auto vt = resultType.dyn_cast<dc::ValueType>();
          if (vt && !vt.getInnerType())
            return pack(builder, inputs.front());

          return inputs[0];
        });

    addSourceMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange inputs,
           mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;

          // Materialize !dc.value<> -> !dc.token
          if (resultType.isa<dc::TokenType>() &&
              inputs.front().getType().isa<dc::ValueType>())
            return unpack(builder, inputs.front()).token;

          // Materialize !dc.token -> !dc.value<>
          auto vt = resultType.dyn_cast<dc::ValueType>();
          if (vt && !vt.getInnerType())
            return pack(builder, inputs.front());

          return inputs[0];
        });
  }
};

template <typename OpTy>
class DCOpConversionPattern : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  DCOpConversionPattern(MLIRContext *context, TypeConverter &typeConverter,
                        ConvertedOps *convertedOps)
      : OpConversionPattern<OpTy>(typeConverter, context),
        convertedOps(convertedOps) {}
  mutable ConvertedOps *convertedOps;
};

class CondBranchConversionPattern
    : public DCOpConversionPattern<handshake::ConditionalBranchOp> {
public:
  using DCOpConversionPattern<
      handshake::ConditionalBranchOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::ConditionalBranchOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::ConditionalBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto condition = unpack(rewriter, adaptor.getConditionOperand());
    auto data = unpack(rewriter, adaptor.getDataOperand());

    // Join the token of the condition and the input.
    auto join = rewriter.create<dc::JoinOp>(
        op.getLoc(), ValueRange{condition.token, data.token});

    // Pack that together with the condition data.
    auto packedCondition = pack(rewriter, join, condition.data);

    // Branch on the input data and the joined control input.
    auto branch = rewriter.create<dc::BranchOp>(op.getLoc(), packedCondition);

    // Pack the branch output tokens with the input data, and replace the uses.
    llvm::SmallVector<Value, 4> packed;
    packed.push_back(pack(rewriter, branch.getTrueToken(), data.data));
    packed.push_back(pack(rewriter, branch.getFalseToken(), data.data));

    rewriter.replaceOp(op, packed);
    return success();
  }
};

class ForkOpConversionPattern
    : public DCOpConversionPattern<handshake::ForkOp> {
public:
  using DCOpConversionPattern<handshake::ForkOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::ForkOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::ForkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = unpack(rewriter, adaptor.getOperand());
    auto forkOut = rewriter.create<dc::ForkOp>(op.getLoc(), input.token,
                                               op.getNumResults());

    // Pack the fork result tokens with the input data, and replace the uses.
    llvm::SmallVector<Value, 4> packed;
    for (auto res : forkOut.getResults())
      packed.push_back(pack(rewriter, res, input.data));

    rewriter.replaceOp(op, packed);
    return success();
  }
};

class JoinOpConversion : public DCOpConversionPattern<handshake::JoinOp> {
public:
  using DCOpConversionPattern<handshake::JoinOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::JoinOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value, 4> inputTokens;
    for (auto input : adaptor.getData())
      inputTokens.push_back(unpack(rewriter, input).token);

    rewriter.replaceOpWithNewOp<dc::JoinOp>(op, inputTokens);
    return success();
  }
};

class ControlMergeOpConversion
    : public DCOpConversionPattern<handshake::ControlMergeOp> {
public:
  using DCOpConversionPattern<handshake::ControlMergeOp>::DCOpConversionPattern;

  using OpAdaptor = typename handshake::ControlMergeOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::ControlMergeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getDataOperands().size() != 2)
      return op.emitOpError("expected two data operands");

    llvm::SmallVector<Value> tokens, data;
    for (auto input : adaptor.getDataOperands()) {
      auto up = unpack(rewriter, input);
      tokens.push_back(up.token);
      if (up.data)
        data.push_back(up.data);
    }

    // control-side
    Value selectedIndex = rewriter.create<dc::MergeOp>(op.getLoc(), tokens);
    auto mergeOpUnpacked = unpack(rewriter, selectedIndex);
    auto selValue = mergeOpUnpacked.data;

    Value dataSide = selectedIndex;
    if (!data.empty()) {
      // Data side mux using the selected input.
      auto dataMux = rewriter.create<arith::SelectOp>(op.getLoc(), selValue,
                                                      data[0], data[1]);
      convertedOps->insert(dataMux);
      // Pack the data mux with the control token.
      auto packed = pack(rewriter, mergeOpUnpacked.token, dataMux);

      dataSide = packed;
    }

    // if the original op used `index` as the select operand type, we need to
    // index-cast the unpacked select operand
    if (op.getIndex().getType().isa<IndexType>()) {
      selValue = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), selValue);
      convertedOps->insert(selValue.getDefiningOp());
      selectedIndex = pack(rewriter, mergeOpUnpacked.token, selValue);
    }

    rewriter.replaceOp(op, {dataSide, selectedIndex});
    return success();
  }
};

class SyncOpConversion : public DCOpConversionPattern<handshake::SyncOp> {
public:
  using DCOpConversionPattern<handshake::SyncOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::SyncOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::SyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value, 4> inputTokens;
    for (auto input : adaptor.getOperands())
      inputTokens.push_back(unpack(rewriter, input).token);

    auto syncToken = rewriter.create<dc::JoinOp>(op.getLoc(), inputTokens);

    // Wrap all outputs with the synchronization token
    llvm::SmallVector<Value, 4> wrappedInputs;
    for (auto input : adaptor.getOperands())
      wrappedInputs.push_back(pack(rewriter, syncToken, input));

    rewriter.replaceOp(op, wrappedInputs);

    return success();
  }
};

class ConstantOpConversion
    : public DCOpConversionPattern<handshake::ConstantOp> {
public:
  using DCOpConversionPattern<handshake::ConstantOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::ConstantOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Wrap the constant with a token.
    auto token = rewriter.create<dc::SourceOp>(op.getLoc());
    auto cst =
        rewriter.create<arith::ConstantOp>(op.getLoc(), adaptor.getValue());
    convertedOps->insert(cst);
    rewriter.replaceOp(op, pack(rewriter, token, cst));
    return success();
  }
};

struct UnitRateConversionPattern : public ConversionPattern {
public:
  UnitRateConversionPattern(MLIRContext *context, TypeConverter &converter,
                            ConvertedOps *joinedOps)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context),
        joinedOps(joinedOps) {}
  using ConversionPattern::ConversionPattern;

  // Generic pattern which replaces an operation by one of the same type, but
  // with the in- and outputs synchronized through join semantics.
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return op->emitOpError("expected single result for pattern to apply");

    llvm::SmallVector<Value, 4> inputData;
    llvm::SmallVector<Value, 4> inputTokens;
    for (auto input : operands) {
      auto dct = unpack(rewriter, input);
      inputData.push_back(dct.data);
      inputTokens.push_back(dct.token);
    }

    // Join the tokens of the inputs.
    auto join = rewriter.create<dc::JoinOp>(op->getLoc(), inputTokens);

    // Patchwork to fix bad IR design in Handshake.
    auto opName = op->getName();
    if (opName.getStringRef() == "handshake.select") {
      opName = OperationName("arith.select", getContext());
    } else if (opName.getStringRef() == "handshake.constant") {
      opName = OperationName("arith.constant", getContext());
    }

    // Re-create the operation using the unpacked input data.
    OperationState state(op->getLoc(), opName, inputData, op->getResultTypes(),
                         op->getAttrs(), op->getSuccessors());

    Operation *newOp = rewriter.create(state);
    joinedOps->insert(newOp);

    // Pack the result token with the output data, and replace the use.
    rewriter.replaceOp(op, ValueRange{pack(rewriter, join.getResult(),
                                           newOp->getResults().front())});

    return success();
  }

  mutable ConvertedOps *joinedOps;
};

class SinkOpConversionPattern
    : public DCOpConversionPattern<handshake::SinkOp> {
public:
  using DCOpConversionPattern<handshake::SinkOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::SinkOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::SinkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = unpack(rewriter, adaptor.getOperand());
    rewriter.replaceOpWithNewOp<dc::SinkOp>(op, input.token);
    return success();
  }
};

class SourceOpConversionPattern
    : public DCOpConversionPattern<handshake::SourceOp> {
public:
  using DCOpConversionPattern<handshake::SourceOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<dc::SourceOp>(op);
    return success();
  }
};

class BufferOpConversion : public DCOpConversionPattern<handshake::BufferOp> {
public:
  using DCOpConversionPattern<handshake::BufferOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::BufferOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::BufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.getI32IntegerAttr(1);
    rewriter.replaceOpWithNewOp<dc::BufferOp>(
        op, adaptor.getOperand(), static_cast<size_t>(op.getNumSlots()));
    return success();
  }
};

class ReturnOpConversion : public DCOpConversionPattern<handshake::ReturnOp> {
public:
  using DCOpConversionPattern<handshake::ReturnOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::ReturnOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Locate existing output op, Append operands to output op, and move to
    // the end of the block.
    auto hwModule = op->getParentOfType<hw::HWModuleOp>();
    auto outputOp = *hwModule.getBodyBlock()->getOps<hw::OutputOp>().begin();
    outputOp->setOperands(adaptor.getOperands());
    outputOp->moveAfter(&hwModule.getBodyBlock()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

class MuxOpConversionPattern : public DCOpConversionPattern<handshake::MuxOp> {
public:
  using DCOpConversionPattern<handshake::MuxOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::MuxOp::Adaptor;

  LogicalResult
  matchAndRewrite(handshake::MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto select = unpack(rewriter, adaptor.getSelectOperand());
    auto selectData = select.data;
    auto selectToken = select.token;
    bool isIndexType = selectData.getType().isa<IndexType>();

    bool withData = !op.getResult().getType().isa<NoneType>();

    llvm::SmallVector<DCTuple> inputs;
    for (auto input : adaptor.getDataOperands())
      inputs.push_back(unpack(rewriter, input));

    Value dataMux;
    Value controlMux = inputs.front().token;
    // Convert the data-side mux to a sequence of arith.select operations.
    // The data and control muxes are assumed one-hot and the base-case is set
    // as the first input.
    if (withData)
      dataMux = inputs[0].data;

    llvm::SmallVector<Value> controlMuxInputs = {inputs.front().token};
    for (auto [i, input] :
         llvm::enumerate(llvm::make_range(inputs.begin() + 1, inputs.end()))) {
      if (!withData)
        continue;

      Value cmpIndex;
      Value inputData = input.data;
      Value inputControl = input.token;
      if (isIndexType) {
        cmpIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
      } else {
        size_t width = selectData.getType().cast<IntegerType>().getWidth();
        cmpIndex = rewriter.create<arith::ConstantIntOp>(op.getLoc(), i, width);
      }
      auto inputSelected = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::eq, selectData, cmpIndex);
      dataMux = rewriter.create<arith::SelectOp>(op.getLoc(), inputSelected,
                                                 inputData, dataMux);

      // Legalize the newly created operations.
      convertedOps->insert(cmpIndex.getDefiningOp());
      convertedOps->insert(dataMux.getDefiningOp());
      convertedOps->insert(inputSelected);

      // And similarly for the control mux, by muxing the input token with a
      // select value that has it's control from the original select token +
      // the inputSelected value.
      auto inputSelectedControl = pack(rewriter, selectToken, inputSelected);
      controlMux = rewriter.create<dc::SelectOp>(
          op.getLoc(), inputSelectedControl, inputControl, controlMux);
      convertedOps->insert(controlMux.getDefiningOp());
    }

    // finally, pack the control and data side muxes into the output value.
    rewriter.replaceOp(
        op, pack(rewriter, controlMux, withData ? dataMux : Value{}));
    return success();
  }
};

static hw::ModulePortInfo getModulePortInfoHS(TypeConverter &tc,
                                              handshake::FuncOp funcOp) {
  SmallVector<hw::PortInfo> inputs, outputs;
  auto *ctx = funcOp->getContext();
  auto ft = funcOp.getFunctionType();

  // Add all inputs of funcOp.
  for (auto [index, type] : llvm::enumerate(ft.getInputs())) {
    inputs.push_back({{StringAttr::get(ctx, "in" + std::to_string(index)),
                       tc.convertType(type), hw::ModulePort::Direction::Input},
                      index,
                      hw::InnerSymAttr{}});
  }

  // Add all outputs of funcOp.
  for (auto [index, type] : llvm::enumerate(ft.getResults())) {
    outputs.push_back(
        {{StringAttr::get(ctx, "out" + std::to_string(index)),
          tc.convertType(type), hw::ModulePort::Direction::Output},
         index,
         hw::InnerSymAttr{}});
  }

  return hw::ModulePortInfo{inputs, outputs};
}

class FuncOpConversion : public DCOpConversionPattern<handshake::FuncOp> {
public:
  using DCOpConversionPattern<handshake::FuncOp>::DCOpConversionPattern;
  using OpAdaptor = typename handshake::FuncOp::Adaptor;

  // Replaces a handshake.func with a hw.module, converting the argument and
  // result types using the provided type converter.
  // @mortbopet: Not a fan of converting to hw here seeing as we don't
  // necessarily have hardware semantics here. But, DC doesn't define a function
  // operation, and there is no "func.graph_func" or any other generic function
  // operation which is a graph region...
  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModulePortInfo ports = getModulePortInfoHS(*getTypeConverter(), op);

    if (op.isExternal()) {
      rewriter.create<hw::HWModuleExternOp>(
          op.getLoc(), rewriter.getStringAttr(op.getName()), ports);
    } else {
      auto hwModule = rewriter.create<hw::HWModuleOp>(
          op.getLoc(), rewriter.getStringAttr(op.getName()), ports);

      auto &region = op->getRegions().front();

      Region &moduleRegion = hwModule->getRegions().front();
      rewriter.mergeBlocks(&region.getBlocks().front(), hwModule.getBodyBlock(),
                           hwModule.getBodyBlock()->getArguments());
      TypeConverter::SignatureConversion result(moduleRegion.getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          TypeRange(moduleRegion.getArgumentTypes()), result);
      rewriter.applySignatureConversion(&moduleRegion, result);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class HandshakeToDCPass : public HandshakeToDCBase<HandshakeToDCPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Maintain the set of operations which has been converted either through
    // unit rate conversion, or as part of other conversions.
    // Rationale:
    // This is needed for all of the arith ops that get created as part of the
    // handshake ops (e.g. arith.select for handshake.mux). There's a bit of a
    // dilemma here seeing as all operations need to be converted/touched in a
    // handshake.func - which is done so by UnitRateConversionPattern (when no
    // other pattern applies). However, we obviously don't want to run said
    // pattern on these newly created ops since they do not have handshake
    // semantics.
    ConvertedOps convertedOps;

    ConversionTarget target(getContext());
    target.addIllegalDialect<handshake::HandshakeDialect>();
    target.addLegalDialect<dc::DCDialect, func::FuncDialect, hw::HWDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    // The various patterns will insert new operations into the module to
    // facilitate the conversion - however, these operations must be
    // distinguishable from already converted operations (which may be of the
    // same type as the newly inserted operations). To do this, we mark all
    // operations which have been converted as legal, and all other operations
    // as illegal.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return convertedOps.contains(op); });

    DCTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());

    // Add handshake conversion patterns.
    // Note: merge/control merge are not supported - these are non-deterministic
    // operators and we do not care for them.
    patterns
        .add<FuncOpConversion, BufferOpConversion, CondBranchConversionPattern,
             SinkOpConversionPattern, SourceOpConversionPattern,
             MuxOpConversionPattern, ReturnOpConversion,
             ForkOpConversionPattern, JoinOpConversion,
             ControlMergeOpConversion, ConstantOpConversion, SyncOpConversion>(
            &getContext(), typeConverter, &convertedOps);

    // ALL other single-result operations are converted via the
    // UnitRateConversionPattern.
    patterns.add<UnitRateConversionPattern>(&getContext(), typeConverter,
                                            &convertedOps);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::createHandshakeToDCPass() {
  return std::make_unique<HandshakeToDCPass>();
}
