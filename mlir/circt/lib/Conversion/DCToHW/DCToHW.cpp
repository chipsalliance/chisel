//===- DCToHW.cpp - Translate DC into HW ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main DC to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/DCToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ConversionPatterns.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

using namespace mlir;
using namespace circt;
using namespace circt::dc;
using namespace circt::hw;

using NameUniquer = std::function<std::string(Operation *)>;

// NOLINTNEXTLINE(misc-no-recursion)
static Type tupleToStruct(TupleType tuple) {
  auto *ctx = tuple.getContext();
  mlir::SmallVector<hw::StructType::FieldInfo, 8> hwfields;
  for (auto [i, innerType] : llvm::enumerate(tuple)) {
    Type convertedInnerType = innerType;
    if (auto tupleInnerType = innerType.dyn_cast<TupleType>())
      convertedInnerType = tupleToStruct(tupleInnerType);
    hwfields.push_back(
        {StringAttr::get(ctx, "field" + Twine(i)), convertedInnerType});
  }

  return hw::StructType::get(ctx, hwfields);
}

/// Converts any type 't' into a `hw`-compatible type.
/// tuple -> hw.struct
/// none -> i0
/// (tuple[...] | hw.struct)[...] -> (tuple | hw.struct)[toHwType(...)]
// NOLINTNEXTLINE(misc-no-recursion)
static Type toHWType(Type t) {
  return TypeSwitch<Type, Type>(t)
      .Case([](TupleType tt) { return toHWType(tupleToStruct(tt)); })
      .Case([](hw::StructType st) {
        llvm::SmallVector<hw::StructType::FieldInfo> structFields(
            st.getElements());
        for (auto &field : structFields)
          field.type = toHWType(field.type);
        return hw::StructType::get(st.getContext(), structFields);
      })
      .Case([](NoneType nt) { return IntegerType::get(nt.getContext(), 0); })
      .Default([](Type t) { return t; });
}

static Type toESIHWType(Type t) {
  Type outType =
      llvm::TypeSwitch<Type, Type>(t)
          .Case([](ValueType vt) {
            return esi::ChannelType::get(vt.getContext(),
                                         toHWType(vt.getInnerType()));
          })
          .Case([](TokenType tt) {
            return esi::ChannelType::get(tt.getContext(),
                                         IntegerType::get(tt.getContext(), 0));
          })
          .Default([](auto t) { return toHWType(t); });

  return outType;
}

namespace {

/// Shared state used by various functions; captured in a struct to reduce the
/// number of arguments that we have to pass around.
struct DCLoweringState {
  ModuleOp parentModule;
  NameUniquer nameUniquer;
};

/// A type converter is needed to perform the in-flight materialization of "raw"
/// (non-ESI channel) types to their ESI channel correspondents. This comes into
/// effect when backedges exist in the input IR.
class ESITypeConverter : public TypeConverter {
public:
  ESITypeConverter() {
    addConversion([](Type type) -> Type { return toESIHWType(type); });
    addConversion([](esi::ChannelType t) -> Type { return t; });
    addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange inputs,
           mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;

          return inputs[0];
        });

    addSourceMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange inputs,
           mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;

          return inputs[0];
        });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// HW Sub-module Related Functions
//===----------------------------------------------------------------------===//

namespace {

/// Input handshakes contain a resolved valid and (optional )data signal, and
/// a to-be-assigned ready signal.
struct InputHandshake {
  Value channel;
  Value valid;
  std::optional<Backedge> ready;
  Value data;
};

/// Output handshakes contain a resolved ready, and to-be-assigned valid and
/// (optional) data signals.
struct OutputHandshake {
  Value channel;
  std::optional<Backedge> valid;
  Value ready;
  std::optional<Backedge> data;
};

///  Directly connect an input handshake to an output handshake
static void connect(InputHandshake &input, OutputHandshake &output) {
  output.valid->setValue(input.valid);
  input.ready->setValue(output.ready);
}

template <typename T, typename TInner>
llvm::SmallVector<T> extractValues(llvm::SmallVector<TInner> &container,
                                   llvm::function_ref<T(TInner &)> extractor) {
  llvm::SmallVector<T> result;
  llvm::transform(container, std::back_inserter(result), extractor);
  return result;
}

// Wraps a set of input and output handshakes with an API that provides
// access to collections of the underlying values.
struct UnwrappedIO {
  llvm::SmallVector<InputHandshake> inputs;
  llvm::SmallVector<OutputHandshake> outputs;

  llvm::SmallVector<Value> getInputValids() {
    return extractValues<Value, InputHandshake>(
        inputs, [](auto &hs) { return hs.valid; });
  }
  llvm::SmallVector<std::optional<Backedge>> getInputReadys() {
    return extractValues<std::optional<Backedge>, InputHandshake>(
        inputs, [](auto &hs) { return hs.ready; });
  }
  llvm::SmallVector<std::optional<Backedge>> getOutputValids() {
    return extractValues<std::optional<Backedge>, OutputHandshake>(
        outputs, [](auto &hs) { return hs.valid; });
  }
  llvm::SmallVector<Value> getInputDatas() {
    return extractValues<Value, InputHandshake>(
        inputs, [](auto &hs) { return hs.data; });
  }
  llvm::SmallVector<Value> getOutputReadys() {
    return extractValues<Value, OutputHandshake>(
        outputs, [](auto &hs) { return hs.ready; });
  }

  llvm::SmallVector<Value> getOutputChannels() {
    return extractValues<Value, OutputHandshake>(
        outputs, [](auto &hs) { return hs.channel; });
  }
  llvm::SmallVector<std::optional<Backedge>> getOutputDatas() {
    return extractValues<std::optional<Backedge>, OutputHandshake>(
        outputs, [](auto &hs) { return hs.data; });
  }
};

///  A class containing a bunch of syntactic sugar to reduce builder function
///  verbosity.
///  @todo: should be moved to support.
struct RTLBuilder {
  RTLBuilder(Location loc, OpBuilder &builder, Value clk = Value(),
             Value rst = Value())
      : b(builder), loc(loc), clk(clk), rst(rst) {}

  Value constant(const APInt &apv, StringRef name = {}) {
    // Cannot use zero-width APInt's in DenseMap's, see
    // https://github.com/llvm/llvm-project/issues/58013
    bool isZeroWidth = apv.getBitWidth() == 0;
    if (!isZeroWidth) {
      auto it = constants.find(apv);
      if (it != constants.end())
        return it->second;
    }

    auto cval = b.create<hw::ConstantOp>(loc, apv);
    if (!isZeroWidth)
      constants[apv] = cval;
    return cval;
  }

  Value constant(unsigned width, int64_t value, StringRef name = {}) {
    return constant(APInt(width, value));
  }
  std::pair<Value, Value> wrap(Value data, Value valid, StringRef name = {}) {
    auto wrapOp = b.create<esi::WrapValidReadyOp>(loc, data, valid);
    return {wrapOp.getResult(0), wrapOp.getResult(1)};
  }
  std::pair<Value, Value> unwrap(Value channel, Value ready,
                                 StringRef name = {}) {
    auto unwrapOp = b.create<esi::UnwrapValidReadyOp>(loc, channel, ready);
    return {unwrapOp.getResult(0), unwrapOp.getResult(1)};
  }

  ///  Various syntactic sugar functions.
  Value reg(StringRef name, Value in, Value rstValue, Value clk = Value(),
            Value rst = Value()) {
    Value resolvedClk = clk ? clk : this->clk;
    Value resolvedRst = rst ? rst : this->rst;
    assert(resolvedClk &&
           "No global clock provided to this RTLBuilder - a clock "
           "signal must be provided to the reg(...) function.");
    assert(resolvedRst &&
           "No global reset provided to this RTLBuilder - a reset "
           "signal must be provided to the reg(...) function.");

    return b.create<seq::CompRegOp>(loc, in.getType(), in, resolvedClk, name,
                                    resolvedRst, rstValue, hw::InnerSymAttr());
  }

  Value cmp(Value lhs, Value rhs, comb::ICmpPredicate predicate,
            StringRef name = {}) {
    return b.create<comb::ICmpOp>(loc, predicate, lhs, rhs);
  }

  Value buildNamedOp(llvm::function_ref<Value()> f, StringRef name) {
    Value v = f();
    StringAttr nameAttr;
    Operation *op = v.getDefiningOp();
    if (!name.empty()) {
      op->setAttr("sv.namehint", b.getStringAttr(name));
      nameAttr = b.getStringAttr(name);
    }
    return v;
  }

  ///  Bitwise 'and'.
  Value bitAnd(ValueRange values, StringRef name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::AndOp>(loc, values, false); }, name);
  }

  // Bitwise 'or'.
  Value bitOr(ValueRange values, StringRef name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::OrOp>(loc, values, false); }, name);
  }

  ///  Bitwise 'not'.
  Value bitNot(Value value, StringRef name = {}) {
    auto allOnes = constant(value.getType().getIntOrFloatBitWidth(), -1);
    std::string inferedName;
    if (!name.empty()) {
      // Try to create a name from the input value.
      if (auto valueName =
              value.getDefiningOp()->getAttrOfType<StringAttr>("sv.namehint")) {
        inferedName = ("not_" + valueName.getValue()).str();
        name = inferedName;
      }
    }

    return buildNamedOp(
        [&]() { return b.create<comb::XorOp>(loc, value, allOnes); }, name);
  }

  Value shl(Value value, Value shift, StringRef name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::ShlOp>(loc, value, shift); }, name);
  }

  Value concat(ValueRange values, StringRef name = {}) {
    return buildNamedOp([&]() { return b.create<comb::ConcatOp>(loc, values); },
                        name);
  }

  llvm::SmallVector<Value> extractBits(Value v, StringRef name = {}) {
    llvm::SmallVector<Value> bits;
    for (unsigned i = 0, e = v.getType().getIntOrFloatBitWidth(); i != e; ++i)
      bits.push_back(b.create<comb::ExtractOp>(loc, v, i, /*bitWidth=*/1));
    return bits;
  }

  ///  OR-reduction of the bits in 'v'.
  Value reduceOr(Value v, StringRef name = {}) {
    return buildNamedOp([&]() { return bitOr(extractBits(v)); }, name);
  }

  ///  Extract bits v[hi:lo] (inclusive).
  Value extract(Value v, unsigned lo, unsigned hi, StringRef name = {}) {
    unsigned width = hi - lo + 1;
    return buildNamedOp(
        [&]() { return b.create<comb::ExtractOp>(loc, v, lo, width); }, name);
  }

  ///  Truncates 'value' to its lower 'width' bits.
  Value truncate(Value value, unsigned width, StringRef name = {}) {
    return extract(value, 0, width - 1, name);
  }

  Value zext(Value value, unsigned outWidth, StringRef name = {}) {
    unsigned inWidth = value.getType().getIntOrFloatBitWidth();
    assert(inWidth <= outWidth && "zext: input width must be <= output width.");
    if (inWidth == outWidth)
      return value;
    auto c0 = constant(outWidth - inWidth, 0);
    return concat({c0, value}, name);
  }

  Value sext(Value value, unsigned outWidth, StringRef name = {}) {
    return comb::createOrFoldSExt(loc, value, b.getIntegerType(outWidth), b);
  }

  ///  Extracts a single bit v[bit].
  Value bit(Value v, unsigned index, StringRef name = {}) {
    return extract(v, index, index, name);
  }

  ///  Creates a hw.array of the given values.
  Value arrayCreate(ValueRange values, StringRef name = {}) {
    return buildNamedOp(
        [&]() { return b.create<hw::ArrayCreateOp>(loc, values); }, name);
  }

  ///  Extract the 'index'th value from the input array.
  Value arrayGet(Value array, Value index, StringRef name = {}) {
    return buildNamedOp(
        [&]() { return b.create<hw::ArrayGetOp>(loc, array, index); }, name);
  }

  ///  Muxes a range of values.
  ///  The select signal is expected to be a decimal value which selects
  ///  starting from the lowest index of value.
  Value mux(Value index, ValueRange values, StringRef name = {}) {
    if (values.size() == 2) {
      return buildNamedOp(
          [&]() {
            return b.create<comb::MuxOp>(loc, index, values[1], values[0]);
          },
          name);
    }
    return arrayGet(arrayCreate(values), index, name);
  }

  ///  Muxes a range of values. The select signal is expected to be a 1-hot
  ///  encoded value.
  Value oneHotMux(Value index, ValueRange inputs) {
    // Confirm the select input can be a one-hot encoding for the inputs.
    unsigned numInputs = inputs.size();
    assert(numInputs == index.getType().getIntOrFloatBitWidth() &&
           "mismatch between width of one-hot select input and the number of "
           "inputs to be selected");

    // Start the mux tree with zero value.
    auto dataType = inputs[0].getType();
    unsigned width =
        dataType.isa<NoneType>() ? 0 : dataType.getIntOrFloatBitWidth();
    Value muxValue = constant(width, 0);

    // Iteratively chain together muxes from the high bit to the low bit.
    for (size_t i = numInputs - 1; i != 0; --i) {
      Value input = inputs[i];
      Value selectBit = bit(index, i);
      muxValue = mux(selectBit, {muxValue, input});
    }

    return muxValue;
  }

  OpBuilder &b;
  Location loc;
  Value clk, rst;
  DenseMap<APInt, Value> constants;
};

static bool isZeroWidthType(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.getWidth() == 0;
  return type.isa<NoneType>();
}

static UnwrappedIO unwrapIO(Location loc, ValueRange operands,
                            TypeRange results,
                            ConversionPatternRewriter &rewriter,
                            BackedgeBuilder &bb) {
  RTLBuilder rtlb(loc, rewriter);
  UnwrappedIO unwrapped;
  for (auto in : operands) {
    assert(isa<esi::ChannelType>(in.getType()));
    auto ready = bb.get(rtlb.b.getI1Type());
    auto [data, valid] = rtlb.unwrap(in, ready);
    unwrapped.inputs.push_back(InputHandshake{in, valid, ready, data});
  }
  for (auto outputType : results) {
    outputType = toESIHWType(outputType);
    esi::ChannelType channelType = cast<esi::ChannelType>(outputType);
    OutputHandshake hs;
    Type innerType = channelType.getInner();
    Value data;
    if (isZeroWidthType(innerType)) {
      // Feed the ESI wrap with an i0 constant.
      data =
          rewriter.create<hw::ConstantOp>(loc, rewriter.getIntegerType(0), 0);
    } else {
      // Create a backedge for the unresolved data.
      auto dataBackedge = bb.get(innerType);
      hs.data = dataBackedge;
      data = dataBackedge;
    }
    auto valid = bb.get(rewriter.getI1Type());
    auto [dataCh, ready] = rtlb.wrap(data, valid);
    hs.valid = valid;
    hs.ready = ready;
    hs.channel = dataCh;
    unwrapped.outputs.push_back(hs);
  }
  return unwrapped;
}

static UnwrappedIO unwrapIO(Operation *op, ValueRange operands,
                            ConversionPatternRewriter &rewriter,
                            BackedgeBuilder &bb) {
  return unwrapIO(op->getLoc(), operands, op->getResultTypes(), rewriter, bb);
}

///  Locate the clock and reset values from the parent operation based on
///  attributes assigned to the arguments.
static FailureOr<std::pair<Value, Value>> getClockAndReset(Operation *op) {
  auto *parent = op->getParentOp();
  mlir::FunctionOpInterface parentFuncOp =
      dyn_cast<mlir::FunctionOpInterface>(parent);
  if (!parentFuncOp)
    return parent->emitOpError(
        "parent op does not implement FunctionOpInterface");

  SmallVector<DictionaryAttr> argAttrs;
  parentFuncOp.getAllArgAttrs(argAttrs);

  std::optional<size_t> clockIdx, resetIdx;

  for (auto [idx, attrs] : llvm::enumerate(argAttrs)) {
    if (attrs.get("dc.clock")) {
      if (clockIdx)
        return parent->emitOpError(
            "multiple arguments contains a 'dc.clock' attribute");
      clockIdx = idx;
    }

    if (attrs.get("dc.reset")) {
      if (resetIdx)
        return parent->emitOpError(
            "multiple arguments contains a 'dc.reset' attribute");
      resetIdx = idx;
    }
  }

  if (!clockIdx)
    return parent->emitOpError("no argument contains a 'dc.clock' attribute");

  if (!resetIdx)
    return parent->emitOpError("no argument contains a 'dc.reset' attribute");

  return {std::make_pair(parentFuncOp.getArgument(*clockIdx),
                         parentFuncOp.getArgument(*resetIdx))};
}

class ForkConversionPattern : public OpConversionPattern<ForkOp> {
public:
  using OpConversionPattern<ForkOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ForkOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    auto crRes = getClockAndReset(op);
    if (failed(crRes))
      return failure();
    auto [clock, reset] = *crRes;
    RTLBuilder rtlb(op.getLoc(), rewriter, clock, reset);
    UnwrappedIO io = unwrapIO(op, operands.getOperands(), rewriter, bb);

    auto &input = io.inputs[0];

    Value c0I1 = rtlb.constant(1, 0);
    llvm::SmallVector<Value> doneWires;
    for (auto [i, output] : llvm::enumerate(io.outputs)) {
      Backedge doneBE = bb.get(rtlb.b.getI1Type());
      Value emitted = rtlb.bitAnd({doneBE, rtlb.bitNot(*input.ready)});
      Value emittedReg =
          rtlb.reg("emitted_" + std::to_string(i), emitted, c0I1);
      Value outValid = rtlb.bitAnd({rtlb.bitNot(emittedReg), input.valid});
      output.valid->setValue(outValid);
      Value validReady = rtlb.bitAnd({output.ready, outValid});
      Value done =
          rtlb.bitOr({validReady, emittedReg}, "done" + std::to_string(i));
      doneBE.setValue(done);
      doneWires.push_back(done);
    }
    input.ready->setValue(rtlb.bitAnd(doneWires, "allDone"));

    rewriter.replaceOp(op, io.getOutputChannels());
    return success();
  }
};

class JoinConversionPattern : public OpConversionPattern<JoinOp> {
public:
  using OpConversionPattern<JoinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(JoinOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    UnwrappedIO io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    RTLBuilder rtlb(op.getLoc(), rewriter);
    auto &output = io.outputs[0];

    Value allValid = rtlb.bitAnd(io.getInputValids());
    output.valid->setValue(allValid);

    auto validAndReady = rtlb.bitAnd({output.ready, allValid});
    for (auto &input : io.inputs)
      input.ready->setValue(validAndReady);

    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }
};

class SelectConversionPattern : public OpConversionPattern<SelectOp> {
public:
  using OpConversionPattern<SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SelectOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    UnwrappedIO io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    RTLBuilder rtlb(op.getLoc(), rewriter);

    // Extract select signal from the unwrapped IO.
    auto select = io.inputs[0];
    io.inputs.erase(io.inputs.begin());
    buildMuxLogic(rtlb, io, select);

    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }

  // Builds mux logic for the given inputs and outputs.
  // Note: it is assumed that the caller has removed the 'select' signal from
  // the 'unwrapped' inputs and provide it as a separate argument.
  void buildMuxLogic(RTLBuilder &rtlb, UnwrappedIO &unwrapped,
                     InputHandshake &select) const {

    // ============================= Control logic =============================
    size_t numInputs = unwrapped.inputs.size();
    size_t selectWidth = llvm::Log2_64_Ceil(numInputs);
    Value truncatedSelect =
        select.data.getType().getIntOrFloatBitWidth() > selectWidth
            ? rtlb.truncate(select.data, selectWidth)
            : select.data;

    // Decimal-to-1-hot decoder. 'shl' operands must be identical in size.
    auto selectZext = rtlb.zext(truncatedSelect, numInputs);
    auto select1h = rtlb.shl(rtlb.constant(numInputs, 1), selectZext);
    auto &res = unwrapped.outputs[0];

    // Mux input valid signals.
    auto selectedInputValid =
        rtlb.mux(truncatedSelect, unwrapped.getInputValids());
    // Result is valid when the selected input and the select input is valid.
    auto selAndInputValid = rtlb.bitAnd({selectedInputValid, select.valid});
    res.valid->setValue(selAndInputValid);
    auto resValidAndReady = rtlb.bitAnd({selAndInputValid, res.ready});

    // Select is ready when result is valid and ready (result transacting).
    select.ready->setValue(resValidAndReady);

    // Assign each input ready signal if it is currently selected.
    for (auto [inIdx, in] : llvm::enumerate(unwrapped.inputs)) {
      // Extract the selection bit for this input.
      auto isSelected = rtlb.bit(select1h, inIdx);

      // '&' that with the result valid and ready, and assign to the input
      // ready signal.
      auto activeAndResultValidAndReady =
          rtlb.bitAnd({isSelected, resValidAndReady});
      in.ready->setValue(activeAndResultValidAndReady);
    }
  }
};

class BranchConversionPattern : public OpConversionPattern<BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(BranchOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    UnwrappedIO io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    RTLBuilder rtlb(op.getLoc(), rewriter);
    auto cond = io.inputs[0];
    auto trueRes = io.outputs[0];
    auto falseRes = io.outputs[1];

    // Connect valid signal of both results.
    trueRes.valid->setValue(rtlb.bitAnd({cond.data, cond.valid}));
    falseRes.valid->setValue(rtlb.bitAnd({rtlb.bitNot(cond.data), cond.valid}));

    // Connect ready signal of condition.
    Value selectedResultReady =
        rtlb.mux(cond.data, {falseRes.ready, trueRes.ready});
    Value condReady = rtlb.bitAnd({selectedResultReady, cond.valid});
    cond.ready->setValue(condReady);

    rewriter.replaceOp(op,
                       SmallVector<Value>{trueRes.channel, falseRes.channel});
    return success();
  }
};

class ToESIConversionPattern : public OpConversionPattern<ToESIOp> {
  // Essentially a no-op, seeing as the type converter does the heavy
  // lifting here.
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ToESIOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands.getOperands());
    return success();
  }
};

class FromESIConversionPattern : public OpConversionPattern<FromESIOp> {
  // Essentially a no-op, seeing as the type converter does the heavy
  // lifting here.
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FromESIOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands.getOperands());
    return success();
  }
};

class SinkConversionPattern : public OpConversionPattern<SinkOp> {
public:
  using OpConversionPattern<SinkOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SinkOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    UnwrappedIO io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    io.inputs[0].ready->setValue(
        RTLBuilder(op.getLoc(), rewriter).constant(1, 1));
    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }
};

class SourceConversionPattern : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    UnwrappedIO io = unwrapIO(op, operands.getOperands(), rewriter, bb);
    RTLBuilder rtlb(op.getLoc(), rewriter);
    io.outputs[0].valid->setValue(rtlb.constant(1, 1));
    io.outputs[0].data->setValue(rtlb.constant(0, 0));
    rewriter.replaceOp(op, io.outputs[0].channel);
    return success();
  }
};

class PackConversionPattern : public OpConversionPattern<PackOp> {
public:
  using OpConversionPattern<PackOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PackOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    UnwrappedIO io = unwrapIO(op, llvm::SmallVector<Value>{operands.getToken()},
                              rewriter, bb);
    RTLBuilder rtlb(op.getLoc(), rewriter);
    auto &input = io.inputs[0];
    auto &output = io.outputs[0];
    output.data->setValue(operands.getInput());
    connect(input, output);
    rewriter.replaceOp(op, output.channel);
    return success();
  }
};

class UnpackConversionPattern : public OpConversionPattern<UnpackOp> {
public:
  using OpConversionPattern<UnpackOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto bb = BackedgeBuilder(rewriter, op.getLoc());
    UnwrappedIO io = unwrapIO(
        op.getLoc(), llvm::SmallVector<Value>{operands.getInput()},
        // Only generate an output channel for the token typed output.
        llvm::SmallVector<Type>{op.getToken().getType()}, rewriter, bb);
    RTLBuilder rtlb(op.getLoc(), rewriter);
    auto &input = io.inputs[0];
    auto &output = io.outputs[0];

    llvm::SmallVector<Value> unpackedValues;
    unpackedValues.push_back(input.data);

    connect(input, output);
    llvm::SmallVector<Value> outputs;
    outputs.push_back(output.channel);
    outputs.append(unpackedValues.begin(), unpackedValues.end());
    rewriter.replaceOp(op, outputs);
    return success();
  }
};

class BufferConversionPattern : public OpConversionPattern<BufferOp> {
public:
  using OpConversionPattern<BufferOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BufferOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto crRes = getClockAndReset(op);
    if (failed(crRes))
      return failure();
    auto [clock, reset] = *crRes;

    // ... esi.buffer should in theory provide a correct (latency-insensitive)
    // implementation...
    Type channelType = operands.getInput().getType();
    rewriter.replaceOpWithNewOp<esi::ChannelBufferOp>(
        op, channelType, clock, reset, operands.getInput(), op.getSizeAttr(),
        nullptr);
    return success();
  };
};

} // namespace

static bool isDCType(Type type) { return type.isa<TokenType, ValueType>(); }

///  Returns true if the given `op` is considered as legal - i.e. it does not
///  contain any dc-typed values.
static bool isLegalOp(Operation *op) {
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    return llvm::none_of(funcOp.getArgumentTypes(), isDCType) &&
           llvm::none_of(funcOp.getResultTypes(), isDCType) &&
           llvm::none_of(funcOp.getFunctionBody().getArgumentTypes(), isDCType);
  }

  bool operandsOK = llvm::none_of(op->getOperandTypes(), isDCType);
  bool resultsOK = llvm::none_of(op->getResultTypes(), isDCType);
  return operandsOK && resultsOK;
}

//===----------------------------------------------------------------------===//
// HW Top-module Related Functions
//===----------------------------------------------------------------------===//

namespace {
class DCToHWPass : public DCToHWBase<DCToHWPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Lowering to HW requires that every DC-typed value is used exactly once.
    // Check whether this precondition is met, and if not, exit.
    auto walkRes = mod.walk([&](Operation *op) {
      for (auto res : op->getResults()) {
        if (res.getType().isa<dc::TokenType, dc::ValueType>()) {
          if (res.use_empty()) {
            op->emitOpError() << "DCToHW: value " << res << " is unused.";
            return WalkResult::interrupt();
          }
          if (!res.hasOneUse()) {
            op->emitOpError()
                << "DCToHW: value " << res << " has multiple uses.";
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });

    if (walkRes.wasInterrupted()) {
      mod->emitOpError()
          << "DCToHW: failed to verify that all values "
             "are used exactly once. Remember to run the "
             "fork/sink materialization pass before HW lowering.";
      signalPassFailure();
      return;
    }

    ESITypeConverter typeConverter;
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal(isLegalOp);

    // All top-level logic of a handshake module will be the interconnectivity
    // between instantiated modules.
    target.addIllegalDialect<dc::DCDialect>();

    RewritePatternSet patterns(mod.getContext());

    patterns.insert<ForkConversionPattern, JoinConversionPattern,
                    SelectConversionPattern, BranchConversionPattern,
                    PackConversionPattern, UnpackConversionPattern,
                    BufferConversionPattern, SourceConversionPattern,
                    SinkConversionPattern, TypeConversionPattern,
                    ToESIConversionPattern, FromESIConversionPattern>(
        typeConverter, mod.getContext());

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::createDCToHWPass() {
  return std::make_unique<DCToHWPass>();
}
