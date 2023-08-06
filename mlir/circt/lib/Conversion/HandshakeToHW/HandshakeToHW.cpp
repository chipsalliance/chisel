//===- HandshakeToHW.cpp - Translate Handshake into HW ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Handshake to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HandshakeToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Handshake/Visitor.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt::hw;

using NameUniquer = std::function<std::string(Operation *)>;

namespace {

static Type tupleToStruct(TypeRange types) {
  return toValidType(mlir::TupleType::get(types[0].getContext(), types));
}

// Shared state used by various functions; captured in a struct to reduce the
// number of arguments that we have to pass around.
struct HandshakeLoweringState {
  ModuleOp parentModule;
  NameUniquer nameUniquer;
};

// A type converter is needed to perform the in-flight materialization of "raw"
// (non-ESI channel) types to their ESI channel correspondents. This comes into
// effect when backedges exist in the input IR.
class ESITypeConverter : public TypeConverter {
public:
  ESITypeConverter() {
    addConversion([](Type type) -> Type { return esiWrapper(type); });

    addTargetMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;
          return inputs[0];
        });

    addSourceMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> std::optional<mlir::Value> {
          if (inputs.size() != 1)
            return std::nullopt;
          return inputs[0];
        });
  }
};

} // namespace

/// Returns a submodule name resulting from an operation, without discriminating
/// type information.
static std::string getBareSubModuleName(Operation *oldOp) {
  // The dialect name is separated from the operation name by '.', which is not
  // valid in SystemVerilog module names. In case this name is used in
  // SystemVerilog output, replace '.' with '_'.
  std::string subModuleName = oldOp->getName().getStringRef().str();
  std::replace(subModuleName.begin(), subModuleName.end(), '.', '_');
  return subModuleName;
}

static std::string getCallName(Operation *op) {
  auto callOp = dyn_cast<handshake::InstanceOp>(op);
  return callOp ? callOp.getModule().str() : getBareSubModuleName(op);
}

/// Extracts the type of the data-carrying type of opType. If opType is an ESI
/// channel, getHandshakeBundleDataType extracts the data-carrying type, else,
/// assume that opType itself is the data-carrying type.
static Type getOperandDataType(Value op) {
  auto opType = op.getType();
  if (auto channelType = opType.dyn_cast<esi::ChannelType>())
    return channelType.getInner();
  return opType;
}

/// Filters NoneType's from the input.
static SmallVector<Type> filterNoneTypes(ArrayRef<Type> input) {
  SmallVector<Type> filterRes;
  llvm::copy_if(input, std::back_inserter(filterRes),
                [](Type type) { return !type.isa<NoneType>(); });
  return filterRes;
}

/// Returns a set of types which may uniquely identify the provided op. Return
/// value is <inputTypes, outputTypes>.
using DiscriminatingTypes = std::pair<SmallVector<Type>, SmallVector<Type>>;
static DiscriminatingTypes getHandshakeDiscriminatingTypes(Operation *op) {
  return TypeSwitch<Operation *, DiscriminatingTypes>(op)
      .Case<MemoryOp, ExternalMemoryOp>([&](auto memOp) {
        return DiscriminatingTypes{{},
                                   {memOp.getMemRefType().getElementType()}};
      })
      .Default([&](auto) {
        // By default, all in- and output types which is not a control type
        // (NoneType) are discriminating types.
        std::vector<Type> inTypes, outTypes;
        llvm::transform(op->getOperands(), std::back_inserter(inTypes),
                        getOperandDataType);
        llvm::transform(op->getResults(), std::back_inserter(outTypes),
                        getOperandDataType);
        return DiscriminatingTypes{filterNoneTypes(inTypes),
                                   filterNoneTypes(outTypes)};
      });
}

/// Get type name. Currently we only support integer or index types.
/// The emitted type aligns with the getFIRRTLType() method. Thus all integers
/// other than signed integers will be emitted as unsigned.
// NOLINTNEXTLINE(misc-no-recursion)
static std::string getTypeName(Location loc, Type type) {
  std::string typeName;
  // Builtin types
  if (type.isIntOrIndex()) {
    if (auto indexType = type.dyn_cast<IndexType>())
      typeName += "_ui" + std::to_string(indexType.kInternalStorageBitWidth);
    else if (type.isSignedInteger())
      typeName += "_si" + std::to_string(type.getIntOrFloatBitWidth());
    else
      typeName += "_ui" + std::to_string(type.getIntOrFloatBitWidth());
  } else if (auto tupleType = type.dyn_cast<TupleType>()) {
    typeName += "_tuple";
    for (auto elementType : tupleType.getTypes())
      typeName += getTypeName(loc, elementType);
  } else if (auto structType = type.dyn_cast<hw::StructType>()) {
    typeName += "_struct";
    for (auto element : structType.getElements())
      typeName += "_" + element.name.str() + getTypeName(loc, element.type);
  } else
    emitError(loc) << "unsupported data type '" << type << "'";

  return typeName;
}

/// Construct a name for creating HW sub-module.
static std::string getSubModuleName(Operation *oldOp) {
  if (auto instanceOp = dyn_cast<handshake::InstanceOp>(oldOp); instanceOp)
    return instanceOp.getModule().str();

  std::string subModuleName = getBareSubModuleName(oldOp);

  // Add value of the constant operation.
  if (auto constOp = dyn_cast<handshake::ConstantOp>(oldOp)) {
    if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
      auto intType = intAttr.getType();

      if (intType.isSignedInteger())
        subModuleName += "_c" + std::to_string(intAttr.getSInt());
      else if (intType.isUnsignedInteger())
        subModuleName += "_c" + std::to_string(intAttr.getUInt());
      else
        subModuleName += "_c" + std::to_string((uint64_t)intAttr.getInt());
    } else
      oldOp->emitError("unsupported constant type");
  }

  // Add discriminating in- and output types.
  auto [inTypes, outTypes] = getHandshakeDiscriminatingTypes(oldOp);
  if (!inTypes.empty())
    subModuleName += "_in";
  for (auto inType : inTypes)
    subModuleName += getTypeName(oldOp->getLoc(), inType);

  if (!outTypes.empty())
    subModuleName += "_out";
  for (auto outType : outTypes)
    subModuleName += getTypeName(oldOp->getLoc(), outType);

  // Add memory ID.
  if (auto memOp = dyn_cast<handshake::MemoryOp>(oldOp))
    subModuleName += "_id" + std::to_string(memOp.getId());

  // Add compare kind.
  if (auto comOp = dyn_cast<mlir::arith::CmpIOp>(oldOp))
    subModuleName += "_" + stringifyEnum(comOp.getPredicate()).str();

  // Add buffer information.
  if (auto bufferOp = dyn_cast<handshake::BufferOp>(oldOp)) {
    subModuleName += "_" + std::to_string(bufferOp.getNumSlots()) + "slots";
    if (bufferOp.isSequential())
      subModuleName += "_seq";
    else
      subModuleName += "_fifo";

    if (auto initValues = bufferOp.getInitValues()) {
      subModuleName += "_init";
      for (const Attribute e : *initValues) {
        assert(e.isa<IntegerAttr>());
        subModuleName +=
            "_" + std::to_string(e.dyn_cast<IntegerAttr>().getInt());
      }
    }
  }

  // Add control information.
  if (auto ctrlInterface = dyn_cast<handshake::ControlInterface>(oldOp);
      ctrlInterface && ctrlInterface.isControl()) {
    // Add some additional discriminating info for non-typed operations.
    subModuleName += "_" + std::to_string(oldOp->getNumOperands()) + "ins_" +
                     std::to_string(oldOp->getNumResults()) + "outs";
    subModuleName += "_ctrl";
  } else {
    assert(
        (!inTypes.empty() || !outTypes.empty()) &&
        "Insufficient discriminating type info generated for the operation!");
  }

  return subModuleName;
}

//===----------------------------------------------------------------------===//
// HW Sub-module Related Functions
//===----------------------------------------------------------------------===//

/// Check whether a submodule with the same name has been created elsewhere in
/// the top level module. Return the matched module operation if true, otherwise
/// return nullptr.
static HWModuleLike checkSubModuleOp(mlir::ModuleOp parentModule,
                                     StringRef modName) {
  if (auto mod = parentModule.lookupSymbol<HWModuleOp>(modName))
    return mod;
  if (auto mod = parentModule.lookupSymbol<HWModuleExternOp>(modName))
    return mod;
  return {};
}

static HWModuleLike checkSubModuleOp(mlir::ModuleOp parentModule,
                                     Operation *oldOp) {
  HWModuleLike targetModule;
  if (auto instanceOp = dyn_cast<handshake::InstanceOp>(oldOp))
    targetModule = checkSubModuleOp(parentModule, instanceOp.getModule());
  else
    targetModule = checkSubModuleOp(parentModule, getSubModuleName(oldOp));

  if (isa<handshake::InstanceOp>(oldOp))
    assert(targetModule &&
           "handshake.instance target modules should always have been lowered "
           "before the modules that reference them!");
  return targetModule;
}

/// Returns a vector of PortInfo's which defines the HW interface of the
/// to-be-converted op.
static ModulePortInfo getPortInfoForOp(Operation *op) {
  return getPortInfoForOpTypes(op, op->getOperandTypes(), op->getResultTypes());
}

static llvm::SmallVector<hw::detail::FieldInfo>
portToFieldInfo(llvm::ArrayRef<hw::PortInfo> portInfo) {
  llvm::SmallVector<hw::detail::FieldInfo> fieldInfo;
  for (auto port : portInfo)
    fieldInfo.push_back({port.name, port.type});

  return fieldInfo;
}

// Convert any handshake.extmemory operations and the top-level I/O
// associated with these.
static LogicalResult convertExtMemoryOps(HWModuleOp mod) {
  auto ports = mod.getPortList();
  auto *ctx = mod.getContext();

  // Gather memref ports to be converted.
  llvm::DenseMap<unsigned, Value> memrefPorts;
  for (auto [i, arg] : llvm::enumerate(mod.getArguments())) {
    auto channel = arg.getType().dyn_cast<esi::ChannelType>();
    if (channel && channel.getInner().isa<MemRefType>())
      memrefPorts[i] = arg;
  }

  if (memrefPorts.empty())
    return success(); // nothing to do.

  OpBuilder b(mod);

  auto getMemoryIOInfo = [&](Location loc, Twine portName, unsigned argIdx,
                             ArrayRef<hw::PortInfo> info,
                             hw::ModulePort::Direction direction) {
    auto type = hw::StructType::get(ctx, portToFieldInfo(info));
    auto portInfo =
        hw::PortInfo{{b.getStringAttr(portName), type, direction}, argIdx};
    return portInfo;
  };

  for (auto [i, arg] : memrefPorts) {
    // Insert ports into the module
    auto memName = mod.getArgNames()[i].cast<StringAttr>();

    // Get the attached extmemory external module.
    auto extmemInstance = cast<hw::InstanceOp>(*arg.getUsers().begin());
    auto extmemMod =
        cast<hw::HWModuleExternOp>(extmemInstance.getReferencedModule());
    auto portInfo = extmemMod.getPortList();

    // The extmemory external module's interface is a direct wrapping of the
    // original handshake.extmemory operation in- and output types. Remove the
    // first input argument (the !esi.channel<memref> op) since that is what
    // we're replacing with a materialized interface.
    portInfo.eraseInput(0);

    // Add memory input - this is the output of the extmemory op.
    SmallVector<PortInfo> outputs(portInfo.getOutputs());
    auto inPortInfo =
        getMemoryIOInfo(arg.getLoc(), memName.strref() + "_in", i, outputs,
                        hw::ModulePort::Direction::Input);
    mod.insertPorts({{i, inPortInfo}}, {});
    auto newInPort = mod.getArgument(i);
    // Replace the extmemory submodule outputs with the newly created inputs.
    b.setInsertionPointToStart(mod.getBodyBlock());
    auto newInPortExploded = b.create<hw::StructExplodeOp>(
        arg.getLoc(), extmemMod.getResultTypes(), newInPort);
    extmemInstance.replaceAllUsesWith(newInPortExploded.getResults());

    // Add memory output - this is the inputs of the extmemory op (without the
    // first argument);
    unsigned outArgI = mod.getNumResults();
    SmallVector<PortInfo> inputs(portInfo.getInputs());
    auto outPortInfo =
        getMemoryIOInfo(arg.getLoc(), memName.strref() + "_out", outArgI,
                        inputs, hw::ModulePort::Direction::Output);

    auto memOutputArgs = extmemInstance.getOperands().drop_front();
    b.setInsertionPoint(mod.getBodyBlock()->getTerminator());
    auto memOutputStruct = b.create<hw::StructCreateOp>(
        arg.getLoc(), outPortInfo.type, memOutputArgs);
    mod.appendOutputs({{outPortInfo.name, memOutputStruct}});

    // Erase the extmemory submodule instace since the i/o has now been
    // plumbed.
    extmemMod.erase();
    extmemInstance.erase();

    // Erase the original memref argument of the top-level i/o now that it's use
    // has been removed.
    mod.modifyPorts(/*insertInputs*/ {}, /*insertOutputs*/ {},
                    /*eraseInputs*/ {i + 1}, /*eraseOutputs*/ {});
  }

  return success();
}

namespace {

// Input handshakes contain a resolved valid and (optional )data signal, and
// a to-be-assigned ready signal.
struct InputHandshake {
  Value valid;
  std::shared_ptr<Backedge> ready;
  Value data;
};

// Output handshakes contain a resolved ready, and to-be-assigned valid and
// (optional) data signals.
struct OutputHandshake {
  std::shared_ptr<Backedge> valid;
  Value ready;
  std::shared_ptr<Backedge> data;
};

/// A helper struct that acts like a wire. Can be used to interact with the
/// RTLBuilder when multiple built components should be connected.
struct HandshakeWire {
  HandshakeWire(BackedgeBuilder &bb, Type dataType) {
    MLIRContext *ctx = dataType.getContext();
    auto i1Type = IntegerType::get(ctx, 1);
    valid = std::make_shared<Backedge>(bb.get(i1Type));
    ready = std::make_shared<Backedge>(bb.get(i1Type));
    data = std::make_shared<Backedge>(bb.get(dataType));
  }

  // Functions that allow to treat a wire like an input or output port.
  // **Careful**: Such a port will not be updated when backedges are resolved.
  InputHandshake getAsInput() { return {*valid, ready, *data}; }
  OutputHandshake getAsOutput() { return {valid, *ready, data}; }

  std::shared_ptr<Backedge> valid;
  std::shared_ptr<Backedge> ready;
  std::shared_ptr<Backedge> data;
};

template <typename T, typename TInner>
llvm::SmallVector<T> extractValues(llvm::SmallVector<TInner> &container,
                                   llvm::function_ref<T(TInner &)> extractor) {
  llvm::SmallVector<T> result;
  llvm::transform(container, std::back_inserter(result), extractor);
  return result;
}
struct UnwrappedIO {
  llvm::SmallVector<InputHandshake> inputs;
  llvm::SmallVector<OutputHandshake> outputs;

  llvm::SmallVector<Value> getInputValids() {
    return extractValues<Value, InputHandshake>(
        inputs, [](auto &hs) { return hs.valid; });
  }
  llvm::SmallVector<std::shared_ptr<Backedge>> getInputReadys() {
    return extractValues<std::shared_ptr<Backedge>, InputHandshake>(
        inputs, [](auto &hs) { return hs.ready; });
  }
  llvm::SmallVector<Value> getInputDatas() {
    return extractValues<Value, InputHandshake>(
        inputs, [](auto &hs) { return hs.data; });
  }
  llvm::SmallVector<std::shared_ptr<Backedge>> getOutputValids() {
    return extractValues<std::shared_ptr<Backedge>, OutputHandshake>(
        outputs, [](auto &hs) { return hs.valid; });
  }
  llvm::SmallVector<Value> getOutputReadys() {
    return extractValues<Value, OutputHandshake>(
        outputs, [](auto &hs) { return hs.ready; });
  }
  llvm::SmallVector<std::shared_ptr<Backedge>> getOutputDatas() {
    return extractValues<std::shared_ptr<Backedge>, OutputHandshake>(
        outputs, [](auto &hs) { return hs.data; });
  }
};

// A class containing a bunch of syntactic sugar to reduce builder function
// verbosity.
// @todo: should be moved to support.
struct RTLBuilder {
  RTLBuilder(hw::ModulePortInfo info, OpBuilder &builder, Location loc,
             Value clk = Value(), Value rst = Value())
      : info(std::move(info)), b(builder), loc(loc), clk(clk), rst(rst) {}

  Value constant(const APInt &apv, std::optional<StringRef> name = {}) {
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

  Value constant(unsigned width, int64_t value,
                 std::optional<StringRef> name = {}) {
    return constant(APInt(width, value));
  }
  std::pair<Value, Value> wrap(Value data, Value valid,
                               std::optional<StringRef> name = {}) {
    auto wrapOp = b.create<esi::WrapValidReadyOp>(loc, data, valid);
    return {wrapOp.getResult(0), wrapOp.getResult(1)};
  }
  std::pair<Value, Value> unwrap(Value channel, Value ready,
                                 std::optional<StringRef> name = {}) {
    auto unwrapOp = b.create<esi::UnwrapValidReadyOp>(loc, channel, ready);
    return {unwrapOp.getResult(0), unwrapOp.getResult(1)};
  }

  // Various syntactic sugar functions.
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
            std::optional<StringRef> name = {}) {
    return b.create<comb::ICmpOp>(loc, predicate, lhs, rhs);
  }

  Value buildNamedOp(llvm::function_ref<Value()> f,
                     std::optional<StringRef> name) {
    Value v = f();
    StringAttr nameAttr;
    Operation *op = v.getDefiningOp();
    if (name.has_value()) {
      op->setAttr("sv.namehint", b.getStringAttr(*name));
      nameAttr = b.getStringAttr(*name);
    }
    return v;
  }

  // Bitwise 'and'.
  Value bAnd(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::AndOp>(loc, values, false); }, name);
  }

  Value bOr(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::OrOp>(loc, values, false); }, name);
  }

  // Bitwise 'not'.
  Value bNot(Value value, std::optional<StringRef> name = {}) {
    auto allOnes = constant(value.getType().getIntOrFloatBitWidth(), -1);
    std::string inferedName;
    if (!name) {
      // Try to create a name from the input value.
      if (auto valueName =
              value.getDefiningOp()->getAttrOfType<StringAttr>("sv.namehint")) {
        inferedName = ("not_" + valueName.getValue()).str();
        name = inferedName;
      }
    }

    return buildNamedOp(
        [&]() { return b.create<comb::XorOp>(loc, value, allOnes); }, name);

    return b.createOrFold<comb::XorOp>(loc, value, allOnes, false);
  }

  Value shl(Value value, Value shift, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<comb::ShlOp>(loc, value, shift); }, name);
  }

  Value concat(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp([&]() { return b.create<comb::ConcatOp>(loc, values); },
                        name);
  }

  // Packs a list of values into a hw.struct.
  Value pack(ValueRange values, Type structType = Type(),
             std::optional<StringRef> name = {}) {
    if (!structType)
      structType = tupleToStruct(values.getTypes());
    return buildNamedOp(
        [&]() { return b.create<hw::StructCreateOp>(loc, structType, values); },
        name);
  }

  // Unpacks a hw.struct into a list of values.
  ValueRange unpack(Value value) {
    auto structType = value.getType().cast<hw::StructType>();
    llvm::SmallVector<Type> innerTypes;
    structType.getInnerTypes(innerTypes);
    return b.create<hw::StructExplodeOp>(loc, innerTypes, value).getResults();
  }

  llvm::SmallVector<Value> toBits(Value v, std::optional<StringRef> name = {}) {
    llvm::SmallVector<Value> bits;
    for (unsigned i = 0, e = v.getType().getIntOrFloatBitWidth(); i != e; ++i)
      bits.push_back(b.create<comb::ExtractOp>(loc, v, i, /*bitWidth=*/1));
    return bits;
  }

  // OR-reduction of the bits in 'v'.
  Value rOr(Value v, std::optional<StringRef> name = {}) {
    return buildNamedOp([&]() { return bOr(toBits(v)); }, name);
  }

  // Extract bits v[hi:lo] (inclusive).
  Value extract(Value v, unsigned lo, unsigned hi,
                std::optional<StringRef> name = {}) {
    unsigned width = hi - lo + 1;
    return buildNamedOp(
        [&]() { return b.create<comb::ExtractOp>(loc, v, lo, width); }, name);
  }

  // Truncates 'value' to its lower 'width' bits.
  Value truncate(Value value, unsigned width,
                 std::optional<StringRef> name = {}) {
    return extract(value, 0, width - 1, name);
  }

  Value zext(Value value, unsigned outWidth,
             std::optional<StringRef> name = {}) {
    unsigned inWidth = value.getType().getIntOrFloatBitWidth();
    assert(inWidth <= outWidth && "zext: input width must be <- output width.");
    if (inWidth == outWidth)
      return value;
    auto c0 = constant(outWidth - inWidth, 0);
    return concat({c0, value}, name);
  }

  Value sext(Value value, unsigned outWidth,
             std::optional<StringRef> name = {}) {
    return comb::createOrFoldSExt(loc, value, b.getIntegerType(outWidth), b);
  }

  // Extracts a single bit v[bit].
  Value bit(Value v, unsigned index, std::optional<StringRef> name = {}) {
    return extract(v, index, index, name);
  }

  // Creates a hw.array of the given values.
  Value arrayCreate(ValueRange values, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<hw::ArrayCreateOp>(loc, values); }, name);
  }

  // Extract the 'index'th value from the input array.
  Value arrayGet(Value array, Value index, std::optional<StringRef> name = {}) {
    return buildNamedOp(
        [&]() { return b.create<hw::ArrayGetOp>(loc, array, index); }, name);
  }

  // Muxes a range of values.
  // The select signal is expected to be a decimal value which selects starting
  // from the lowest index of value.
  Value mux(Value index, ValueRange values,
            std::optional<StringRef> name = {}) {
    if (values.size() == 2)
      return b.create<comb::MuxOp>(loc, index, values[1], values[0]);

    return arrayGet(arrayCreate(values), index, name);
  }

  // Muxes a range of values. The select signal is expected to be a 1-hot
  // encoded value.
  Value ohMux(Value index, ValueRange inputs) {
    // Confirm the select input can be a one-hot encoding for the inputs.
    unsigned numInputs = inputs.size();
    assert(numInputs == index.getType().getIntOrFloatBitWidth() &&
           "one-hot select can't mux inputs");

    // Start the mux tree with zero value.
    // Todo: clean up when handshake supports i0.
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

  hw::ModulePortInfo info;
  OpBuilder &b;
  Location loc;
  Value clk, rst;
  DenseMap<APInt, Value> constants;
};

/// Creates a Value that has an assigned zero value. For structs, this
/// corresponds to assigning zero to each element recursively.
static Value createZeroDataConst(RTLBuilder &s, Location loc, Type type) {
  return TypeSwitch<Type, Value>(type)
      .Case<NoneType>([&](NoneType) { return s.constant(0, 0); })
      .Case<IntType, IntegerType>([&](auto type) {
        return s.constant(type.getIntOrFloatBitWidth(), 0);
      })
      .Case<hw::StructType>([&](auto structType) {
        SmallVector<Value> zeroValues;
        for (auto field : structType.getElements())
          zeroValues.push_back(createZeroDataConst(s, loc, field.type));
        return s.b.create<hw::StructCreateOp>(loc, structType, zeroValues);
      })
      .Default([&](Type) -> Value {
        emitError(loc) << "unsupported type for zero value: " << type;
        assert(false);
        return {};
      });
}

static void
addSequentialIOOperandsIfNeeded(Operation *op,
                                llvm::SmallVectorImpl<Value> &operands) {
  if (op->hasTrait<mlir::OpTrait::HasClock>()) {
    // Parent should at this point be a hw.module and have clock and reset
    // ports.
    auto parent = cast<hw::HWModuleOp>(op->getParentOp());
    operands.push_back(parent.getArgument(parent.getNumArguments() - 2));
    operands.push_back(parent.getArgument(parent.getNumArguments() - 1));
  }
}

template <typename T>
class HandshakeConversionPattern : public OpConversionPattern<T> {
public:
  HandshakeConversionPattern(ESITypeConverter &typeConverter,
                             MLIRContext *context, OpBuilder &submoduleBuilder,
                             HandshakeLoweringState &ls)
      : OpConversionPattern<T>::OpConversionPattern(typeConverter, context),
        submoduleBuilder(submoduleBuilder), ls(ls) {}

  using OpAdaptor = typename T::Adaptor;

  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check if a submodule has already been created for the op. If so,
    // instantiate the submodule. Else, run the pattern-defined module
    // builder.
    hw::HWModuleLike implModule = checkSubModuleOp(ls.parentModule, op);
    if (!implModule) {
      auto portInfo = ModulePortInfo(getPortInfoForOp(op));

      submoduleBuilder.setInsertionPoint(op->getParentOp());
      implModule = submoduleBuilder.create<hw::HWModuleOp>(
          op.getLoc(), submoduleBuilder.getStringAttr(getSubModuleName(op)),
          portInfo, [&](OpBuilder &b, hw::HWModulePortAccessor &ports) {
            // if 'op' has clock trait, extract these and provide them to the
            // RTL builder.
            Value clk, rst;
            if (op->template hasTrait<mlir::OpTrait::HasClock>()) {
              clk = ports.getInput("clock");
              rst = ports.getInput("reset");
            }

            BackedgeBuilder bb(b, op.getLoc());
            RTLBuilder s(ports.getPortList(), b, op.getLoc(), clk, rst);
            this->buildModule(op, bb, s, ports);
          });
    }

    // Instantiate the submodule.
    llvm::SmallVector<Value> operands = adaptor.getOperands();
    addSequentialIOOperandsIfNeeded(op, operands);
    rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, implModule, rewriter.getStringAttr(ls.nameUniquer(op)), operands);
    return success();
  }

  virtual void buildModule(T op, BackedgeBuilder &bb, RTLBuilder &builder,
                           hw::HWModulePortAccessor &ports) const = 0;

  // Syntactic sugar functions.
  // Unwraps an ESI-interfaced module into its constituent handshake signals.
  // Backedges are created for the to-be-resolved signals, and output ports
  // are assigned to their wrapped counterparts.
  UnwrappedIO unwrapIO(RTLBuilder &s, BackedgeBuilder &bb,
                       hw::HWModulePortAccessor &ports) const {
    UnwrappedIO unwrapped;
    for (auto port : ports.getInputs()) {
      if (!isa<esi::ChannelType>(port.getType()))
        continue;
      InputHandshake hs;
      auto ready = std::make_shared<Backedge>(bb.get(s.b.getI1Type()));
      auto [data, valid] = s.unwrap(port, *ready);
      hs.data = data;
      hs.valid = valid;
      hs.ready = ready;
      unwrapped.inputs.push_back(hs);
    }
    for (auto &outputInfo : ports.getPortList().getOutputs()) {
      esi::ChannelType channelType =
          dyn_cast<esi::ChannelType>(outputInfo.type);
      if (!channelType)
        continue;
      OutputHandshake hs;
      Type innerType = channelType.getInner();
      auto data = std::make_shared<Backedge>(bb.get(innerType));
      auto valid = std::make_shared<Backedge>(bb.get(s.b.getI1Type()));
      auto [dataCh, ready] = s.wrap(*data, *valid);
      hs.data = data;
      hs.valid = valid;
      hs.ready = ready;
      ports.setOutput(outputInfo.name, dataCh);
      unwrapped.outputs.push_back(hs);
    }
    return unwrapped;
  }

  void setAllReadyWithCond(RTLBuilder &s, ArrayRef<InputHandshake> inputs,
                           OutputHandshake &output, Value cond) const {
    auto validAndReady = s.bAnd({output.ready, cond});
    for (auto &input : inputs)
      input.ready->setValue(validAndReady);
  }

  void buildJoinLogic(RTLBuilder &s, ArrayRef<InputHandshake> inputs,
                      OutputHandshake &output) const {
    llvm::SmallVector<Value> valids;
    for (auto &input : inputs)
      valids.push_back(input.valid);
    Value allValid = s.bAnd(valids);
    output.valid->setValue(allValid);
    setAllReadyWithCond(s, inputs, output, allValid);
  }

  // Builds mux logic for the given inputs and outputs.
  // Note: it is assumed that the caller has removed the 'select' signal from
  // the 'unwrapped' inputs and provide it as a separate argument.
  void buildMuxLogic(RTLBuilder &s, UnwrappedIO &unwrapped,
                     InputHandshake &select) const {
    // ============================= Control logic =============================
    size_t numInputs = unwrapped.inputs.size();
    size_t selectWidth = llvm::Log2_64_Ceil(numInputs);
    Value truncatedSelect =
        select.data.getType().getIntOrFloatBitWidth() > selectWidth
            ? s.truncate(select.data, selectWidth)
            : select.data;

    // Decimal-to-1-hot decoder. 'shl' operands must be identical in size.
    auto selectZext = s.zext(truncatedSelect, numInputs);
    auto select1h = s.shl(s.constant(numInputs, 1), selectZext);
    auto &res = unwrapped.outputs[0];

    // Mux input valid signals.
    auto selectedInputValid =
        s.mux(truncatedSelect, unwrapped.getInputValids());
    // Result is valid when the selected input and the select input is valid.
    auto selAndInputValid = s.bAnd({selectedInputValid, select.valid});
    res.valid->setValue(selAndInputValid);
    auto resValidAndReady = s.bAnd({selAndInputValid, res.ready});

    // Select is ready when result is valid and ready (result transacting).
    select.ready->setValue(resValidAndReady);

    // Assign each input ready signal if it is currently selected.
    for (auto [inIdx, in] : llvm::enumerate(unwrapped.inputs)) {
      // Extract the selection bit for this input.
      auto isSelected = s.bit(select1h, inIdx);

      // '&' that with the result valid and ready, and assign to the input
      // ready signal.
      auto activeAndResultValidAndReady =
          s.bAnd({isSelected, resValidAndReady});
      in.ready->setValue(activeAndResultValidAndReady);
    }

    // ============================== Data logic ===============================
    res.data->setValue(s.mux(truncatedSelect, unwrapped.getInputDatas()));
  }

  // Builds fork logic between the single input and multiple outputs' control
  // networks. Caller is expected to handle data separately.
  void buildForkLogic(RTLBuilder &s, BackedgeBuilder &bb, InputHandshake &input,
                      ArrayRef<OutputHandshake> outputs) const {
    auto c0I1 = s.constant(1, 0);
    llvm::SmallVector<Value> doneWires;
    for (auto [i, output] : llvm::enumerate(outputs)) {
      auto doneBE = bb.get(s.b.getI1Type());
      auto emitted = s.bAnd({doneBE, s.bNot(*input.ready)});
      auto emittedReg = s.reg("emitted_" + std::to_string(i), emitted, c0I1);
      auto outValid = s.bAnd({s.bNot(emittedReg), input.valid});
      output.valid->setValue(outValid);
      auto validReady = s.bAnd({output.ready, outValid});
      auto done = s.bOr({validReady, emittedReg}, "done" + std::to_string(i));
      doneBE.setValue(done);
      doneWires.push_back(done);
    }
    input.ready->setValue(s.bAnd(doneWires, "allDone"));
  }

  // Builds a unit-rate actor around an inner operation. 'unitBuilder' is a
  // function which takes the set of unwrapped data inputs, and returns a
  // value which should be assigned to the output data value.
  void buildUnitRateJoinLogic(
      RTLBuilder &s, UnwrappedIO &unwrappedIO,
      llvm::function_ref<Value(ValueRange)> unitBuilder) const {
    assert(unwrappedIO.outputs.size() == 1 &&
           "Expected exactly one output for unit-rate join actor");
    // Control logic.
    this->buildJoinLogic(s, unwrappedIO.inputs, unwrappedIO.outputs[0]);

    // Data logic.
    auto unitRes = unitBuilder(unwrappedIO.getInputDatas());
    unwrappedIO.outputs[0].data->setValue(unitRes);
  }

  void buildUnitRateForkLogic(
      RTLBuilder &s, BackedgeBuilder &bb, UnwrappedIO &unwrappedIO,
      llvm::function_ref<llvm::SmallVector<Value>(Value)> unitBuilder) const {
    assert(unwrappedIO.inputs.size() == 1 &&
           "Expected exactly one input for unit-rate fork actor");
    // Control logic.
    this->buildForkLogic(s, bb, unwrappedIO.inputs[0], unwrappedIO.outputs);

    // Data logic.
    auto unitResults = unitBuilder(unwrappedIO.inputs[0].data);
    assert(unitResults.size() == unwrappedIO.outputs.size() &&
           "Expected unit builder to return one result per output");
    for (auto [res, outport] : llvm::zip(unitResults, unwrappedIO.outputs))
      outport.data->setValue(res);
  }

  void buildExtendLogic(RTLBuilder &s, UnwrappedIO &unwrappedIO,
                        bool signExtend) const {
    size_t outWidth =
        toValidType(static_cast<Value>(*unwrappedIO.outputs[0].data).getType())
            .getIntOrFloatBitWidth();
    buildUnitRateJoinLogic(s, unwrappedIO, [&](ValueRange inputs) {
      if (signExtend)
        return s.sext(inputs[0], outWidth);
      return s.zext(inputs[0], outWidth);
    });
  }

  void buildTruncateLogic(RTLBuilder &s, UnwrappedIO &unwrappedIO,
                          unsigned targetWidth) const {
    size_t outWidth =
        toValidType(static_cast<Value>(*unwrappedIO.outputs[0].data).getType())
            .getIntOrFloatBitWidth();
    buildUnitRateJoinLogic(s, unwrappedIO, [&](ValueRange inputs) {
      return s.truncate(inputs[0], outWidth);
    });
  }

  /// Return the number of bits needed to index the given number of values.
  static size_t getNumIndexBits(uint64_t numValues) {
    return numValues > 1 ? llvm::Log2_64_Ceil(numValues) : 1;
  }

  Value buildPriorityArbiter(RTLBuilder &s, ArrayRef<Value> inputs,
                             Value defaultValue,
                             DenseMap<size_t, Value> &indexMapping) const {
    auto numInputs = inputs.size();
    auto priorityArb = defaultValue;

    for (size_t i = numInputs; i > 0; --i) {
      size_t inputIndex = i - 1;
      size_t oneHotIndex = size_t{1} << inputIndex;
      auto constIndex = s.constant(numInputs, oneHotIndex);
      indexMapping[inputIndex] = constIndex;
      priorityArb = s.mux(inputs[inputIndex], {priorityArb, constIndex});
    }
    return priorityArb;
  }

private:
  OpBuilder &submoduleBuilder;
  HandshakeLoweringState &ls;
};

class ForkConversionPattern : public HandshakeConversionPattern<ForkOp> {
public:
  using HandshakeConversionPattern<ForkOp>::HandshakeConversionPattern;
  void buildModule(ForkOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrapped = unwrapIO(s, bb, ports);
    buildUnitRateForkLogic(s, bb, unwrapped, [&](Value input) {
      return llvm::SmallVector<Value>(unwrapped.outputs.size(), input);
    });
  }
};

class JoinConversionPattern : public HandshakeConversionPattern<JoinOp> {
public:
  using HandshakeConversionPattern<JoinOp>::HandshakeConversionPattern;
  void buildModule(JoinOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = unwrapIO(s, bb, ports);
    buildJoinLogic(s, unwrappedIO.inputs, unwrappedIO.outputs[0]);
    unwrappedIO.outputs[0].data->setValue(s.constant(0, 0));
  };
};

class SyncConversionPattern : public HandshakeConversionPattern<SyncOp> {
public:
  using HandshakeConversionPattern<SyncOp>::HandshakeConversionPattern;
  void buildModule(SyncOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = unwrapIO(s, bb, ports);

    // A helper wire that will be used to connect the two built logics
    HandshakeWire wire(bb, s.b.getNoneType());

    OutputHandshake output = wire.getAsOutput();
    buildJoinLogic(s, unwrappedIO.inputs, output);

    InputHandshake input = wire.getAsInput();

    // The state-keeping fork logic is required here, as the circuit isn't
    // allowed to wait for all the consumers to be ready. Connecting the ready
    // signals of the outputs to their corresponding valid signals leads to
    // combinatorial cycles. The paper which introduced compositional dataflow
    // circuits explicitly mentions this limitation:
    // http://arcade.cs.columbia.edu/df-memocode17.pdf
    buildForkLogic(s, bb, input, unwrappedIO.outputs);

    // Directly connect the data wires, only the control signals need to be
    // combined.
    for (auto &&[in, out] : llvm::zip(unwrappedIO.inputs, unwrappedIO.outputs))
      out.data->setValue(in.data);
  };
};

class MuxConversionPattern : public HandshakeConversionPattern<MuxOp> {
public:
  using HandshakeConversionPattern<MuxOp>::HandshakeConversionPattern;
  void buildModule(MuxOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = unwrapIO(s, bb, ports);

    // Extract select signal from the unwrapped IO.
    auto select = unwrappedIO.inputs[0];
    unwrappedIO.inputs.erase(unwrappedIO.inputs.begin());
    buildMuxLogic(s, unwrappedIO, select);
  };
};

class InstanceConversionPattern
    : public HandshakeConversionPattern<handshake::InstanceOp> {
public:
  using HandshakeConversionPattern<
      handshake::InstanceOp>::HandshakeConversionPattern;
  void buildModule(handshake::InstanceOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    assert(false &&
           "If we indeed perform conversion in post-order, this "
           "should never be called. The base HandshakeConversionPattern logic "
           "will instantiate the external module.");
  }
};

class ReturnConversionPattern
    : public OpConversionPattern<handshake::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Locate existing output op, Append operands to output op, and move to
    // the end of the block.
    auto parent = cast<hw::HWModuleOp>(op->getParentOp());
    auto outputOp = *parent.getBodyBlock()->getOps<hw::OutputOp>().begin();
    outputOp->setOperands(adaptor.getOperands());
    outputOp->moveAfter(&parent.getBodyBlock()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

// Converts an arbitrary operation into a unit rate actor. A unit rate actor
// will transact once all inputs are valid and its output is ready.
template <typename TIn, typename TOut = TIn>
class UnitRateConversionPattern : public HandshakeConversionPattern<TIn> {
public:
  using HandshakeConversionPattern<TIn>::HandshakeConversionPattern;
  void buildModule(TIn op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    this->buildUnitRateJoinLogic(s, unwrappedIO, [&](ValueRange inputs) {
      // Create TOut - it is assumed that TOut trivially
      // constructs from the input data signals of TIn.
      // To disambiguate ambiguous builders with default arguments (e.g.,
      // twoState UnitAttr), specify attribute array explicitly.
      return s.b.create<TOut>(op.getLoc(), inputs,
                              /* attributes */ ArrayRef<NamedAttribute>{});
    });
  };
};

class PackConversionPattern : public HandshakeConversionPattern<PackOp> {
public:
  using HandshakeConversionPattern<PackOp>::HandshakeConversionPattern;
  void buildModule(PackOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = unwrapIO(s, bb, ports);
    buildUnitRateJoinLogic(s, unwrappedIO,
                           [&](ValueRange inputs) { return s.pack(inputs); });
  };
};

class StructCreateConversionPattern
    : public HandshakeConversionPattern<hw::StructCreateOp> {
public:
  using HandshakeConversionPattern<
      hw::StructCreateOp>::HandshakeConversionPattern;
  void buildModule(hw::StructCreateOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = unwrapIO(s, bb, ports);
    auto structType = op.getResult().getType();
    buildUnitRateJoinLogic(s, unwrappedIO, [&](ValueRange inputs) {
      return s.pack(inputs, structType);
    });
  };
};

class UnpackConversionPattern : public HandshakeConversionPattern<UnpackOp> {
public:
  using HandshakeConversionPattern<UnpackOp>::HandshakeConversionPattern;
  void buildModule(UnpackOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = unwrapIO(s, bb, ports);
    buildUnitRateForkLogic(s, bb, unwrappedIO,
                           [&](Value input) { return s.unpack(input); });
  };
};

class ConditionalBranchConversionPattern
    : public HandshakeConversionPattern<ConditionalBranchOp> {
public:
  using HandshakeConversionPattern<
      ConditionalBranchOp>::HandshakeConversionPattern;
  void buildModule(ConditionalBranchOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = unwrapIO(s, bb, ports);
    auto cond = unwrappedIO.inputs[0];
    auto arg = unwrappedIO.inputs[1];
    auto trueRes = unwrappedIO.outputs[0];
    auto falseRes = unwrappedIO.outputs[1];

    auto condArgValid = s.bAnd({cond.valid, arg.valid});

    // Connect valid signal of both results.
    trueRes.valid->setValue(s.bAnd({cond.data, condArgValid}));
    falseRes.valid->setValue(s.bAnd({s.bNot(cond.data), condArgValid}));

    // Connecte data signals of both results.
    trueRes.data->setValue(arg.data);
    falseRes.data->setValue(arg.data);

    // Connect ready signal of input and condition.
    auto selectedResultReady =
        s.mux(cond.data, {falseRes.ready, trueRes.ready});
    auto condArgReady = s.bAnd({selectedResultReady, condArgValid});
    arg.ready->setValue(condArgReady);
    cond.ready->setValue(condArgReady);
  };
};

template <typename TIn, bool signExtend>
class ExtendConversionPattern : public HandshakeConversionPattern<TIn> {
public:
  using HandshakeConversionPattern<TIn>::HandshakeConversionPattern;
  void buildModule(TIn op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    this->buildExtendLogic(s, unwrappedIO, /*signExtend=*/signExtend);
  };
};

class ComparisonConversionPattern
    : public HandshakeConversionPattern<arith::CmpIOp> {
public:
  using HandshakeConversionPattern<arith::CmpIOp>::HandshakeConversionPattern;
  void buildModule(arith::CmpIOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    auto buildCompareLogic = [&](comb::ICmpPredicate predicate) {
      return buildUnitRateJoinLogic(s, unwrappedIO, [&](ValueRange inputs) {
        return s.b.create<comb::ICmpOp>(op.getLoc(), predicate, inputs[0],
                                        inputs[1]);
      });
    };

    switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return buildCompareLogic(comb::ICmpPredicate::eq);
    case arith::CmpIPredicate::ne:
      return buildCompareLogic(comb::ICmpPredicate::ne);
    case arith::CmpIPredicate::slt:
      return buildCompareLogic(comb::ICmpPredicate::slt);
    case arith::CmpIPredicate::ult:
      return buildCompareLogic(comb::ICmpPredicate::ult);
    case arith::CmpIPredicate::sle:
      return buildCompareLogic(comb::ICmpPredicate::sle);
    case arith::CmpIPredicate::ule:
      return buildCompareLogic(comb::ICmpPredicate::ule);
    case arith::CmpIPredicate::sgt:
      return buildCompareLogic(comb::ICmpPredicate::sgt);
    case arith::CmpIPredicate::ugt:
      return buildCompareLogic(comb::ICmpPredicate::ugt);
    case arith::CmpIPredicate::sge:
      return buildCompareLogic(comb::ICmpPredicate::sge);
    case arith::CmpIPredicate::uge:
      return buildCompareLogic(comb::ICmpPredicate::uge);
    }
    assert(false && "invalid CmpIOp");
  };
};

class TruncateConversionPattern
    : public HandshakeConversionPattern<arith::TruncIOp> {
public:
  using HandshakeConversionPattern<arith::TruncIOp>::HandshakeConversionPattern;
  void buildModule(arith::TruncIOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    unsigned targetBits =
        toValidType(op.getResult().getType()).getIntOrFloatBitWidth();
    buildTruncateLogic(s, unwrappedIO, targetBits);
  };
};

class ControlMergeConversionPattern
    : public HandshakeConversionPattern<ControlMergeOp> {
public:
  using HandshakeConversionPattern<ControlMergeOp>::HandshakeConversionPattern;
  void buildModule(ControlMergeOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    auto resData = unwrappedIO.outputs[0];
    auto resIndex = unwrappedIO.outputs[1];

    // Define some common types and values that will be used.
    unsigned numInputs = unwrappedIO.inputs.size();
    auto indexType = s.b.getIntegerType(numInputs);
    Value noWinner = s.constant(numInputs, 0);
    Value c0I1 = s.constant(1, 0);

    // Declare register for storing arbitration winner.
    auto won = bb.get(indexType);
    Value wonReg = s.reg("won_reg", won, noWinner);

    // Declare wire for arbitration winner.
    auto win = bb.get(indexType);

    // Declare wire for whether the circuit just fired and emitted both
    // outputs.
    auto fired = bb.get(s.b.getI1Type());

    // Declare registers for storing if each output has been emitted.
    auto resultEmitted = bb.get(s.b.getI1Type());
    Value resultEmittedReg = s.reg("result_emitted_reg", resultEmitted, c0I1);
    auto indexEmitted = bb.get(s.b.getI1Type());
    Value indexEmittedReg = s.reg("index_emitted_reg", indexEmitted, c0I1);

    // Declare wires for if each output is done.
    auto resultDone = bb.get(s.b.getI1Type());
    auto indexDone = bb.get(s.b.getI1Type());

    // Create predicates to assert if the win wire or won register hold a
    // valid index.
    auto hasWinnerCondition = s.rOr({win});
    auto hadWinnerCondition = s.rOr({wonReg});

    // Create an arbiter based on a simple priority-encoding scheme to assign
    // an index to the win wire. If the won register is set, just use that. In
    // the case that won is not set and no input is valid, set a sentinel
    // value to indicate no winner was chosen. The constant values are
    // remembered in a map so they can be re-used later to assign the arg
    // ready outputs.
    DenseMap<size_t, Value> argIndexValues;
    Value priorityArb = buildPriorityArbiter(s, unwrappedIO.getInputValids(),
                                             noWinner, argIndexValues);
    priorityArb = s.mux(hadWinnerCondition, {priorityArb, wonReg});
    win.setValue(priorityArb);

    // Create the logic to assign the result and index outputs. The result
    // valid output will always be assigned, and if isControl is not set, the
    // result data output will also be assigned. The index valid and data
    // outputs will always be assigned. The win wire from the arbiter is used
    // to index into a tree of muxes to select the chosen input's signal(s),
    // and is fed directly to the index output. Both the result and index
    // valid outputs are gated on the win wire being set to something other
    // than the sentinel value.
    auto resultNotEmitted = s.bNot(resultEmittedReg);
    auto resultValid = s.bAnd({hasWinnerCondition, resultNotEmitted});
    resData.valid->setValue(resultValid);
    resData.data->setValue(s.ohMux(win, unwrappedIO.getInputDatas()));

    auto indexNotEmitted = s.bNot(indexEmittedReg);
    auto indexValid = s.bAnd({hasWinnerCondition, indexNotEmitted});
    resIndex.valid->setValue(indexValid);

    // Use the one-hot win wire to select the index to output in the index
    // data.
    SmallVector<Value, 8> indexOutputs;
    for (size_t i = 0; i < numInputs; ++i)
      indexOutputs.push_back(s.constant(64, i));

    auto indexOutput = s.ohMux(win, indexOutputs);
    resIndex.data->setValue(indexOutput);

    // Create the logic to set the won register. If the fired wire is
    // asserted, we have finished this round and can and reset the register to
    // the sentinel value that indicates there is no winner. Otherwise, we
    // need to hold the value of the win register until we can fire.
    won.setValue(s.mux(fired, {win, noWinner}));

    // Create the logic to set the done wires for the result and index. For
    // both outputs, the done wire is asserted when the output is valid and
    // ready, or the emitted register for that output is set.
    auto resultValidAndReady = s.bAnd({resultValid, resData.ready});
    resultDone.setValue(s.bOr({resultValidAndReady, resultEmittedReg}));

    auto indexValidAndReady = s.bAnd({indexValid, resIndex.ready});
    indexDone.setValue(s.bOr({indexValidAndReady, indexEmittedReg}));

    // Create the logic to set the fired wire. It is asserted when both result
    // and index are done.
    fired.setValue(s.bAnd({resultDone, indexDone}));

    // Create the logic to assign the emitted registers. If the fired wire is
    // asserted, we have finished this round and can reset the registers to 0.
    // Otherwise, we need to hold the values of the done registers until we
    // can fire.
    resultEmitted.setValue(s.mux(fired, {resultDone, c0I1}));
    indexEmitted.setValue(s.mux(fired, {indexDone, c0I1}));

    // Create the logic to assign the arg ready outputs. The logic is
    // identical for each arg. If the fired wire is asserted, and the win wire
    // holds an arg's index, that arg is ready.
    auto winnerOrDefault = s.mux(fired, {noWinner, win});
    for (auto [i, ir] : llvm::enumerate(unwrappedIO.getInputReadys())) {
      auto &indexValue = argIndexValues[i];
      ir->setValue(s.cmp(winnerOrDefault, indexValue, comb::ICmpPredicate::eq));
    }
  };
};

class MergeConversionPattern : public HandshakeConversionPattern<MergeOp> {
public:
  using HandshakeConversionPattern<MergeOp>::HandshakeConversionPattern;
  void buildModule(MergeOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    auto resData = unwrappedIO.outputs[0];

    // Define some common types and values that will be used.
    unsigned numInputs = unwrappedIO.inputs.size();
    auto indexType = s.b.getIntegerType(numInputs);
    Value noWinner = s.constant(numInputs, 0);

    // Declare wire for arbitration winner.
    auto win = bb.get(indexType);

    // Create predicates to assert if the win wire holds a valid index.
    auto hasWinnerCondition = s.rOr(win);

    // Create an arbiter based on a simple priority-encoding scheme to assign an
    // index to the win wire. In the case that no input is valid, set a sentinel
    // value to indicate no winner was chosen. The constant values are
    // remembered in a map so they can be re-used later to assign the arg ready
    // outputs.
    DenseMap<size_t, Value> argIndexValues;
    Value priorityArb = buildPriorityArbiter(s, unwrappedIO.getInputValids(),
                                             noWinner, argIndexValues);
    win.setValue(priorityArb);

    // Create the logic to assign the result outputs. The result valid and data
    // outputs will always be assigned. The win wire from the arbiter is used to
    // index into a tree of muxes to select the chosen input's signal(s). The
    // result outputs are gated on the win wire being non-zero.

    resData.valid->setValue(hasWinnerCondition);
    resData.data->setValue(s.ohMux(win, unwrappedIO.getInputDatas()));

    // Create the logic to set the done wires for the result. The done wire is
    // asserted when the output is valid and ready, or the emitted register is
    // set.
    auto resultValidAndReady = s.bAnd({hasWinnerCondition, resData.ready});

    // Create the logic to assign the arg ready outputs. The logic is
    // identical for each arg. If the fired wire is asserted, and the win wire
    // holds an arg's index, that arg is ready.
    auto winnerOrDefault = s.mux(resultValidAndReady, {noWinner, win});
    for (auto [i, ir] : llvm::enumerate(unwrappedIO.getInputReadys())) {
      auto &indexValue = argIndexValues[i];
      ir->setValue(s.cmp(winnerOrDefault, indexValue, comb::ICmpPredicate::eq));
    }
  };
};

class LoadConversionPattern
    : public HandshakeConversionPattern<handshake::LoadOp> {
public:
  using HandshakeConversionPattern<
      handshake::LoadOp>::HandshakeConversionPattern;
  void buildModule(handshake::LoadOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    auto addrFromUser = unwrappedIO.inputs[0];
    auto dataFromMem = unwrappedIO.inputs[1];
    auto controlIn = unwrappedIO.inputs[2];
    auto dataToUser = unwrappedIO.outputs[0];
    auto addrToMem = unwrappedIO.outputs[1];

    addrToMem.data->setValue(addrFromUser.data);
    dataToUser.data->setValue(dataFromMem.data);

    // The valid/ready logic between user address/control to memoryAddr is
    // join logic.
    buildJoinLogic(s, {addrFromUser, controlIn}, addrToMem);

    // The valid/ready logic between memoryData and outputData is a direct
    // connection.
    dataToUser.valid->setValue(dataFromMem.valid);
    dataFromMem.ready->setValue(dataToUser.ready);
  };
};

class StoreConversionPattern
    : public HandshakeConversionPattern<handshake::StoreOp> {
public:
  using HandshakeConversionPattern<
      handshake::StoreOp>::HandshakeConversionPattern;
  void buildModule(handshake::StoreOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    auto addrFromUser = unwrappedIO.inputs[0];
    auto dataFromUser = unwrappedIO.inputs[1];
    auto controlIn = unwrappedIO.inputs[2];
    auto dataToMem = unwrappedIO.outputs[0];
    auto addrToMem = unwrappedIO.outputs[1];

    // Create a gate that will be asserted when all outputs are ready.
    auto outputsReady = s.bAnd({dataToMem.ready, addrToMem.ready});

    // Build the standard join logic from the inputs to the inputsValid and
    // outputsReady signals.
    HandshakeWire joinWire(bb, s.b.getNoneType());
    joinWire.ready->setValue(outputsReady);
    OutputHandshake joinOutput = joinWire.getAsOutput();
    buildJoinLogic(s, {dataFromUser, addrFromUser, controlIn}, joinOutput);

    // Output address and data signals are connected directly.
    addrToMem.data->setValue(addrFromUser.data);
    dataToMem.data->setValue(dataFromUser.data);

    // Output valid signals are connected from the inputsValid wire.
    addrToMem.valid->setValue(*joinWire.valid);
    dataToMem.valid->setValue(*joinWire.valid);
  };
};

class MemoryConversionPattern
    : public HandshakeConversionPattern<handshake::MemoryOp> {
public:
  using HandshakeConversionPattern<
      handshake::MemoryOp>::HandshakeConversionPattern;
  void buildModule(handshake::MemoryOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto loc = op.getLoc();

    // Gather up the load and store ports.
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    struct LoadPort {
      InputHandshake &addr;
      OutputHandshake &data;
      OutputHandshake &done;
    };
    struct StorePort {
      InputHandshake &addr;
      InputHandshake &data;
      OutputHandshake &done;
    };
    SmallVector<LoadPort, 4> loadPorts;
    SmallVector<StorePort, 4> storePorts;

    unsigned stCount = op.getStCount();
    unsigned ldCount = op.getLdCount();
    for (unsigned i = 0, e = ldCount; i != e; ++i) {
      LoadPort port = {unwrappedIO.inputs[stCount * 2 + i],
                       unwrappedIO.outputs[i],
                       unwrappedIO.outputs[ldCount + stCount + i]};
      loadPorts.push_back(port);
    }

    for (unsigned i = 0, e = stCount; i != e; ++i) {
      StorePort port = {unwrappedIO.inputs[i * 2 + 1],
                        unwrappedIO.inputs[i * 2],
                        unwrappedIO.outputs[ldCount + i]};
      storePorts.push_back(port);
    }

    // used to drive the data wire of the control-only channels.
    auto c0I0 = s.constant(0, 0);

    auto cl2dim = llvm::Log2_64_Ceil(op.getMemRefType().getShape()[0]);
    auto hlmem = s.b.create<seq::HLMemOp>(
        loc, s.clk, s.rst, "_handshake_memory_" + std::to_string(op.getId()),
        op.getMemRefType().getShape(), op.getMemRefType().getElementType());

    // Create load ports...
    for (auto &ld : loadPorts) {
      llvm::SmallVector<Value> addresses = {s.truncate(ld.addr.data, cl2dim)};
      auto readData = s.b.create<seq::ReadPortOp>(loc, hlmem.getHandle(),
                                                  addresses, ld.addr.valid,
                                                  /*latency=*/0);
      ld.data.data->setValue(readData);
      ld.done.data->setValue(c0I0);
      // Create control fork for the load address valid and ready signals.
      buildForkLogic(s, bb, ld.addr, {ld.data, ld.done});
    }

    // Create store ports...
    for (auto &st : storePorts) {
      // Create a register to buffer the valid path by 1 cycle, to match the
      // write latency of 1.
      auto writeValidBufferMuxBE = bb.get(s.b.getI1Type());
      auto writeValidBuffer =
          s.reg("writeValidBuffer", writeValidBufferMuxBE, s.constant(1, 0));
      st.done.valid->setValue(writeValidBuffer);
      st.done.data->setValue(c0I0);

      // Create the logic for when both the buffered write valid signal and the
      // store complete ready signal are asserted.
      auto storeCompleted =
          s.bAnd({st.done.ready, writeValidBuffer}, "storeCompleted");

      // Create a signal for when the write valid buffer is empty or the output
      // is ready.
      auto notWriteValidBuffer = s.bNot(writeValidBuffer);
      auto emptyOrComplete =
          s.bOr({notWriteValidBuffer, storeCompleted}, "emptyOrComplete");

      // Connect the gate to both the store address ready and store data ready
      st.addr.ready->setValue(emptyOrComplete);
      st.data.ready->setValue(emptyOrComplete);

      // Create a wire for when both the store address and data are valid.
      auto writeValid = s.bAnd({st.addr.valid, st.data.valid}, "writeValid");

      // Create a mux that drives the buffer input. If the emptyOrComplete
      // signal is asserted, the mux selects the writeValid signal. Otherwise,
      // it selects the buffer output, keeping the output registered until the
      // emptyOrComplete signal is asserted.
      writeValidBufferMuxBE.setValue(
          s.mux(emptyOrComplete, {writeValidBuffer, writeValid}));

      // Instantiate the write port operation - truncate address width to memory
      // width.
      llvm::SmallVector<Value> addresses = {s.truncate(st.addr.data, cl2dim)};
      s.b.create<seq::WritePortOp>(loc, hlmem.getHandle(), addresses,
                                   st.data.data, writeValid,
                                   /*latency=*/1);
    }
  }
}; // namespace

class SinkConversionPattern : public HandshakeConversionPattern<SinkOp> {
public:
  using HandshakeConversionPattern<SinkOp>::HandshakeConversionPattern;
  void buildModule(SinkOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    // A sink is always ready to accept a new value.
    unwrappedIO.inputs[0].ready->setValue(s.constant(1, 1));
  };
};

class SourceConversionPattern : public HandshakeConversionPattern<SourceOp> {
public:
  using HandshakeConversionPattern<SourceOp>::HandshakeConversionPattern;
  void buildModule(SourceOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    // A source always provides a new (i0-typed) value.
    unwrappedIO.outputs[0].valid->setValue(s.constant(1, 1));
    unwrappedIO.outputs[0].data->setValue(s.constant(0, 0));
  };
};

class ConstantConversionPattern
    : public HandshakeConversionPattern<handshake::ConstantOp> {
public:
  using HandshakeConversionPattern<
      handshake::ConstantOp>::HandshakeConversionPattern;
  void buildModule(handshake::ConstantOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    unwrappedIO.outputs[0].valid->setValue(unwrappedIO.inputs[0].valid);
    unwrappedIO.inputs[0].ready->setValue(unwrappedIO.outputs[0].ready);
    auto constantValue = op->getAttrOfType<IntegerAttr>("value").getValue();
    unwrappedIO.outputs[0].data->setValue(s.constant(constantValue));
  };
};

class BufferConversionPattern : public HandshakeConversionPattern<BufferOp> {
public:
  using HandshakeConversionPattern<BufferOp>::HandshakeConversionPattern;
  void buildModule(BufferOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    auto input = unwrappedIO.inputs[0];
    auto output = unwrappedIO.outputs[0];
    InputHandshake lastStage;
    SmallVector<int64_t> initValues;

    // For now, always build seq buffers.
    if (op.getInitValues())
      initValues = op.getInitValueArray();

    lastStage =
        buildSeqBufferLogic(s, bb, toValidType(op.getDataType()),
                            op.getNumSlots(), input, output, initValues);

    // Connect the last stage to the output handshake.
    output.data->setValue(lastStage.data);
    output.valid->setValue(lastStage.valid);
    lastStage.ready->setValue(output.ready);
  };

  struct SeqBufferStage {
    SeqBufferStage(Type dataType, InputHandshake &preStage, BackedgeBuilder &bb,
                   RTLBuilder &s, size_t index,
                   std::optional<int64_t> initValue)
        : dataType(dataType), preStage(preStage), s(s), bb(bb), index(index) {

      // Todo: Change when i0 support is added.
      c0s = createZeroDataConst(s, s.loc, dataType);
      currentStage.ready = std::make_shared<Backedge>(bb.get(s.b.getI1Type()));

      auto hasInitValue = s.constant(1, initValue.has_value());
      auto validBE = bb.get(s.b.getI1Type());
      auto validReg = s.reg(getRegName("valid"), validBE, hasInitValue);
      auto readyBE = bb.get(s.b.getI1Type());

      Value initValueCs = c0s;
      if (initValue.has_value())
        initValueCs = s.constant(dataType.getIntOrFloatBitWidth(), *initValue);

      // This could/should be revised but needs a larger rethinking to avoid
      // introducing new bugs.
      Value dataReg =
          buildDataBufferLogic(validReg, initValueCs, validBE, readyBE);
      buildControlBufferLogic(validReg, readyBE, dataReg);
    }

    StringAttr getRegName(StringRef name) {
      return s.b.getStringAttr(name + std::to_string(index) + "_reg");
    }

    void buildControlBufferLogic(Value validReg, Backedge &readyBE,
                                 Value dataReg) {
      auto c0I1 = s.constant(1, 0);
      auto readyRegWire = bb.get(s.b.getI1Type());
      auto readyReg = s.reg(getRegName("ready"), readyRegWire, c0I1);

      // Create the logic to drive the current stage valid and potentially
      // data.
      currentStage.valid = s.mux(readyReg, {validReg, readyReg},
                                 "controlValid" + std::to_string(index));

      // Create the logic to drive the current stage ready.
      auto notReadyReg = s.bNot(readyReg);
      readyBE.setValue(notReadyReg);

      auto succNotReady = s.bNot(*currentStage.ready);
      auto neitherReady = s.bAnd({succNotReady, notReadyReg});
      auto ctrlNotReady = s.mux(neitherReady, {readyReg, validReg});
      auto bothReady = s.bAnd({*currentStage.ready, readyReg});

      // Create a mux for emptying the register when both are ready.
      auto resetSignal = s.mux(bothReady, {ctrlNotReady, c0I1});
      readyRegWire.setValue(resetSignal);

      // Add same logic for the data path if necessary.
      auto ctrlDataRegBE = bb.get(dataType);
      auto ctrlDataReg = s.reg(getRegName("ctrl_data"), ctrlDataRegBE, c0s);
      auto dataResult = s.mux(readyReg, {dataReg, ctrlDataReg});
      currentStage.data = dataResult;

      auto dataNotReadyMux = s.mux(neitherReady, {ctrlDataReg, dataReg});
      auto dataResetSignal = s.mux(bothReady, {dataNotReadyMux, c0s});
      ctrlDataRegBE.setValue(dataResetSignal);
    }

    Value buildDataBufferLogic(Value validReg, Value initValue,
                               Backedge &validBE, Backedge &readyBE) {
      // Create a signal for when the valid register is empty or the successor
      // is ready to accept new token.
      auto notValidReg = s.bNot(validReg);
      auto emptyOrReady = s.bOr({notValidReg, readyBE});
      preStage.ready->setValue(emptyOrReady);

      // Create a mux that drives the register input. If the emptyOrReady
      // signal is asserted, the mux selects the predValid signal. Otherwise,
      // it selects the register output, keeping the output registered
      // unchanged.
      auto validRegMux = s.mux(emptyOrReady, {validReg, preStage.valid});

      // Now we can drive the valid register.
      validBE.setValue(validRegMux);

      // Create a mux that drives the date register.
      auto dataRegBE = bb.get(dataType);
      auto dataReg =
          s.reg(getRegName("data"),
                s.mux(emptyOrReady, {dataRegBE, preStage.data}), initValue);
      dataRegBE.setValue(dataReg);
      return dataReg;
    }

    InputHandshake getOutput() { return currentStage; }

    Type dataType;
    InputHandshake &preStage;
    InputHandshake currentStage;
    RTLBuilder &s;
    BackedgeBuilder &bb;
    size_t index;

    // A zero-valued constant of equal type as the data type of this buffer.
    Value c0s;
  };

  InputHandshake buildSeqBufferLogic(RTLBuilder &s, BackedgeBuilder &bb,
                                     Type dataType, unsigned size,
                                     InputHandshake &input,
                                     OutputHandshake &output,
                                     llvm::ArrayRef<int64_t> initValues) const {
    // Prime the buffer building logic with an initial stage, which just
    // wraps the input handshake.
    InputHandshake currentStage = input;

    for (unsigned i = 0; i < size; ++i) {
      bool isInitialized = i < initValues.size();
      auto initValue =
          isInitialized ? std::optional<int64_t>(initValues[i]) : std::nullopt;
      currentStage = SeqBufferStage(dataType, currentStage, bb, s, i, initValue)
                         .getOutput();
    }

    return currentStage;
  };
};

class IndexCastConversionPattern
    : public HandshakeConversionPattern<arith::IndexCastOp> {
public:
  using HandshakeConversionPattern<
      arith::IndexCastOp>::HandshakeConversionPattern;
  void buildModule(arith::IndexCastOp op, BackedgeBuilder &bb, RTLBuilder &s,
                   hw::HWModulePortAccessor &ports) const override {
    auto unwrappedIO = this->unwrapIO(s, bb, ports);
    unsigned sourceBits =
        toValidType(op.getIn().getType()).getIntOrFloatBitWidth();
    unsigned targetBits =
        toValidType(op.getResult().getType()).getIntOrFloatBitWidth();
    if (targetBits < sourceBits)
      buildTruncateLogic(s, unwrappedIO, targetBits);
    else
      buildExtendLogic(s, unwrappedIO, /*signExtend=*/true);
  };
};

template <typename T>
class ExtModuleConversionPattern : public OpConversionPattern<T> {
public:
  ExtModuleConversionPattern(ESITypeConverter &typeConverter,
                             MLIRContext *context, OpBuilder &submoduleBuilder,
                             HandshakeLoweringState &ls)
      : OpConversionPattern<T>::OpConversionPattern(typeConverter, context),
        submoduleBuilder(submoduleBuilder), ls(ls) {}
  using OpAdaptor = typename T::Adaptor;

  LogicalResult
  matchAndRewrite(T op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    hw::HWModuleLike implModule = checkSubModuleOp(ls.parentModule, op);
    if (!implModule) {
      auto portInfo = ModulePortInfo(getPortInfoForOp(op));
      implModule = submoduleBuilder.create<hw::HWModuleExternOp>(
          op.getLoc(), submoduleBuilder.getStringAttr(getSubModuleName(op)),
          portInfo);
    }

    llvm::SmallVector<Value> operands = adaptor.getOperands();
    addSequentialIOOperandsIfNeeded(op, operands);
    rewriter.replaceOpWithNewOp<hw::InstanceOp>(
        op, implModule, rewriter.getStringAttr(ls.nameUniquer(op)), operands);
    return success();
  }

private:
  OpBuilder &submoduleBuilder;
  HandshakeLoweringState &ls;
};

class FuncOpConversionPattern : public OpConversionPattern<handshake::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(handshake::FuncOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    ModulePortInfo ports =
        getPortInfoForOpTypes(op, op.getArgumentTypes(), op.getResultTypes());

    HWModuleLike hwModule;
    if (op.isExternal()) {
      hwModule = rewriter.create<hw::HWModuleExternOp>(
          op.getLoc(), rewriter.getStringAttr(op.getName()), ports);
    } else {
      auto hwModuleOp = rewriter.create<hw::HWModuleOp>(
          op.getLoc(), rewriter.getStringAttr(op.getName()), ports);
      auto args = hwModuleOp.getArguments().drop_back(2);
      rewriter.inlineBlockBefore(&op.getBody().front(),
                                 hwModuleOp.getBodyBlock()->getTerminator(),
                                 args);
      hwModule = hwModuleOp;
    }

    // Was any predeclaration associated with this func? If so, replace uses
    // with the newly created module and erase the predeclaration.
    if (auto predecl =
            op->getAttrOfType<FlatSymbolRefAttr>(kPredeclarationAttr)) {
      auto *parentOp = op->getParentOp();
      auto *predeclModule =
          SymbolTable::lookupSymbolIn(parentOp, predecl.getValue());
      if (predeclModule) {
        if (failed(SymbolTable::replaceAllSymbolUses(
                predeclModule, hwModule.getModuleNameAttr(), parentOp)))
          return failure();
        rewriter.eraseOp(predeclModule);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// HW Top-module Related Functions
//===----------------------------------------------------------------------===//

static LogicalResult convertFuncOp(ESITypeConverter &typeConverter,
                                   ConversionTarget &target,
                                   handshake::FuncOp op,
                                   OpBuilder &moduleBuilder) {

  std::map<std::string, unsigned> instanceNameCntr;
  NameUniquer instanceUniquer = [&](Operation *op) {
    std::string instName = getCallName(op);
    if (auto idAttr = op->getAttrOfType<IntegerAttr>("handshake_id"); idAttr) {
      // We use a special naming convention for operations which have a
      // 'handshake_id' attribute.
      instName += "_id" + std::to_string(idAttr.getValue().getZExtValue());
    } else {
      // Fallback to just prefixing with an integer.
      instName += std::to_string(instanceNameCntr[instName]++);
    }
    return instName;
  };

  auto ls = HandshakeLoweringState{op->getParentOfType<mlir::ModuleOp>(),
                                   instanceUniquer};
  RewritePatternSet patterns(op.getContext());
  patterns.insert<FuncOpConversionPattern, ReturnConversionPattern>(
      op.getContext());
  patterns.insert<JoinConversionPattern, ForkConversionPattern,
                  SyncConversionPattern>(typeConverter, op.getContext(),
                                         moduleBuilder, ls);

  patterns.insert<
      // Comb operations.
      UnitRateConversionPattern<arith::AddIOp, comb::AddOp>,
      UnitRateConversionPattern<arith::SubIOp, comb::SubOp>,
      UnitRateConversionPattern<arith::MulIOp, comb::MulOp>,
      UnitRateConversionPattern<arith::DivUIOp, comb::DivSOp>,
      UnitRateConversionPattern<arith::DivSIOp, comb::DivUOp>,
      UnitRateConversionPattern<arith::RemUIOp, comb::ModUOp>,
      UnitRateConversionPattern<arith::RemSIOp, comb::ModSOp>,
      UnitRateConversionPattern<arith::AndIOp, comb::AndOp>,
      UnitRateConversionPattern<arith::OrIOp, comb::OrOp>,
      UnitRateConversionPattern<arith::XOrIOp, comb::XorOp>,
      UnitRateConversionPattern<arith::ShLIOp, comb::OrOp>,
      UnitRateConversionPattern<arith::ShRUIOp, comb::ShrUOp>,
      UnitRateConversionPattern<arith::ShRSIOp, comb::ShrSOp>,
      UnitRateConversionPattern<arith::SelectOp, comb::MuxOp>,
      // HW operations.
      StructCreateConversionPattern,
      // Handshake operations.
      ConditionalBranchConversionPattern, MuxConversionPattern,
      PackConversionPattern, UnpackConversionPattern,
      ComparisonConversionPattern, BufferConversionPattern,
      SourceConversionPattern, SinkConversionPattern, ConstantConversionPattern,
      MergeConversionPattern, ControlMergeConversionPattern,
      LoadConversionPattern, StoreConversionPattern, MemoryConversionPattern,
      InstanceConversionPattern,
      // Arith operations.
      ExtendConversionPattern<arith::ExtUIOp, /*signExtend=*/false>,
      ExtendConversionPattern<arith::ExtSIOp, /*signExtend=*/true>,
      TruncateConversionPattern, IndexCastConversionPattern>(
      typeConverter, op.getContext(), moduleBuilder, ls);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return op->emitOpError() << "error during conversion";
  return success();
}

namespace {
class HandshakeToHWPass : public HandshakeToHWBase<HandshakeToHWPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    // Lowering to HW requires that every value is used exactly once. Check
    // whether this precondition is met, and if not, exit.
    for (auto f : mod.getOps<handshake::FuncOp>()) {
      if (failed(verifyAllValuesHasOneUse(f))) {
        f.emitOpError() << "HandshakeToHW: failed to verify that all values "
                           "are used exactly once. Remember to run the "
                           "fork/sink materialization pass before HW lowering.";
        signalPassFailure();
        return;
      }
    }

    // Resolve the instance graph to get a top-level module.
    std::string topLevel;
    handshake::InstanceGraph uses;
    SmallVector<std::string> sortedFuncs;
    if (resolveInstanceGraph(mod, uses, topLevel, sortedFuncs).failed()) {
      signalPassFailure();
      return;
    }

    ESITypeConverter typeConverter;
    ConversionTarget target(getContext());
    // All top-level logic of a handshake module will be the interconnectivity
    // between instantiated modules.
    target.addLegalOp<hw::HWModuleOp, hw::HWModuleExternOp, hw::OutputOp,
                      hw::InstanceOp>();
    target
        .addIllegalDialect<handshake::HandshakeDialect, arith::ArithDialect>();

    // Convert the handshake.func operations in post-order wrt. the instance
    // graph. This ensures that any referenced submodules (through
    // handshake.instance) has already been lowered, and their HW module
    // equivalents are available.
    OpBuilder submoduleBuilder(mod.getContext());
    submoduleBuilder.setInsertionPointToStart(mod.getBody());
    for (auto &funcName : llvm::reverse(sortedFuncs)) {
      auto funcOp = mod.lookupSymbol<handshake::FuncOp>(funcName);
      assert(funcOp && "handshake.func not found in module!");
      if (failed(
              convertFuncOp(typeConverter, target, funcOp, submoduleBuilder))) {
        signalPassFailure();
        return;
      }
    }

    // Second stage: Convert any handshake.extmemory operations and the
    // top-level I/O associated with these.
    for (auto hwModule : mod.getOps<hw::HWModuleOp>())
      if (failed(convertExtMemoryOps(hwModule)))
        return signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::createHandshakeToHWPass() {
  return std::make_unique<HandshakeToHWPass>();
}
