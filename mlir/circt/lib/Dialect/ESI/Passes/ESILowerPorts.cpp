//===- ESILowerPorts.cpp - Lower ESI ports pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SymCache.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;

// Returns either the string dialect attr stored in 'op' going by the name
// 'attrName' or 'def' if the attribute doesn't exist in 'op'.
inline static StringRef getStringAttributeOr(Operation *op, StringRef attrName,
                                             StringRef def) {
  auto attr = op->getAttrOfType<StringAttr>(attrName);
  if (attr)
    return attr.getValue();
  return def;
}

namespace {

// Base class for all ESI signaling standards, which handles potentially funky
// attribute-based naming conventions.
struct ESISignalingStandad : public PortConversion {
  ESISignalingStandad(PortConverterImpl &converter, hw::PortInfo origPort)
      : PortConversion(converter, origPort) {}
};

/// Implement the Valid/Ready signaling standard.
class ValidReady : public ESISignalingStandad {
public:
  ValidReady(PortConverterImpl &converter, hw::PortInfo origPort)
      : ESISignalingStandad(converter, origPort), validPort(origPort),
        readyPort(origPort) {}

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  void buildInputSignals() override;
  void buildOutputSignals() override;

  // Keep around information about the port numbers of the relevant ports and
  // use that later to update the instances.
  PortInfo validPort, readyPort, dataPort;
};

/// Implement the FIFO signaling standard.
class FIFO : public ESISignalingStandad {
public:
  FIFO(PortConverterImpl &converter, hw::PortInfo origPort)
      : ESISignalingStandad(converter, origPort) {}

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  void buildInputSignals() override;
  void buildOutputSignals() override;

  // Keep around information about the port numbers of the relevant ports and
  // use that later to update the instances.
  PortInfo rdenPort, emptyPort, dataPort;
};

class ESIPortConversionBuilder : public PortConversionBuilder {
public:
  using PortConversionBuilder::PortConversionBuilder;
  FailureOr<std::unique_ptr<PortConversion>> build(hw::PortInfo port) override {
    return llvm::TypeSwitch<Type, FailureOr<std::unique_ptr<PortConversion>>>(
               port.type)
        .Case([&](esi::ChannelType chanTy)
                  -> FailureOr<std::unique_ptr<PortConversion>> {
          // Determine which ESI signaling standard is specified.
          ChannelSignaling signaling = chanTy.getSignaling();
          if (signaling == ChannelSignaling::ValidReady)
            return {std::make_unique<ValidReady>(converter, port)};

          if (signaling == ChannelSignaling::FIFO0)
            return {std::make_unique<FIFO>(converter, port)};

          auto error = converter.getModule().emitOpError(
                           "encountered unknown signaling standard on port '")
                       << stringifyEnum(signaling) << "'";
          error.attachNote(port.loc);
          return error;
        })
        .Default([&](auto) { return PortConversionBuilder::build(port); });
  }
};
} // namespace

void ValidReady::buildInputSignals() {
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);

  StringRef inSuffix =
      getStringAttributeOr(converter.getModule(), extModPortInSuffix, "");
  StringRef validSuffix(getStringAttributeOr(converter.getModule(),
                                             extModPortValidSuffix, "_valid"));

  // When we find one, add a data and valid signal to the new args.
  Value data = converter.createNewInput(
      origPort, inSuffix, cast<esi::ChannelType>(origPort.type).getInner(),
      dataPort);
  Value valid =
      converter.createNewInput(origPort, validSuffix + inSuffix, i1, validPort);

  Value ready;
  if (body) {
    ImplicitLocOpBuilder b(origPort.loc, body, body->begin());
    // Build the ESI wrap operation to translate the lowered signals to what
    // they were. (A later pass takes care of eliminating the ESI ops.)
    auto wrap = b.create<WrapValidReadyOp>(data, valid);
    ready = wrap.getReady();
    // Replace uses of the old ESI port argument with the new one from the
    // wrap.
    body->getArgument(origPort.argNum).replaceAllUsesWith(wrap.getChanOutput());
  }

  StringRef readySuffix = getStringAttributeOr(converter.getModule(),
                                               extModPortReadySuffix, "_ready");
  StringRef outSuffix =
      getStringAttributeOr(converter.getModule(), extModPortOutSuffix, "");
  converter.createNewOutput(origPort, readySuffix + outSuffix, i1, ready,
                            readyPort);
}

void ValidReady::mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                                 SmallVectorImpl<Value> &newOperands,
                                 ArrayRef<Backedge> newResults) {
  auto unwrap = b.create<UnwrapValidReadyOp>(inst->getLoc(),
                                             inst->getOperand(origPort.argNum),
                                             newResults[readyPort.argNum]);
  newOperands[dataPort.argNum] = unwrap.getRawOutput();
  newOperands[validPort.argNum] = unwrap.getValid();
}

void ValidReady::buildOutputSignals() {
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);

  StringRef readySuffix = getStringAttributeOr(converter.getModule(),
                                               extModPortReadySuffix, "_ready");
  StringRef inSuffix =
      getStringAttributeOr(converter.getModule(), extModPortInSuffix, "");

  Value ready =
      converter.createNewInput(origPort, readySuffix + inSuffix, i1, readyPort);
  Value data, valid;
  if (body) {
    auto *terminator = body->getTerminator();
    ImplicitLocOpBuilder b(origPort.loc, terminator);

    auto unwrap = b.create<UnwrapValidReadyOp>(
        terminator->getOperand(origPort.argNum), ready);
    data = unwrap.getRawOutput();
    valid = unwrap.getValid();
  }

  // New outputs.
  StringRef outSuffix =
      getStringAttributeOr(converter.getModule(), extModPortOutSuffix, "");
  StringRef validSuffix = getStringAttributeOr(converter.getModule(),
                                               extModPortValidSuffix, "_valid");
  converter.createNewOutput(origPort, outSuffix,
                            origPort.type.cast<esi::ChannelType>().getInner(),
                            data, dataPort);
  converter.createNewOutput(origPort, validSuffix + outSuffix, i1, valid,
                            validPort);
}

void ValidReady::mapOutputSignals(OpBuilder &b, Operation *inst,
                                  Value instValue,
                                  SmallVectorImpl<Value> &newOperands,
                                  ArrayRef<Backedge> newResults) {
  auto wrap =
      b.create<WrapValidReadyOp>(inst->getLoc(), newResults[dataPort.argNum],
                                 newResults[validPort.argNum]);
  inst->getResult(origPort.argNum).replaceAllUsesWith(wrap.getChanOutput());
  newOperands[readyPort.argNum] = wrap.getReady();
}

void FIFO::buildInputSignals() {
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);
  auto chanTy = origPort.type.cast<ChannelType>();

  StringRef rdenSuffix(getStringAttributeOr(converter.getModule(),
                                            extModPortRdenSuffix, "_rden"));
  StringRef emptySuffix(getStringAttributeOr(converter.getModule(),
                                             extModPortEmptySuffix, "_empty"));
  StringRef inSuffix =
      getStringAttributeOr(converter.getModule(), extModPortInSuffix, "");
  StringRef outSuffix =
      getStringAttributeOr(converter.getModule(), extModPortOutSuffix, "");

  // When we find one, add a data and valid signal to the new args.
  Value data = converter.createNewInput(
      origPort, inSuffix, origPort.type.cast<esi::ChannelType>().getInner(),
      dataPort);
  Value empty =
      converter.createNewInput(origPort, emptySuffix + inSuffix, i1, emptyPort);

  Value rden;
  if (body) {
    ImplicitLocOpBuilder b(origPort.loc, body, body->begin());
    // Build the ESI wrap operation to translate the lowered signals to what
    // they were. (A later pass takes care of eliminating the ESI ops.)
    auto wrap = b.create<WrapFIFOOp>(ArrayRef<Type>({chanTy, b.getI1Type()}),
                                     data, empty);
    rden = wrap.getRden();
    // Replace uses of the old ESI port argument with the new one from the
    // wrap.
    body->getArgument(origPort.argNum).replaceAllUsesWith(wrap.getChanOutput());
  }

  converter.createNewOutput(origPort, rdenSuffix + outSuffix, i1, rden,
                            rdenPort);
}

void FIFO::mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                           SmallVectorImpl<Value> &newOperands,
                           ArrayRef<Backedge> newResults) {
  auto unwrap =
      b.create<UnwrapFIFOOp>(inst->getLoc(), inst->getOperand(origPort.argNum),
                             newResults[rdenPort.argNum]);
  newOperands[dataPort.argNum] = unwrap.getData();
  newOperands[emptyPort.argNum] = unwrap.getEmpty();
}

void FIFO::buildOutputSignals() {
  Type i1 = IntegerType::get(getContext(), 1, IntegerType::Signless);

  StringRef inSuffix =
      getStringAttributeOr(converter.getModule(), extModPortInSuffix, "");
  StringRef outSuffix =
      getStringAttributeOr(converter.getModule(), extModPortOutSuffix, "");
  StringRef rdenSuffix(getStringAttributeOr(converter.getModule(),
                                            extModPortRdenSuffix, "_rden"));
  StringRef emptySuffix(getStringAttributeOr(converter.getModule(),
                                             extModPortEmptySuffix, "_empty"));
  Value rden =
      converter.createNewInput(origPort, rdenSuffix + inSuffix, i1, rdenPort);
  Value data, empty;
  if (body) {
    auto *terminator = body->getTerminator();
    ImplicitLocOpBuilder b(origPort.loc, terminator);

    auto unwrap =
        b.create<UnwrapFIFOOp>(terminator->getOperand(origPort.argNum), rden);
    data = unwrap.getData();
    empty = unwrap.getEmpty();
  }

  // New outputs.
  converter.createNewOutput(origPort, outSuffix,
                            origPort.type.cast<esi::ChannelType>().getInner(),
                            data, dataPort);
  converter.createNewOutput(origPort, emptySuffix + outSuffix, i1, empty,
                            emptyPort);
}

void FIFO::mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                            SmallVectorImpl<Value> &newOperands,
                            ArrayRef<Backedge> newResults) {
  auto wrap = b.create<WrapFIFOOp>(
      inst->getLoc(), ArrayRef<Type>({origPort.type, b.getI1Type()}),
      newResults[dataPort.argNum], newResults[emptyPort.argNum]);
  inst->getResult(origPort.argNum).replaceAllUsesWith(wrap.getChanOutput());
  newOperands[rdenPort.argNum] = wrap.getRden();
}

namespace {
/// Convert all the ESI ports on modules to some lower construct. SV
/// interfaces for now on external modules, ready/valid to modules defined
/// internally. In the future, it may be possible to select a different
/// format.
struct ESIPortsPass : public LowerESIPortsBase<ESIPortsPass> {
  void runOnOperation() override;

private:
  bool updateFunc(HWModuleExternOp mod);
  void updateInstance(HWModuleExternOp mod, InstanceOp inst);
  ESIHWBuilder *build;
};
} // anonymous namespace

/// Iterate through the `hw.module[.extern]`s and lower their ports.
void ESIPortsPass::runOnOperation() {
  ModuleOp top = getOperation();
  ESIHWBuilder b(top);
  build = &b;

  // Find all externmodules and try to modify them. Remember the modified
  // ones.
  DenseMap<SymbolRefAttr, HWModuleExternOp> externModsMutated;
  for (auto mod : top.getOps<HWModuleExternOp>())
    if (mod->hasAttrOfType<UnitAttr>(extModBundleSignalsAttrName) &&
        updateFunc(mod))
      externModsMutated[FlatSymbolRefAttr::get(mod)] = mod;

  // Find all instances and update them.
  top.walk([&externModsMutated, this](InstanceOp inst) {
    auto mapIter = externModsMutated.find(inst.getModuleNameAttr());
    if (mapIter != externModsMutated.end())
      updateInstance(mapIter->second, inst);
  });

  // Find all modules and run port conversion on them.
  circt::hw::InstanceGraph &instanceGraph =
      getAnalysis<circt::hw::InstanceGraph>();

  for (auto mod : top.getOps<HWMutableModuleLike>()) {
    if (failed(
            PortConverter<ESIPortConversionBuilder>(instanceGraph, mod).run()))
      return signalPassFailure();
  }

  build = nullptr;
}

/// Convert all input and output ChannelTypes into SV Interfaces. For inputs,
/// just switch the type to `ModportType`. For outputs, append a `ModportType`
/// to the inputs and remove the output channel from the results. Returns true
/// if 'mod' was updated. Delay updating the instances to amortize the IR walk
/// over all the module updates.
bool ESIPortsPass::updateFunc(HWModuleExternOp mod) {
  auto *ctxt = &getContext();

  bool updated = false;

  SmallVector<Attribute> newArgNames, newArgLocs, newResultNames, newResultLocs;

  // Reconstruct the list of operand types, changing the type whenever an ESI
  // port is found.
  SmallVector<Type, 16> newArgTypes;
  size_t nextArgNo = 0;
  for (auto argTy : mod.getArgumentTypes()) {
    auto chanTy = argTy.dyn_cast<ChannelType>();
    newArgNames.push_back(getModuleArgumentNameAttr(mod, nextArgNo));
    newArgLocs.push_back(getModuleArgumentLocAttr(mod, nextArgNo));
    nextArgNo++;

    if (!chanTy) {
      newArgTypes.push_back(argTy);
      continue;
    }

    // When we find one, construct an interface, and add the 'source' modport
    // to the type list.
    auto iface = build->getOrConstructInterface(chanTy);
    newArgTypes.push_back(iface.getModportType(ESIHWBuilder::sourceStr));
    updated = true;
  }

  // Iterate through the results and append to one of the two below lists. The
  // first for non-ESI-ports. The second, ports which have been re-located to
  // an operand.
  SmallVector<Type, 8> newResultTypes;
  SmallVector<DictionaryAttr, 4> newResultAttrs;
  auto funcType = mod.getFunctionType();
  for (size_t resNum = 0, numRes = mod.getNumResults(); resNum < numRes;
       ++resNum) {
    Type resTy = funcType.getResult(resNum);
    auto chanTy = resTy.dyn_cast<ChannelType>();
    auto resNameAttr = getModuleResultNameAttr(mod, resNum);
    auto resLocAttr = getModuleResultLocAttr(mod, resNum);
    if (!chanTy) {
      newResultTypes.push_back(resTy);
      newResultNames.push_back(resNameAttr);
      newResultLocs.push_back(resLocAttr);
      continue;
    }

    // When we find one, construct an interface, and add the 'sink' modport to
    // the type list.
    sv::InterfaceOp iface = build->getOrConstructInterface(chanTy);
    sv::ModportType sinkPort = iface.getModportType(ESIHWBuilder::sinkStr);
    newArgTypes.push_back(sinkPort);
    newArgNames.push_back(resNameAttr);
    newArgLocs.push_back(resLocAttr);
    updated = true;
  }

  mod->removeAttr(extModBundleSignalsAttrName);
  if (!updated)
    return false;

  // Set the new types.
  auto newFuncType = FunctionType::get(ctxt, newArgTypes, newResultTypes);
  mod.setType(newFuncType);
  setModuleArgumentNames(mod, newArgNames);
  setModuleArgumentLocs(mod, newArgLocs);
  setModuleResultNames(mod, newResultNames);
  setModuleResultLocs(mod, newResultLocs);
  return true;
}

static StringRef getOperandName(Value operand) {
  if (BlockArgument arg = operand.dyn_cast<BlockArgument>()) {
    auto *op = arg.getParentBlock()->getParentOp();
    if (op && hw::isAnyModule(op))
      return hw::getModuleArgumentName(op, arg.getArgNumber());
  } else {
    auto *srcOp = operand.getDefiningOp();
    if (auto instOp = dyn_cast<InstanceOp>(srcOp))
      return instOp.getInstanceName();

    if (auto srcName = srcOp->getAttrOfType<StringAttr>("name"))
      return srcName.getValue();
  }
  return "";
}

/// Create a reasonable name for a SV interface instance.
static std::string &constructInstanceName(Value operand, sv::InterfaceOp iface,
                                          std::string &name) {
  llvm::raw_string_ostream s(name);
  // Drop the "IValidReady_" part of the interface name.
  s << llvm::toLower(iface.getSymName()[12]) << iface.getSymName().substr(13);

  // Indicate to where the source is connected.
  if (operand.hasOneUse()) {
    Operation *dstOp = *operand.getUsers().begin();
    if (auto instOp = dyn_cast<InstanceOp>(dstOp))
      s << "To" << llvm::toUpper(instOp.getInstanceName()[0])
        << instOp.getInstanceName().substr(1);
    else if (auto dstName = dstOp->getAttrOfType<StringAttr>("name"))
      s << "To" << dstName.getValue();
  }

  // Indicate to where the sink is connected.
  StringRef operName = getOperandName(operand);
  if (!operName.empty())
    s << "From" << llvm::toUpper(operName[0]) << operName.substr(1);
  return s.str();
}

/// Update an instance of an updated module by adding `esi.(un)wrap.iface`
/// around the instance. Create a new instance at the end from the lists built
/// up before.
void ESIPortsPass::updateInstance(HWModuleExternOp mod, InstanceOp inst) {
  using namespace circt::sv;
  circt::ImplicitLocOpBuilder instBuilder(inst.getLoc(), inst);
  FunctionType funcTy = mod.getFunctionType();

  // op counter for error reporting purposes.
  size_t opNum = 0;
  // List of new operands.
  SmallVector<Value, 16> newOperands;

  // Fill the new operand list with old plain operands and mutated ones.
  std::string nameStringBuffer; // raw_string_ostream uses std::string.
  for (auto op : inst.getOperands()) {
    auto instChanTy = op.getType().dyn_cast<ChannelType>();
    if (!instChanTy) {
      newOperands.push_back(op);
      ++opNum;
      continue;
    }

    // Get the interface from the cache, and make sure it's the same one as
    // being used in the module.
    auto iface = build->getOrConstructInterface(instChanTy);
    if (iface.getModportType(ESIHWBuilder::sourceStr) !=
        funcTy.getInput(opNum)) {
      inst.emitOpError("ESI ChannelType (operand #")
          << opNum << ") doesn't match module!";
      ++opNum;
      newOperands.push_back(op);
      continue;
    }
    ++opNum;

    // Build a gasket by instantiating an interface, connecting one end to an
    // `esi.unwrap.iface` and the other end to the instance.
    auto ifaceInst =
        instBuilder.create<InterfaceInstanceOp>(iface.getInterfaceType());
    nameStringBuffer.clear();
    ifaceInst->setAttr(
        "name",
        StringAttr::get(mod.getContext(),
                        constructInstanceName(op, iface, nameStringBuffer)));
    GetModportOp sinkModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sinkStr);
    instBuilder.create<UnwrapSVInterfaceOp>(op, sinkModport);
    GetModportOp sourceModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sourceStr);
    // Finally, add the correct modport to the list of operands.
    newOperands.push_back(sourceModport);
  }

  // Go through the results and get both a list of the plain old values being
  // produced and their types.
  SmallVector<Value, 8> newResults;
  SmallVector<Type, 8> newResultTypes;
  for (size_t resNum = 0, numRes = inst.getNumResults(); resNum < numRes;
       ++resNum) {
    Value res = inst.getResult(resNum);
    auto instChanTy = res.getType().dyn_cast<ChannelType>();
    if (!instChanTy) {
      newResults.push_back(res);
      newResultTypes.push_back(res.getType());
      continue;
    }

    // Get the interface from the cache, and make sure it's the same one as
    // being used in the module.
    auto iface = build->getOrConstructInterface(instChanTy);
    if (iface.getModportType(ESIHWBuilder::sinkStr) != funcTy.getInput(opNum)) {
      inst.emitOpError("ESI ChannelType (result #")
          << resNum << ", operand #" << opNum << ") doesn't match module!";
      ++opNum;
      newResults.push_back(res);
      newResultTypes.push_back(res.getType());
      continue;
    }
    ++opNum;

    // Build a gasket by instantiating an interface, connecting one end to an
    // `esi.wrap.iface` and the other end to the instance. Append it to the
    // operand list.
    auto ifaceInst =
        instBuilder.create<InterfaceInstanceOp>(iface.getInterfaceType());
    nameStringBuffer.clear();
    ifaceInst->setAttr(
        "name",
        StringAttr::get(mod.getContext(),
                        constructInstanceName(res, iface, nameStringBuffer)));
    GetModportOp sourceModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sourceStr);
    auto newChannel =
        instBuilder.create<WrapSVInterfaceOp>(res.getType(), sourceModport);
    // Connect all the old users of the output channel with the newly
    // wrapped replacement channel.
    res.replaceAllUsesWith(newChannel);
    GetModportOp sinkModport =
        instBuilder.create<GetModportOp>(ifaceInst, ESIHWBuilder::sinkStr);
    // And add the modport on the other side to the new operand list.
    newOperands.push_back(sinkModport);
  }

  // Create the new instance!
  InstanceOp newInst = instBuilder.create<InstanceOp>(
      mod, inst.getInstanceNameAttr(), newOperands, inst.getParameters(),
      inst.getInnerSymAttr());

  // Go through the old list of non-ESI result values, and replace them with
  // the new non-ESI results.
  for (size_t resNum = 0, numRes = newResults.size(); resNum < numRes;
       ++resNum) {
    newResults[resNum].replaceAllUsesWith(newInst.getResult(resNum));
  }
  // Erase the old instance!
  inst.erase();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIPortLoweringPass() {
  return std::make_unique<ESIPortsPass>();
}
