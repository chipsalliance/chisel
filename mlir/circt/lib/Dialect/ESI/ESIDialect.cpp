//===- ESIDialect.cpp - ESI dialect code defs -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect definitions. Should be relatively standard boilerplate.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace circt::esi;

void ESIDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/ESI/ESI.cpp.inc"
      >();
}

Operation *ESIDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  // Integer constants.
  if (auto intType = type.dyn_cast<IntegerType>())
    if (auto attrValue = value.dyn_cast<IntegerAttr>())
      return builder.create<hw::ConstantOp>(loc, attrValue.getType(),
                                            attrValue);
  if (value.isa<mlir::UnitAttr>())
    return builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 1);
  return nullptr;
}

/// Try to find a valid/ready port for the specified data port. If found, append
/// to 'names'.
static void findValidReady(Operation *modOp,
                           const llvm::StringMap<hw::PortInfo> &nameMap,
                           hw::PortInfo dataPort, bool trimName, bool warn,
                           SmallVectorImpl<ESIPortValidReadyMapping> &names) {
  if (dataPort.dir == hw::ModulePort::Direction::InOut) {
    if (warn)
      modOp->emitWarning("Data port '")
          << dataPort.getName() << "' cannot be inout direction.";
    return;
  }

  SmallString<64> name = dataPort.getName();
  size_t nameLen = name.size();
  if (trimName && name.endswith("_data")) { // Detect both `foo` and `foo_data`.
    nameLen -= 5;
    name.resize(nameLen);
  }

  // Look for a 'valid' port.
  name.append("_valid");
  auto valid = nameMap.find(name);
  if (valid == nameMap.end() || valid->second.dir != dataPort.dir ||
      !valid->second.type.isSignlessInteger(1)) {
    if (warn)
      modOp->emitWarning("Could not find appropriate valid port for '")
          << name << "'.";
    return;
  }

  // Try to find a corresponding 'ready' port.
  name.resize(nameLen);
  name.append("_ready");
  hw::ModulePort::Direction readyDir =
      dataPort.dir == hw::ModulePort::Direction::Input
          ? hw::ModulePort::Direction::Output
          : hw::ModulePort::Direction::Input;
  auto ready = nameMap.find(name);
  if (ready == nameMap.end() || ready->second.dir != readyDir ||
      !ready->second.type.isSignlessInteger(1)) {
    if (warn)
      modOp->emitWarning("Could not find appropriate ready port for '")
          << name << "'.";
    return;
  }

  // Found one.
  names.push_back(
      ESIPortValidReadyMapping{dataPort, valid->second, ready->second});
}

/// Find all the port triples on a module which fit the
/// <name>[_data]/<name>_valid/<name>_ready pattern. Ready must be the opposite
/// direction of the other two.
void circt::esi::findValidReadySignals(
    Operation *modOp, SmallVectorImpl<ESIPortValidReadyMapping> &names) {

  auto ports = cast<hw::PortList>(modOp).getPortList();
  llvm::StringMap<hw::PortInfo> nameMap(ports.size());
  for (auto port : ports)
    nameMap[port.getName()] = port;
  for (auto port : ports)
    findValidReady(modOp, nameMap, port, /*trimName=*/true, /*warn=*/false,
                   names);
}

/// Given a list of logical port names, find the data/valid/ready port triples.
void circt::esi::resolvePortNames(
    Operation *modOp, ArrayRef<StringRef> portNames,
    SmallVectorImpl<ESIPortValidReadyMapping> &names) {

  auto ports = cast<hw::PortList>(modOp).getPortList();
  llvm::StringMap<hw::PortInfo> nameMap(ports.size());
  for (auto port : ports)
    nameMap[port.getName()] = port;

  SmallString<64> nameBuffer;
  for (auto name : portNames) {
    nameBuffer = name;

    // Look for a 'data' port.
    auto it = nameMap.find(nameBuffer);
    if (it == nameMap.end()) {
      nameBuffer.append("_data");
      it = nameMap.find(nameBuffer);
      if (it == nameMap.end())
        modOp->emitWarning("Could not find data port '") << name << "'.";
      else
        findValidReady(modOp, nameMap, it->second, /*trimName=*/true,
                       /*warn=*/true, names);
    } else {
      findValidReady(modOp, nameMap, it->second, /*trimName=*/false,
                     /*warn=*/true, names);
    }
  }
}

/// Build an ESI module wrapper, converting the wires with latency-insensitive
/// semantics to ESI channels and passing through the rest.
Operation *
circt::esi::buildESIWrapper(OpBuilder &b, Operation *pearl,
                            ArrayRef<ESIPortValidReadyMapping> portsToConvert) {
  // In order to avoid the similar sounding and looking "wrapped" and "wrapper"
  // names or the ambiguous "module", we use "pearl" for the module _being
  // wrapped_ and "shell" for the _wrapper_ modules which is being created
  // (terms typically used in latency insensitive design papers).

  auto *ctxt = b.getContext();
  Location loc = pearl->getLoc();
  FunctionType modType = hw::getModuleType(pearl);

  // -----
  // First, build up a set of data structures to use throughout this function.

  StringSet<> controlPorts; // Memoize the list of ready/valid ports to ignore.
  llvm::StringMap<ESIPortValidReadyMapping>
      dataPortMap; // Store a lookup table of ports to convert indexed on the
                   // data port name.
  // Validate input and assemble lookup structures.
  for (const auto &esiPort : portsToConvert) {
    if (esiPort.data.dir == hw::ModulePort::Direction::InOut) {
      pearl->emitError("Data signal '")
          << esiPort.data.name << "' must not be INOUT";
      return nullptr;
    }
    dataPortMap[esiPort.data.name.getValue()] = esiPort;

    if (esiPort.valid.dir != esiPort.data.dir) {
      pearl->emitError("Valid port '")
          << esiPort.valid.name << "' direction must match data port.";
      return nullptr;
    }
    if (esiPort.valid.type != b.getI1Type()) {
      pearl->emitError("Valid signal '")
          << esiPort.valid.name << "' must be i1 type";
      return nullptr;
    }
    controlPorts.insert(esiPort.valid.name.getValue());

    if (esiPort.ready.dir != (esiPort.data.isOutput()
                                  ? hw::ModulePort::Direction::Input
                                  : hw::ModulePort::Direction::Output)) {
      pearl->emitError("Ready port '")
          << esiPort.ready.name
          << "' must be opposite direction to data signal.";
      return nullptr;
    }
    if (esiPort.ready.type != b.getI1Type()) {
      pearl->emitError("Ready signal '")
          << esiPort.ready.name << "' must be i1 type";
      return nullptr;
    }
    controlPorts.insert(esiPort.ready.name.getValue());
  }

  // -----
  // Second, build a list of ports for the shell module, skipping the
  // valid/ready, and converting the ESI data ports to the ESI channel port
  // type. Store some bookkeeping information.

  SmallVector<hw::PortInfo, 64> shellPorts;
  // Map the shell operand to the pearl port.
  SmallVector<hw::PortInfo, 64> inputPortMap;
  // Map the shell result to the pearl port.
  SmallVector<hw::PortInfo, 64> outputPortMap;

  auto pearlPorts = cast<hw::PortList>(pearl).getPortList();
  for (const auto &port : pearlPorts) {
    if (controlPorts.contains(port.name.getValue()))
      continue;

    hw::PortInfo newPort = port;
    if (dataPortMap.find(port.name.getValue()) != dataPortMap.end())
      newPort.type = esi::ChannelType::get(ctxt, port.type);

    if (port.isOutput()) {
      newPort.argNum = outputPortMap.size();
      outputPortMap.push_back(port);
    } else {
      newPort.argNum = inputPortMap.size();
      inputPortMap.push_back(port);
    }
    shellPorts.push_back(newPort);
  }

  // -----
  // Third, create the shell module and also some builders for the inside.

  SmallString<64> shellNameBuf;
  StringAttr shellName =
      b.getStringAttr((SymbolTable::getSymbolName(pearl).getValue() + "_esi")
                          .toStringRef(shellNameBuf));
  auto shell = b.create<hw::HWModuleOp>(loc, shellName, shellPorts);
  shell.getBodyBlock()->clear(); // Erase the terminator.
  auto modBuilder =
      ImplicitLocOpBuilder::atBlockBegin(loc, shell.getBodyBlock());
  BackedgeBuilder bb(modBuilder, modBuilder.getLoc());

  // Hold the operands for `hw.output` here.
  SmallVector<Value, 64> outputs(shell.getNumResults());

  // -----
  // Fourth, assemble the inputs for the pearl module AND build all the ESI wrap
  // and unwrap ops for both the input and output channels.

  SmallVector<Value, 64> pearlOperands(modType.getNumInputs());

  // Since we build all the ESI wrap and unwrap operations before pearl
  // instantiation, we only need backedges from the pearl instance result. Index
  // the backedges by the pearl modules result number.
  llvm::DenseMap<size_t, Backedge> backedges;

  // Go through the shell input ports, either tunneling them through or
  // unwrapping the ESI channels. We'll need backedges for the ready signals
  // since they are results from the upcoming pearl instance.
  for (const auto &port : shellPorts) {
    if (port.isOutput())
      continue;

    Value arg = shell.getArgument(port.argNum);
    auto esiPort = dataPortMap.find(port.name.getValue());
    if (esiPort == dataPortMap.end()) {
      // If it's just a regular port, it just gets passed through.
      size_t pearlOpNum = inputPortMap[port.argNum].argNum;
      pearlOperands[pearlOpNum] = arg;
      continue;
    }

    Backedge ready = bb.get(modBuilder.getI1Type());
    backedges.insert(std::make_pair(esiPort->second.ready.argNum, ready));
    auto unwrap = modBuilder.create<UnwrapValidReadyOp>(arg, ready);
    pearlOperands[esiPort->second.data.argNum] = unwrap.getRawOutput();
    pearlOperands[esiPort->second.valid.argNum] = unwrap.getValid();
  }

  // Iterate through the shell output ports, identify the ESI channels, and
  // build ESI wrapper ops for signals being output from the pearl. The data and
  // valid for these wrap ops will need to be backedges.
  for (const auto &port : shellPorts) {
    if (!port.isOutput())
      continue;
    auto esiPort = dataPortMap.find(port.name.getValue());
    if (esiPort == dataPortMap.end())
      continue;

    Backedge data = bb.get(esiPort->second.data.type);
    Backedge valid = bb.get(modBuilder.getI1Type());
    auto wrap = modBuilder.create<WrapValidReadyOp>(data, valid);
    backedges.insert(std::make_pair(esiPort->second.data.argNum, data));
    backedges.insert(std::make_pair(esiPort->second.valid.argNum, valid));
    outputs[port.argNum] = wrap.getChanOutput();
    pearlOperands[esiPort->second.ready.argNum] = wrap.getReady();
  }

  // -----
  // Fifth, instantiate the pearl module.

  auto pearlInst =
      modBuilder.create<hw::InstanceOp>(pearl, "pearl", pearlOperands);

  // Hookup all the backedges.
  for (size_t i = 0, e = pearlInst.getNumResults(); i < e; ++i) {
    auto backedge = backedges.find(i);
    if (backedge != backedges.end())
      backedge->second.setValue(pearlInst.getResult(i));
  }

  // -----
  // Finally, find all the regular outputs and either tunnel them through.
  for (const auto &port : shellPorts) {
    if (!port.isOutput())
      continue;
    auto esiPort = dataPortMap.find(port.name.getValue());
    if (esiPort != dataPortMap.end())
      continue;
    size_t pearlResNum = outputPortMap[port.argNum].argNum;
    outputs[port.argNum] = pearlInst.getResult(pearlResNum);
  }

  modBuilder.create<hw::OutputOp>(outputs);
  return shell;
}

// Provide implementations for the enums we use.
#include "circt/Dialect/ESI/ESIEnums.cpp.inc"

#include "circt/Dialect/ESI/ESIDialect.cpp.inc"
