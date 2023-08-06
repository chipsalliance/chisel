//===- ESIPasses.cpp - Common code for ESI passes ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace circt::esi;
using namespace circt::esi::detail;
using namespace circt::hw;
using namespace circt::sv;

//===----------------------------------------------------------------------===//
// ESI custom op builder.
//===----------------------------------------------------------------------===//

// C++ requires this for showing it what object file it should store these
// symbols in. They should be inline but that feature wasn't added until C++17.
constexpr char ESIHWBuilder::dataStr[], ESIHWBuilder::validStr[],
    ESIHWBuilder::readyStr[], ESIHWBuilder::sourceStr[],
    ESIHWBuilder::sinkStr[];

ESIHWBuilder::ESIHWBuilder(Operation *top)
    : ImplicitLocOpBuilder(UnknownLoc::get(top->getContext()), top),
      a(StringAttr::get(getContext(), "a")),
      aValid(StringAttr::get(getContext(), "a_valid")),
      aReady(StringAttr::get(getContext(), "a_ready")),
      x(StringAttr::get(getContext(), "x")),
      xValid(StringAttr::get(getContext(), "x_valid")),
      xReady(StringAttr::get(getContext(), "x_ready")),
      dataOutValid(StringAttr::get(getContext(), "DataOutValid")),
      dataOutReady(StringAttr::get(getContext(), "DataOutReady")),
      dataOut(StringAttr::get(getContext(), "DataOut")),
      dataInValid(StringAttr::get(getContext(), "DataInValid")),
      dataInReady(StringAttr::get(getContext(), "DataInReady")),
      dataIn(StringAttr::get(getContext(), "DataIn")),
      clk(StringAttr::get(getContext(), "clk")),
      rst(StringAttr::get(getContext(), "rst")),
      width(StringAttr::get(getContext(), "WIDTH")) {

  auto regions = top->getRegions();
  if (regions.empty()) {
    top->emitError("ESI HW Builder needs a region to insert HW.");
  }
  auto &region = regions.front();
  if (!region.empty())
    setInsertionPoint(&region.front(), region.front().begin());
}

static StringAttr constructUniqueSymbol(Operation *tableOp,
                                        StringRef proposedNameRef) {
  SmallString<64> proposedName = proposedNameRef;

  // Normalize the type name.
  for (char &ch : proposedName) {
    if (isalpha(ch) || isdigit(ch) || ch == '_')
      continue;
    ch = '_';
  }

  // Make sure that this symbol isn't taken. If it is, append a number and try
  // again.
  size_t baseLength = proposedName.size();
  size_t tries = 0;
  while (SymbolTable::lookupSymbolIn(tableOp, proposedName)) {
    proposedName.resize(baseLength);
    proposedName.append(llvm::utostr(++tries));
  }

  return StringAttr::get(tableOp->getContext(), proposedName);
}

StringAttr ESIHWBuilder::constructInterfaceName(ChannelType port) {
  Operation *tableOp =
      getInsertionPoint()->getParentWithTrait<mlir::OpTrait::SymbolTable>();

  // Get a name based on the type.
  std::string portTypeName;
  llvm::raw_string_ostream nameOS(portTypeName);
  TypeSwitch<Type>(port.getInner())
      .Case([&](hw::ArrayType arr) {
        nameOS << "ArrayOf" << arr.getSize() << 'x' << arr.getElementType();
      })
      .Case([&](hw::StructType t) { nameOS << "Struct"; })
      .Default([&](Type t) { nameOS << port.getInner(); });

  // Don't allow the name to end with '_'.
  ssize_t i = portTypeName.size() - 1;
  while (i >= 0 && portTypeName[i] == '_') {
    --i;
  }
  portTypeName = portTypeName.substr(0, i + 1);

  // All stage names start with this.
  SmallString<64> proposedName("IValidReady_");
  proposedName.append(portTypeName);
  return constructUniqueSymbol(tableOp, proposedName);
}

/// Return a parameter list for the stage module with the specified value.
ArrayAttr ESIHWBuilder::getStageParameterList(Attribute value) {
  auto type = IntegerType::get(width.getContext(), 32, IntegerType::Unsigned);
  auto widthParam = ParamDeclAttr::get(width.getContext(), width, type, value);
  return ArrayAttr::get(width.getContext(), widthParam);
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module implements pipeline stage, adding 1 cycle latency. This particular
/// implementation is double-buffered and fully pipelines the reverse-flow ready
/// signal.
HWModuleExternOp ESIHWBuilder::declareStage(Operation *symTable,
                                            PipelineStageOp stage) {
  Type dataType = stage.innerType();
  HWModuleExternOp &stageMod = declaredStage[dataType];
  if (stageMod)
    return stageMod;

  // Since this module has parameterized widths on the a input and x output,
  // give the extern declation a None type since nothing else makes sense.
  // Will be refining this when we decide how to better handle parameterized
  // types and ops.
  size_t argn = 0;
  size_t resn = 0;
  llvm::SmallVector<PortInfo> ports = {
      {{clk, getI1Type(), ModulePort::Direction::Input}, argn++},
      {{rst, getI1Type(), ModulePort::Direction::Input}, argn++}};

  ports.push_back({{a, dataType, ModulePort::Direction::Input}, argn++});
  ports.push_back(
      {{aValid, getI1Type(), ModulePort::Direction::Input}, argn++});
  ports.push_back(
      {{aReady, getI1Type(), ModulePort::Direction::Output}, resn++});
  ports.push_back({{x, dataType, ModulePort::Direction::Output}, resn++});

  ports.push_back(
      {{xValid, getI1Type(), ModulePort::Direction::Output}, resn++});
  ports.push_back(
      {{xReady, getI1Type(), ModulePort::Direction::Input}, argn++});

  stageMod = create<HWModuleExternOp>(
      constructUniqueSymbol(symTable, "ESI_PipelineStage"), ports,
      "ESI_PipelineStage", getStageParameterList({}));
  return stageMod;
}

/// Write an 'ExternModuleOp' to use a hand-coded SystemVerilog module. Said
/// module contains a bi-directional Cosimulation DPI interface with valid/ready
/// semantics.
HWModuleExternOp ESIHWBuilder::declareCosimEndpointOp(Operation *symTable,
                                                      Type sendType,
                                                      Type recvType) {
  HWModuleExternOp &endpoint =
      declaredCosimEndpointOp[std::make_pair(sendType, recvType)];
  if (endpoint)
    return endpoint;
  // Since this module has parameterized widths on the a input and x output,
  // give the extern declation a None type since nothing else makes sense.
  // Will be refining this when we decide how to better handle parameterized
  // types and ops.
  PortInfo ports[] = {
      {{clk, getI1Type(), ModulePort::Direction::Input}, 0},
      {{rst, getI1Type(), ModulePort::Direction::Input}, 1},
      {{dataOutValid, getI1Type(), ModulePort::Direction::Output}, 0},
      {{dataOutReady, getI1Type(), ModulePort::Direction::Input}, 2},
      {{dataOut, recvType, ModulePort::Direction::Output}, 1},
      {{dataInValid, getI1Type(), ModulePort::Direction::Input}, 3},
      {{dataInReady, getI1Type(), ModulePort::Direction::Output}, 2},
      {{dataIn, sendType, ModulePort::Direction::Input}, 4}};
  SmallVector<Attribute, 8> params;
  params.push_back(ParamDeclAttr::get("ENDPOINT_ID_EXT", getStringAttr("")));
  params.push_back(
      ParamDeclAttr::get("SEND_TYPE_ID", getIntegerType(64, false)));
  params.push_back(ParamDeclAttr::get("SEND_TYPE_SIZE_BITS", getI32Type()));
  params.push_back(
      ParamDeclAttr::get("RECV_TYPE_ID", getIntegerType(64, false)));
  params.push_back(ParamDeclAttr::get("RECV_TYPE_SIZE_BITS", getI32Type()));
  endpoint = create<HWModuleExternOp>(
      constructUniqueSymbol(symTable, "Cosim_Endpoint"), ports,
      "Cosim_Endpoint", ArrayAttr::get(getContext(), params));
  return endpoint;
}

/// Return the InterfaceType which corresponds to an ESI port type. If it
/// doesn't exist in the cache, build the InterfaceOp and the corresponding
/// type.
InterfaceOp ESIHWBuilder::getOrConstructInterface(ChannelType t) {
  auto ifaceIter = portTypeLookup.find(t);
  if (ifaceIter != portTypeLookup.end())
    return ifaceIter->second;
  auto iface = constructInterface(t);
  portTypeLookup[t] = iface;
  return iface;
}

InterfaceOp ESIHWBuilder::constructInterface(ChannelType chan) {
  return create<InterfaceOp>(constructInterfaceName(chan).getValue(), [&]() {
    create<InterfaceSignalOp>(validStr, getI1Type());
    create<InterfaceSignalOp>(readyStr, getI1Type());
    create<InterfaceSignalOp>(dataStr, chan.getInner());
    llvm::SmallVector<StringRef> validDataStrs;
    validDataStrs.push_back(validStr);
    validDataStrs.push_back(dataStr);
    create<InterfaceModportOp>(sinkStr,
                               /*inputs=*/ArrayRef<StringRef>{readyStr},
                               /*outputs=*/validDataStrs);
    create<InterfaceModportOp>(sourceStr,
                               /*inputs=*/validDataStrs,
                               /*outputs=*/ArrayRef<StringRef>{readyStr});
  });
}

void circt::esi::registerESIPasses() { registerPasses(); }
