//===- ESIServices.cpp - Code related to ESI services ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include <memory>
#include <utility>

using namespace circt;
using namespace circt::esi;

LogicalResult
ServiceGeneratorDispatcher::generate(ServiceImplementReqOp req,
                                     ServiceDeclOpInterface decl) {
  // Lookup based on 'impl_type' attribute and pass through the generate request
  // if found.
  auto genF = genLookupTable.find(req.getImplTypeAttr().getValue());
  if (genF == genLookupTable.end()) {
    if (failIfNotFound)
      return req.emitOpError("Could not find service generator for attribute '")
             << req.getImplTypeAttr() << "'";
    return success();
  }
  return genF->second(req, decl);
}

/// The generator for the "cosim" impl_type.
static LogicalResult instantiateCosimEndpointOps(ServiceImplementReqOp implReq,
                                                 ServiceDeclOpInterface) {
  auto *ctxt = implReq.getContext();
  OpBuilder b(implReq);
  Value clk = implReq.getOperand(0);
  Value rst = implReq.getOperand(1);

  // Determine which EndpointID this generator should start with.
  if (implReq.getImplOpts()) {
    auto opts = implReq.getImplOpts()->getValue();
    for (auto nameAttr : opts) {
      return implReq.emitOpError("did not recognize option name ")
             << nameAttr.getName();
    }
  }

  // Assemble the name to use for an endpoint.
  auto toStringAttr = [&](ArrayAttr strArr) {
    std::string buff;
    llvm::raw_string_ostream os(buff);
    llvm::interleave(strArr.getAsValueRange<StringAttr>(), os, ".");
    return StringAttr::get(ctxt, os.str());
  };

  llvm::DenseMap<ServiceReqOpInterface, unsigned> toClientResultNum;
  for (auto req : implReq.getOps<ServiceReqOpInterface>())
    if (req.getToClient())
      toClientResultNum[req] = toClientResultNum.size();

  // Iterate through them, building a cosim endpoint for each one.
  for (auto req : implReq.getOps<ServiceReqOpInterface>()) {
    Location loc = req->getLoc();
    ArrayAttr clientNamePathAttr = req.getClientNamePath();

    Value toServerValue = req.getToServer();
    if (!toServerValue)
      toServerValue =
          b.create<NullSourceOp>(loc, ChannelType::get(ctxt, b.getI1Type()));

    Type toClientType = req.getToClientType();
    if (!toClientType)
      toClientType = ChannelType::get(ctxt, b.getI1Type());

    auto cosim =
        b.create<CosimEndpointOp>(loc, toClientType, clk, rst, toServerValue,
                                  toStringAttr(clientNamePathAttr));

    if (req.getToClient()) {
      unsigned clientReqIdx = toClientResultNum[req];
      implReq.getResult(clientReqIdx).replaceAllUsesWith(cosim.getRecv());
    }
  }

  // Erase the generation request.
  implReq.erase();
  return success();
}

// Generator for "sv_mem" implementation type. Emits SV ops for an unpacked
// array, hopefully inferred as a memory to the SV compiler.
static LogicalResult
instantiateSystemVerilogMemory(ServiceImplementReqOp implReq,
                               ServiceDeclOpInterface decl) {
  if (!decl)
    return implReq.emitOpError(
        "Must specify a service declaration to use 'sv_mem'.");

  ImplicitLocOpBuilder b(implReq.getLoc(), implReq);
  BackedgeBuilder bb(b, implReq.getLoc());

  RandomAccessMemoryDeclOp ramDecl =
      dyn_cast<RandomAccessMemoryDeclOp>(decl.getOperation());
  if (!ramDecl)
    return implReq.emitOpError(
        "'sv_mem' implementation type can only be used to "
        "implement RandomAccessMemory declarations");

  if (implReq.getNumOperands() != 2)
    return implReq.emitOpError("Implementation requires clk and rst operands");
  auto clk = implReq.getOperand(0);
  auto rst = implReq.getOperand(1);
  auto write = b.getStringAttr("write");
  auto read = b.getStringAttr("read");
  auto none = b.create<hw::ConstantOp>(
      APInt(/*numBits*/ 0, /*val*/ 0, /*isSigned*/ false));
  auto i1 = b.getI1Type();
  auto c0 = b.create<hw::ConstantOp>(i1, 0);

  // List of reqs which have a result.
  SmallVector<ServiceReqOpInterface, 8> toClientReqs(llvm::make_filter_range(
      implReq.getOps<ServiceReqOpInterface>(),
      [](auto req) { return req.getToClient() != nullptr; }));

  // Assemble a mapping of toClient results to actual consumers.
  DenseMap<Value, Value> outputMap;
  for (auto [bout, reqout] :
       llvm::zip_longest(toClientReqs, implReq.getResults())) {
    assert(bout.has_value());
    assert(reqout.has_value());
    Value toClient = bout->getToClient();
    outputMap[toClient] = *reqout;
  }

  // Create the SV memory.
  hw::UnpackedArrayType memType =
      hw::UnpackedArrayType::get(ramDecl.getInnerType(), ramDecl.getDepth());
  auto mem =
      b.create<sv::RegOp>(memType, implReq.getServiceSymbolAttr().getAttr())
          .getResult();

  // Do everything which doesn't actually write to the memory, store the signals
  // needed for the actual memory writes for later.
  SmallVector<std::tuple<Value, Value, Value>> writeGoAddressData;
  for (auto req : implReq.getOps<ServiceReqOpInterface>()) {
    auto port = req.getServicePort().getName();
    WrapValidReadyOp toClientResp;

    if (port == write) {
      // If this pair is doing a write...
      auto ioReq = dyn_cast<RequestInOutChannelOp>(*req);
      if (!ioReq)
        return req->emitOpError("Memory write requests must be to/from server");

      // Construct the response channel.
      auto doneValid = bb.get(i1);
      toClientResp = b.create<WrapValidReadyOp>(none, doneValid);

      // Unwrap the write request and 'explode' the struct.
      auto unwrap = b.create<UnwrapValidReadyOp>(ioReq.getToServer(),
                                                 toClientResp.getReady());

      Value address = b.create<hw::StructExtractOp>(unwrap.getRawOutput(),
                                                    b.getStringAttr("address"));
      Value data = b.create<hw::StructExtractOp>(unwrap.getRawOutput(),
                                                 b.getStringAttr("data"));

      // Determine if the write should occur this cycle.
      auto go = b.create<comb::AndOp>(unwrap.getValid(), unwrap.getReady());
      go->setAttr("sv.namehint", b.getStringAttr("write_go"));
      // Register the 'go' signal and use it as the done message.
      doneValid.setValue(
          b.create<seq::CompRegOp>(go, clk, rst, c0, "write_done"));
      // Store the necessary data for the 'always' memory writing block.
      writeGoAddressData.push_back(std::make_tuple(go, address, data));

    } else if (port == read) {
      // If it's a read...
      auto ioReq = dyn_cast<RequestInOutChannelOp>(*req);
      if (!ioReq)
        return req->emitOpError("Memory read requests must be to/from server");

      // Construct the response channel.
      auto dataValid = bb.get(i1);
      auto data = bb.get(ramDecl.getInnerType());
      toClientResp = b.create<WrapValidReadyOp>(data, dataValid);

      // Unwrap the requested address and read from that memory location.
      auto addressUnwrap = b.create<UnwrapValidReadyOp>(
          ioReq.getToServer(), toClientResp.getReady());
      Value memLoc =
          b.create<sv::ArrayIndexInOutOp>(mem, addressUnwrap.getRawOutput());
      auto readData = b.create<sv::ReadInOutOp>(memLoc);

      // Set the data on the response.
      data.setValue(readData);
      dataValid.setValue(addressUnwrap.getValid());
    } else {
      assert(false && "Port should be either 'read' or 'write'");
    }

    outputMap[req.getToClient()].replaceAllUsesWith(
        toClientResp.getChanOutput());
  }

  // Now construct the memory writes.
  b.create<sv::AlwaysFFOp>(
      sv::EventControl::AtPosEdge, clk, ResetType::SyncReset,
      sv::EventControl::AtPosEdge, rst, [&] {
        for (auto [go, address, data] : writeGoAddressData) {
          Value a = address, d = data; // So the lambda can capture.
          // If we're told to go, do the write.
          b.create<sv::IfOp>(go, [&] {
            Value memLoc = b.create<sv::ArrayIndexInOutOp>(mem, a);
            b.create<sv::PAssignOp>(memLoc, d);
          });
        }
      });

  implReq.erase();
  return success();
}

static ServiceGeneratorDispatcher
    globalDispatcher({{"cosim", instantiateCosimEndpointOps},
                      {"sv_mem", instantiateSystemVerilogMemory}},
                     false);

ServiceGeneratorDispatcher &ServiceGeneratorDispatcher::globalDispatcher() {
  return ::globalDispatcher;
}

void ServiceGeneratorDispatcher::registerGenerator(StringRef implType,
                                                   ServiceGeneratorFunc gen) {
  genLookupTable[implType] = std::move(gen);
}

//===----------------------------------------------------------------------===//
// Wire up services pass.
//===----------------------------------------------------------------------===//

namespace {
/// Implements a pass to connect up ESI services clients to the nearest server
/// instantiation. Wires up the ports and generates a generation request to
/// call a user-specified generator.
struct ESIConnectServicesPass
    : public ESIConnectServicesBase<ESIConnectServicesPass>,
      msft::PassCommon {

  ESIConnectServicesPass(const ServiceGeneratorDispatcher &gen)
      : genDispatcher(gen) {}
  ESIConnectServicesPass()
      : genDispatcher(ServiceGeneratorDispatcher::globalDispatcher()) {}

  void runOnOperation() override;

  /// "Bubble up" the specified requests to all of the instantiations of the
  /// module specified. Create and connect up ports to tunnel the ESI channels
  /// through.
  LogicalResult surfaceReqs(hw::HWMutableModuleLike,
                            ArrayRef<ServiceReqOpInterface>);

  /// Copy all service metadata up the instance hierarchy. Modify the service
  /// name path while copying.
  void copyMetadata(hw::HWModuleLike);

  /// For any service which is "local" (provides the requested service) in a
  /// module, replace it with a ServiceImplementOp. Said op is to be replaced
  /// with an instantiation by a generator.
  LogicalResult replaceInst(ServiceInstanceOp, Block *portReqs);

  /// Figure out which requests are "local" vs need to be surfaced. Call
  /// 'surfaceReqs' and/or 'replaceInst' as appropriate.
  LogicalResult process(hw::HWModuleLike);

private:
  ServiceGeneratorDispatcher genDispatcher;
};
} // anonymous namespace

void ESIConnectServicesPass::runOnOperation() {
  ModuleOp outerMod = getOperation();
  topLevelSyms.addDefinitions(outerMod);
  if (failed(verifyInstances(outerMod))) {
    signalPassFailure();
    return;
  }

  // Get a partially-ordered list of modules based on the instantiation DAG.
  // It's _very_ important that we process modules before their instantiations
  // so that the modules where they're instantiated correctly process the
  // surfaced connections.
  SmallVector<hw::HWModuleLike, 64> sortedMods;
  getAndSortModules(outerMod, sortedMods);

  // Process each module.
  for (auto mod : sortedMods) {
    hw::HWModuleLike mutableMod = dyn_cast<hw::HWModuleLike>(*mod);
    if (mutableMod && failed(process(mutableMod))) {
      signalPassFailure();
      return;
    }
  }
}

LogicalResult ESIConnectServicesPass::process(hw::HWModuleLike mod) {
  // If 'mod' doesn't have a body, assume it's an external module.
  if (mod->getNumRegions() == 0 || mod->getRegion(0).empty())
    return success();

  Block &modBlock = mod->getRegion(0).front();

  // Index the local services and create blocks in which to put the requests.
  DenseMap<SymbolRefAttr, Block *> localImplReqs;
  Block *anyServiceInst = nullptr;
  for (auto instOp : modBlock.getOps<ServiceInstanceOp>()) {
    auto *b = new Block();
    localImplReqs[instOp.getServiceSymbolAttr()] = b;
    if (!instOp.getServiceSymbol().has_value())
      anyServiceInst = b;
  }

  // Find all of the "local" requests.
  mod.walk([&](ServiceReqOpInterface req) {
    auto service = req.getServicePort().getModuleRef();
    auto implOpF = localImplReqs.find(service);
    if (implOpF != localImplReqs.end())
      req->moveBefore(implOpF->second, implOpF->second->end());
    else if (anyServiceInst)
      req->moveBefore(anyServiceInst, anyServiceInst->end());
  });

  // Replace each service instance with a generation request. If a service
  // generator is registered, generate the server.
  for (auto instOp :
       llvm::make_early_inc_range(modBlock.getOps<ServiceInstanceOp>())) {
    Block *portReqs = localImplReqs[instOp.getServiceSymbolAttr()];
    if (failed(replaceInst(instOp, portReqs)))
      return failure();
  }

  // Copy any metadata up the instance hierarchy.
  copyMetadata(mod);

  // Identify the non-local reqs which need to be surfaced from this module.
  SmallVector<ServiceReqOpInterface, 4> nonLocalReqs;
  mod.walk([&](ServiceReqOpInterface req) {
    auto service = req.getServicePort().getModuleRef();
    auto implOpF = localImplReqs.find(service);
    if (implOpF == localImplReqs.end())
      nonLocalReqs.push_back(req);
  });

  // Surface all of the requests which cannot be fulfilled locally.
  if (nonLocalReqs.empty())
    return success();

  if (auto mutableMod = dyn_cast<hw::HWMutableModuleLike>(mod.getOperation()))
    return surfaceReqs(mutableMod, nonLocalReqs);
  return mod.emitOpError(
      "Cannot surface requests through module without mutable ports");
}

void ESIConnectServicesPass::copyMetadata(hw::HWModuleLike mod) {
  SmallVector<ServiceHierarchyMetadataOp, 8> metadataOps;
  mod.walk([&](ServiceHierarchyMetadataOp op) { metadataOps.push_back(op); });

  for (auto inst : moduleInstantiations[mod]) {
    OpBuilder b(inst);
    auto instName = inst.getInstanceNameAttr();
    for (auto metadata : metadataOps) {
      SmallVector<Attribute, 4> path;
      path.push_back(hw::InnerRefAttr::get(mod.getModuleNameAttr(), instName));
      for (auto attr : metadata.getServerNamePathAttr())
        path.push_back(attr);

      auto metadataCopy = cast<ServiceHierarchyMetadataOp>(b.clone(*metadata));
      metadataCopy.setServerNamePathAttr(b.getArrayAttr(path));
    }
  }
}

/// Create an op which contains metadata about the soon-to-be implemented
/// service. To be used by later passes which require these data (e.g.
/// automated software API creation).
static void emitServiceMetadata(ServiceImplementReqOp implReqOp) {
  ImplicitLocOpBuilder b(implReqOp.getLoc(), implReqOp);

  // Check if there are any "BSP" service providers -- ones which implement any
  // service -- and create an implicit service declaration for them.
  std::unique_ptr<Block> bspPorts = nullptr;
  if (!implReqOp.getServiceSymbol().has_value()) {
    bspPorts = std::make_unique<Block>();
    b.setInsertionPointToStart(bspPorts.get());
  }

  SmallVector<Attribute, 8> clients;
  for (auto req : implReqOp.getOps<ServiceReqOpInterface>()) {
    SmallVector<NamedAttribute, 4> clientAttrs;
    Attribute clientNamePath = req.getClientNamePath();
    Attribute servicePort = req.getServicePort();
    if (req.getToServerType())
      clientAttrs.push_back(b.getNamedAttr(
          "to_server_type", TypeAttr::get(req.getToServerType())));
    if (req.getToClient())
      clientAttrs.push_back(b.getNamedAttr(
          "to_client_type", TypeAttr::get(req.getToClientType())));

    clientAttrs.push_back(b.getNamedAttr("port", servicePort));
    clientAttrs.push_back(b.getNamedAttr("client_name", clientNamePath));

    clients.push_back(b.getDictionaryAttr(clientAttrs));

    if (!bspPorts)
      continue;

    llvm::TypeSwitch<Operation *>(req)
        .Case([&](RequestInOutChannelOp) {
          assert(req.getToClientType());
          assert(req.getToServerType());
          b.create<ServiceDeclInOutOp>(req.getServicePort().getName(),
                                       TypeAttr::get(req.getToServerType()),
                                       TypeAttr::get(req.getToClientType()));
        })
        .Case([&](RequestToClientConnectionOp) {
          b.create<ToClientOp>(req.getServicePort().getName(),
                               TypeAttr::get(req.getToClientType()));
        })
        .Case([&](RequestToServerConnectionOp) {
          b.create<ToServerOp>(req.getServicePort().getName(),
                               TypeAttr::get(req.getToServerType()));
        })
        .Default([](Operation *) {});
  }

  if (bspPorts && !bspPorts->empty()) {
    b.setInsertionPointToEnd(
        implReqOp->getParentOfType<mlir::ModuleOp>().getBody());
    // TODO: we currently only support one BSP. Should we support more?
    auto decl = b.create<CustomServiceDeclOp>("BSP");
    decl.getPorts().push_back(bspPorts.release());
    implReqOp.setServiceSymbol(decl.getSymNameAttr().getValue());
  }

  auto clientsAttr = b.getArrayAttr(clients);
  auto nameAttr = b.getArrayAttr(ArrayRef<Attribute>{});
  b.setInsertionPointAfter(implReqOp);
  b.create<ServiceHierarchyMetadataOp>(
      implReqOp.getServiceSymbolAttr(), nameAttr, implReqOp.getImplTypeAttr(),
      implReqOp.getImplOptsAttr(), clientsAttr);
}

LogicalResult ESIConnectServicesPass::replaceInst(ServiceInstanceOp instOp,
                                                  Block *portReqs) {
  assert(portReqs);
  auto declSym = instOp.getServiceSymbolAttr();
  ServiceDeclOpInterface decl;
  if (declSym) {
    decl = dyn_cast_or_null<ServiceDeclOpInterface>(
        topLevelSyms.getDefinition(declSym));
    if (!decl)
      return instOp.emitOpError("Could not find service declaration ")
             << declSym;
  }

  // Compute the result types for the new op -- the instance op's output types
  // + the to_client types.
  SmallVector<Type, 8> resultTypes(instOp.getResultTypes().begin(),
                                   instOp.getResultTypes().end());
  for (auto req : portReqs->getOps<ServiceReqOpInterface>())
    if (auto t = req.getToClientType())
      resultTypes.push_back(t);

  // Create the generation request op.
  OpBuilder b(instOp);
  auto implOp = b.create<ServiceImplementReqOp>(
      instOp.getLoc(), resultTypes, instOp.getServiceSymbolAttr(),
      instOp.getImplTypeAttr(), instOp.getImplOptsAttr(), instOp.getOperands());
  implOp->setDialectAttrs(instOp->getDialectAttrs());
  implOp.getPortReqs().push_back(portReqs);

  // Update the users.
  for (auto [n, o] : llvm::zip(implOp.getResults(), instOp.getResults()))
    o.replaceAllUsesWith(n);
  unsigned instOpNumResults = instOp.getNumResults();
  for (auto [idx, req] : llvm::enumerate(
           llvm::make_filter_range(portReqs->getOps<ServiceReqOpInterface>(),
                                   [](ServiceReqOpInterface req) -> bool {
                                     return req.getToClient() != nullptr;
                                   }))) {
    req.getToClient().replaceAllUsesWith(
        implOp.getResult(idx + instOpNumResults));
  }

  emitServiceMetadata(implOp);

  // Try to generate the service provider.
  if (failed(genDispatcher.generate(implOp, decl)))
    return instOp.emitOpError("failed to generate server");

  instOp.erase();
  return success();
}

LogicalResult
ESIConnectServicesPass::surfaceReqs(hw::HWMutableModuleLike mod,
                                    ArrayRef<ServiceReqOpInterface> reqs) {
  auto *ctxt = mod.getContext();
  Block *body = &mod->getRegion(0).front();

  // Track initial operand/result counts and the new IO.
  unsigned origNumInputs = mod.getNumInputs();
  SmallVector<std::pair<unsigned, hw::PortInfo>> newInputs;
  unsigned origNumOutputs = mod.getNumOutputs();
  SmallVector<std::pair<mlir::StringAttr, Value>> newOutputs;

  // Assemble a port name from an array.
  auto getPortName = [&](ArrayAttr namePath) {
    std::string portName;
    llvm::raw_string_ostream nameOS(portName);
    llvm::interleave(
        namePath.getValue(), nameOS,
        [&](Attribute attr) { nameOS << attr.cast<StringAttr>().getValue(); },
        ".");
    return StringAttr::get(ctxt, nameOS.str());
  };

  // Insert new module input ESI ports.
  for (auto req : reqs) {
    Type toClientType = req.getToClientType();
    if (!toClientType)
      continue;
    newInputs.push_back(std::make_pair(
        origNumInputs,
        hw::PortInfo{{getPortName(req.getClientNamePath()), toClientType,
                      hw::ModulePort::Direction::Input},
                     origNumInputs,
                     {},
                     {},
                     req->getLoc()}));

    // Replace uses with new block args which will correspond to said ports.
    Value replValue = body->addArgument(toClientType, req->getLoc());
    req.getToClient().replaceAllUsesWith(replValue);
  }
  mod.insertPorts(newInputs, {});

  // Append output ports to new port list and redirect toServer inputs to
  // output op.
  unsigned outputCounter = origNumOutputs;
  for (auto req : reqs) {
    Value toServer = req.getToServer();
    if (!toServer)
      continue;
    newOutputs.push_back({getPortName(req.getClientNamePath()), toServer});
  }

  mod.appendOutputs(newOutputs);

  // Prepend a name to the instance tracking array.
  auto prependNamePart = [&](ArrayAttr namePath, StringRef part) {
    SmallVector<Attribute, 8> newNamePath;
    newNamePath.push_back(StringAttr::get(namePath.getContext(), part));
    newNamePath.append(namePath.begin(), namePath.end());
    return ArrayAttr::get(namePath.getContext(), newNamePath);
  };

  // Update the module instantiations.
  SmallVector<hw::HWInstanceLike, 1> newModuleInstantiations;
  StringAttr argsAttrName = StringAttr::get(ctxt, "argNames");
  StringAttr resultsAttrName = StringAttr::get(ctxt, "resultNames");
  for (auto inst : moduleInstantiations[mod]) {
    OpBuilder b(inst);

    // Assemble lists for the new instance op. Seed it with the existing
    // values.
    SmallVector<Value, 16> newOperands(inst->getOperands().begin(),
                                       inst->getOperands().end());
    SmallVector<Type, 16> newResultTypes(inst->getResultTypes().begin(),
                                         inst->getResultTypes().end());

    // Add new inputs for the new to_client requests and clone the request
    // into the module containing `inst`.
    circt::BackedgeBuilder beb(b, mod.getLoc());
    SmallVector<circt::Backedge, 8> newResultBackedges;
    for (auto req : reqs) {
      auto clone = cast<ServiceReqOpInterface>(b.clone(*req));
      clone.setClientNamePath(
          prependNamePart(clone.getClientNamePath(), inst.getInstanceName()));
      if (Value toClient = clone.getToClient())
        newOperands.push_back(toClient);
      if (Type toServerType = clone.getToServerType()) {
        newResultTypes.push_back(toServerType);
        Backedge result = beb.get(toServerType);
        newResultBackedges.push_back(result);
        clone.setToServer(result);
      }
    }

    // Create a replacement instance of the same operation type.
    SmallVector<NamedAttribute> newAttrs;
    for (auto attr : inst->getAttrs()) {
      if (attr.getName() == argsAttrName)
        newAttrs.push_back(b.getNamedAttr(argsAttrName, mod.getInputNames()));
      else if (attr.getName() == resultsAttrName)
        newAttrs.push_back(
            b.getNamedAttr(resultsAttrName, mod.getOutputNames()));
      else
        newAttrs.push_back(attr);
    }
    auto *newInst = b.insert(Operation::create(
        inst->getLoc(), inst->getName(), newResultTypes, newOperands,
        b.getDictionaryAttr(newAttrs), inst->getPropertiesStorage(),
        inst->getSuccessors(), inst->getRegions()));
    newModuleInstantiations.push_back(cast<hw::HWInstanceLike>(newInst));

    // Replace all uses of the instance being replaced.
    for (auto [newV, oldV] :
         llvm::zip(newInst->getResults(), inst->getResults()))
      oldV.replaceAllUsesWith(newV);

    // Clone the to_server requests and wire them up to the new instance.
    outputCounter = origNumOutputs;
    for (Backedge newResult : newResultBackedges)
      newResult.setValue(newInst->getResult(outputCounter++));
  }

  // Replace the list of instantiations and erase the old ones.
  moduleInstantiations[mod].swap(newModuleInstantiations);
  for (auto oldInst : newModuleInstantiations)
    oldInst->erase();

  // Erase the original requests since they have been cloned into the proper
  // destination modules.
  for (auto req : reqs)
    req.erase();
  return success();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESIConnectServicesPass() {
  return std::make_unique<ESIConnectServicesPass>();
}
