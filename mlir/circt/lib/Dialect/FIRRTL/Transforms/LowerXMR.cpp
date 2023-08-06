//===- LowerXMR.cpp - FIRRTL Lower to XMR -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements FIRRTL XMR Lowering.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-lower-xmr"

using namespace circt;
using namespace firrtl;
using hw::InnerRefAttr;

/// The LowerXMRPass will replace every RefResolveOp with an XMR encoded within
/// a verbatim expr op. This also removes every RefType port from the modules
/// and corresponding instances. This is a dataflow analysis over a very
/// constrained RefType. Domain of the dataflow analysis is the set of all
/// RefSendOps. It computes an interprocedural reaching definitions (of
/// RefSendOp) analysis. Essentially every RefType value must be mapped to one
/// and only one RefSendOp. The analysis propagates the dataflow from every
/// RefSendOp to every value of RefType across modules. The RefResolveOp is the
/// final leaf into which the dataflow must reach.
///
/// Since there can be multiple readers, multiple RefResolveOps can be reachable
/// from a single RefSendOp. To support multiply instantiated modules and
/// multiple readers, it is essential to track the path to the RefSendOp, other
/// than just the RefSendOp. For example, if there exists a wire `xmr_wire` in
/// module `Foo`, the algorithm needs to support generating Top.Bar.Foo.xmr_wire
/// and Top.Foo.xmr_wire and Top.Zoo.Foo.xmr_wire for different instance paths
/// that exist in the circuit.

namespace {
struct XMRNode {
  using NextNodeOnPath = std::optional<size_t>;
  using SymOrIndexOp = PointerUnion<Attribute, Operation *>;
  SymOrIndexOp info;
  NextNodeOnPath next;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const XMRNode &node) {
  os << "node(";
  if (auto attr = dyn_cast<Attribute>(node.info))
    os << "path=" << attr;
  else {
    auto subOp = cast<RefSubOp>(cast<Operation *>(node.info));
    os << "index=" << subOp.getIndex() << " (-> " << subOp.getType() << ")";
  }
  os << ", next=" << node.next << ")";
  return os;
}
} // end anonymous namespace

class LowerXMRPass : public LowerXMRBase<LowerXMRPass> {

  void runOnOperation() override {

    // Populate a CircuitNamespace that can be used to generate unique
    // circuit-level symbols.
    CircuitNamespace ns(getOperation());
    circuitNamespace = &ns;

    llvm::EquivalenceClasses<Value, ValueComparator> eq;
    dataFlowClasses = &eq;

    InstanceGraph &instanceGraph = getAnalysis<InstanceGraph>();
    SmallVector<RefResolveOp> resolveOps;
    SmallVector<RefSubOp> indexingOps;
    SmallVector<Operation *> forceAndReleaseOps;
    // The dataflow function, that propagates the reachable RefSendOp across
    // RefType Ops.
    auto transferFunc = [&](Operation &op) -> LogicalResult {
      return TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<RefSendOp>([&](RefSendOp send) {
            // Get a reference to the actual signal to which the XMR will be
            // generated.
            Value xmrDef = send.getBase();
            if (isZeroWidth(send.getType().getType())) {
              markForRemoval(send);
              return success();
            }

            if (auto verbExpr = xmrDef.getDefiningOp<VerbatimExprOp>())
              if (verbExpr.getSymbolsAttr().empty() && verbExpr->hasOneUse()) {
                // This represents the internal path into a module. For
                // generating the correct XMR, no node can be created in this
                // module. Create a null InnerRef and ensure the hierarchical
                // path ends at the parent that instantiates this module.
                auto inRef = InnerRefAttr();
                auto ind = addReachingSendsEntry(send.getResult(), inRef);
                xmrPathSuffix[ind] = verbExpr.getText();
                markForRemoval(verbExpr);
                markForRemoval(send);
                return success();
              }
            // Get an InnerRefAttr to the value being sent.

            // Add a node, don't need to have symbol on defining operation,
            // just a way to send out the value.
            ImplicitLocOpBuilder b(xmrDef.getLoc(), &getContext());
            b.setInsertionPointAfterValue(xmrDef);
            SmallString<32> opName;
            auto nameKind = NameKindEnum::DroppableName;

            if (auto [name, rootKnown] = getFieldName(
                    getFieldRefFromValue(xmrDef), /*nameSafe=*/true);
                rootKnown) {
              opName = name + "_probe";
              nameKind = NameKindEnum::InterestingName;
            } else if (auto *xmrDefOp = xmrDef.getDefiningOp()) {
              // Inspect "name" directly for ops that aren't named by above.
              // (e.g., firrtl.constant)
              if (auto name = xmrDefOp->getAttrOfType<StringAttr>("name")) {
                (Twine(name.strref()) + "_probe").toVector(opName);
                nameKind = NameKindEnum::InterestingName;
              }
            }
            xmrDef = b.create<NodeOp>(xmrDef, opName, nameKind).getResult();

            // Create a new entry for this RefSendOp. The path is currently
            // local.
            addReachingSendsEntry(send.getResult(), getInnerRefTo(xmrDef));
            markForRemoval(send);
            return success();
          })
          .Case<RWProbeOp>([&](RWProbeOp rwprobe) {
            if (!isZeroWidth(rwprobe.getType()))
              addReachingSendsEntry(rwprobe.getResult(), rwprobe.getTarget());
            markForRemoval(rwprobe);
            return success();
          })
          .Case<MemOp>([&](MemOp mem) {
            // MemOp can produce debug ports of RefType. Each debug port
            // represents the RefType for the corresponding register of the
            // memory. Since the memory is not yet generated the register name
            // is assumed to be "Memory". Note that MemOp creates RefType
            // without a RefSend.
            for (const auto &res : llvm::enumerate(mem.getResults()))
              if (isa<RefType>(mem.getResult(res.index()).getType())) {
                auto inRef = getInnerRefTo(mem);
                auto ind = addReachingSendsEntry(res.value(), inRef);
                xmrPathSuffix[ind] = "Memory";
                // Just node that all the debug ports of memory must be removed.
                // So this does not record the port index.
                refPortsToRemoveMap[mem].resize(1);
              }
            return success();
          })
          .Case<InstanceOp>(
              [&](auto inst) { return handleInstanceOp(inst, instanceGraph); })
          .Case<FConnectLike>([&](FConnectLike connect) {
            // Ignore BaseType.
            if (!isa<RefType>(connect.getSrc().getType()))
              return success();
            markForRemoval(connect);
            if (isZeroWidth(
                    type_cast<RefType>(connect.getSrc().getType()).getType()))
              return success();
            // Merge the dataflow classes of destination into the source of the
            // Connect. This handles two cases:
            // 1. If the dataflow at the source is known, then the
            // destination is also inferred. By merging the dataflow class of
            // destination with source, every value reachable from the
            // destination automatically infers a reaching RefSend.
            // 2. If dataflow at source is unkown, then just record that both
            // source and destination will have the same dataflow information.
            // Later in the pass when the reaching RefSend is inferred at the
            // leader of the dataflowClass, then we automatically infer the
            // dataflow at this connect and every value reachable from the
            // destination.
            dataFlowClasses->unionSets(connect.getSrc(), connect.getDest());
            return success();
          })
          .Case<RefSubOp>([&](RefSubOp op) -> LogicalResult {
            markForRemoval(op);
            if (isZeroWidth(op.getType().getType()))
              return success();

            // Enqueue for processing after visiting other operations.
            indexingOps.push_back(op);
            return success();
          })
          .Case<RefResolveOp>([&](RefResolveOp resolve) {
            // Merge dataflow, under the same conditions as above for Connect.
            // 1. If dataflow at the resolve.getRef is known, propagate that to
            // the result. This is true for downward scoped XMRs, that is,
            // RefSendOp must be visited before the corresponding RefResolveOp
            // is visited.
            // 2. Else, just record that both result and ref should have the
            // same reaching RefSend. This condition is true for upward scoped
            // XMRs. That is, RefResolveOp can be visited before the
            // corresponding RefSendOp is recorded.

            markForRemoval(resolve);
            if (!isZeroWidth(resolve.getType()))
              dataFlowClasses->unionSets(resolve.getRef(), resolve.getResult());
            resolveOps.push_back(resolve);
            return success();
          })
          .Case<RefCastOp>([&](RefCastOp op) {
            markForRemoval(op);
            if (!isZeroWidth(op.getType().getType()))
              dataFlowClasses->unionSets(op.getInput(), op.getResult());
            return success();
          })
          .Case<Forceable>([&](Forceable op) {
            if (!op.isForceable() || op.getDataRef().use_empty() ||
                isZeroWidth(op.getDataType()))
              return success();

            addReachingSendsEntry(op.getDataRef(), getInnerRefTo(op));
            return success();
          })
          .Case<RefForceOp, RefForceInitialOp, RefReleaseOp,
                RefReleaseInitialOp>([&](auto op) {
            forceAndReleaseOps.push_back(op);
            return success();
          })
          .Default([&](auto) { return success(); });
    };

    SmallVector<FModuleOp> publicModules;

    // Traverse the modules in post order.
    for (auto node : llvm::post_order(&instanceGraph)) {
      auto module = dyn_cast<FModuleOp>(*node->getModule());
      if (!module)
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "Traversing module:" << module.getModuleNameAttr() << "\n");

      if (module.isPublic())
        publicModules.push_back(module);

      for (Operation &op : module.getBodyBlock()->getOperations())
        if (transferFunc(op).failed())
          return signalPassFailure();

      // Since we walk operations pre-order and not along dataflow edges,
      // ref.sub may not be resolvable when we encounter them (they're not just
      // unification). This can happen when refs go through an output port or
      // input instance result and back into the design. Handle these by walking
      // them, resolving what we can, until all are handled or nothing can be
      // resolved.
      while (!indexingOps.empty()) {
        // Grab the set of unresolved ref.sub's.
        decltype(indexingOps) worklist;
        worklist.swap(indexingOps);

        for (auto op : worklist) {
          auto inputEntry =
              getRemoteRefSend(op.getInput(), /*errorIfNotFound=*/false);
          // If we can't resolve, add back and move on.
          if (!inputEntry)
            indexingOps.push_back(op);
          else
            addReachingSendsEntry(op.getResult(), op.getOperation(),
                                  inputEntry);
        }
        // If nothing was resolved, give up.
        if (worklist.size() == indexingOps.size()) {
          auto op = worklist.front();
          getRemoteRefSend(op.getInput());
          op.emitError(
                "indexing through probe of unknown origin (input probe?)")
              .attachNote(op.getInput().getLoc())
              .append("indexing through this reference");
          return signalPassFailure();
        }
      }

      // Record all the RefType ports to be removed later.
      size_t numPorts = module.getNumPorts();
      for (size_t portNum = 0; portNum < numPorts; ++portNum)
        if (isa<RefType>(module.getPortType(portNum))) {
          setPortToRemove(module, portNum, numPorts);
        }
    }

    LLVM_DEBUG({
      for (auto I = dataFlowClasses->begin(), E = dataFlowClasses->end();
           I != E; ++I) { // Iterate over all of the equivalence sets.
        if (!I->isLeader())
          continue; // Ignore non-leader sets.
        // Print members in this set.
        llvm::interleave(llvm::make_range(dataFlowClasses->member_begin(I),
                                          dataFlowClasses->member_end()),
                         llvm::dbgs(), "\n");
        llvm::dbgs() << "\n dataflow at leader::" << I->getData() << "\n =>";
        auto iter = dataflowAt.find(I->getData());
        if (iter != dataflowAt.end()) {
          for (auto init = refSendPathList[iter->getSecond()]; init.next;
               init = refSendPathList[*init.next])
            llvm::dbgs() << "\n " << init;
        }
        llvm::dbgs() << "\n Done\n"; // Finish set.
      }
    });
    for (auto refResolve : resolveOps)
      if (handleRefResolve(refResolve).failed())
        return signalPassFailure();
    for (auto *op : forceAndReleaseOps)
      if (failed(handleForceReleaseOp(op)))
        return signalPassFailure();
    for (auto module : publicModules) {
      if (failed(handlePublicModuleRefPorts(module)))
        return signalPassFailure();
    }
    garbageCollect();

    // Clean up
    moduleNamespaces.clear();
    visitedModules.clear();
    dataflowAt.clear();
    refSendPathList.clear();
    dataFlowClasses = nullptr;
    refPortsToRemoveMap.clear();
    opsToRemove.clear();
    xmrPathSuffix.clear();
    circuitNamespace = nullptr;
    pathCache.clear();
    pathInsertPoint = {};
  }

  /// Generate the ABI ref_<circuit>_<module> prefix string into `prefix`.
  void getRefABIPrefix(FModuleLike mod, SmallVectorImpl<char> &prefix) {
    (Twine("ref_") +
     (isa<FExtModuleOp>(mod) ? mod.getModuleName() : getOperation().getName()) +
     "_" + mod.getModuleName())
        .toVector(prefix);
  }

  /// Get full macro name as StringAttr for the specified ref port.
  /// Uses existing 'prefix', optionally preprends the backtick character.
  StringAttr getRefABIMacroForPort(FModuleLike mod, size_t portIndex,
                                   const Twine &prefix, bool backTick = false) {
    return StringAttr::get(&getContext(), Twine(backTick ? "`" : "") + prefix +
                                              "_" + mod.getPortName(portIndex));
  }

  LogicalResult resolveReferencePath(mlir::TypedValue<RefType> refVal,
                                     ImplicitLocOpBuilder builder,
                                     mlir::FlatSymbolRefAttr &ref,
                                     SmallString<128> &stringLeaf) {
    assert(stringLeaf.empty());

    auto remoteOpPath = getRemoteRefSend(refVal);
    if (!remoteOpPath)
      return failure();
    SmallVector<Attribute> refSendPath;
    SmallVector<RefSubOp> indexing;
    size_t lastIndex;
    while (remoteOpPath) {
      lastIndex = *remoteOpPath;
      auto entr = refSendPathList[*remoteOpPath];
      TypeSwitch<XMRNode::SymOrIndexOp>(entr.info)
          .Case<Attribute>([&](auto attr) {
            // If the path is a singular verbatim expression, the attribute of
            // the send path list entry will be null.
            if (attr)
              refSendPath.push_back(attr);
          })
          .Case<Operation *>(
              [&](auto *op) { indexing.push_back(cast<RefSubOp>(op)); });
      remoteOpPath = entr.next;
    }
    auto iter = xmrPathSuffix.find(lastIndex);

    // If this xmr has a suffix string (internal path into a module, that is not
    // yet generated).
    if (iter != xmrPathSuffix.end()) {
      if (!refSendPath.empty())
        stringLeaf.append(".");
      stringLeaf.append(iter->getSecond());
    }

    assert(!(refSendPath.empty() && stringLeaf.empty()) &&
           "nothing to index through");

    // All indexing done as the ref is plumbed around indexes through
    // the target/referent, not the current point of the path which
    // describes how to access the referent we're indexing through.
    // Above we gathered all indexing operations, so now append them
    // to the path (after any relevant `xmrPathSuffix`) to reach
    // the target element.
    // Generating these strings here (especially if ref is sent
    // out from a different design) is fragile but should get this
    // working well enough while sorting out how to do this better.
    // Some discussion of this can be found here:
    // https://github.com/llvm/circt/pull/5551#discussion_r1258908834
    for (auto subOp : llvm::reverse(indexing)) {
      TypeSwitch<FIRRTLBaseType>(subOp.getInput().getType().getType())
          .Case<FVectorType, OpenVectorType>([&](auto vecType) {
            (Twine("[") + Twine(subOp.getIndex()) + "]").toVector(stringLeaf);
          })
          .Case<BundleType, OpenBundleType>([&](auto bundleType) {
            auto fieldName = bundleType.getElementName(subOp.getIndex());
            stringLeaf.append({".", fieldName});
          });
    }

    if (!refSendPath.empty())
      // Compute the HierPathOp that stores the path.
      ref = FlatSymbolRefAttr::get(
          getOrCreatePath(builder.getArrayAttr(refSendPath), builder)
              .getSymNameAttr());

    return success();
  }

  LogicalResult resolveReference(mlir::TypedValue<RefType> refVal,
                                 Type desiredType, Location loc,
                                 Operation *insertBefore, Value &out) {
    auto remoteOpPath = getRemoteRefSend(refVal);
    if (!remoteOpPath)
      return failure();

    ImplicitLocOpBuilder builder(loc, insertBefore);
    mlir::FlatSymbolRefAttr ref;
    SmallString<128> xmrString;
    if (failed(resolveReferencePath(refVal, builder, ref, xmrString)))
      return failure();

    // Create the XMR op and convert it to the referenced FIRRTL type.
    auto referentType = refVal.getType().getType();
    Value xmrResult;
    auto xmrType = sv::InOutType::get(lowerType(referentType));
    xmrResult = builder
                    .create<sv::XMRRefOp>(
                        xmrType, ref,
                        xmrString.empty() ? StringAttr{}
                                          : builder.getStringAttr(xmrString))
                    .getResult();
    out =
        builder.create<mlir::UnrealizedConversionCastOp>(desiredType, xmrResult)
            .getResult(0);
    return success();
  }

  // Replace the Force/Release's ref argument with a resolved XMRRef.
  LogicalResult handleForceReleaseOp(Operation *op) {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<RefForceOp, RefForceInitialOp, RefReleaseOp, RefReleaseInitialOp>(
            [&](auto op) {
              // Drop if zero-width target.
              if (isZeroWidth(op.getDest().getType().getType())) {
                op.erase();
                return success();
              }
              Value ref;
              if (failed(resolveReference(op.getDest(), op.getDest().getType(),
                                          op.getLoc(), op, ref)))
                return failure();
              op.getDestMutable().assign(ref);
              return success();
            })
        .Default([](auto *op) {
          return op->emitError("unexpected operation kind");
        });
  }

  // Replace the RefResolveOp with verbatim op representing the XMR.
  LogicalResult handleRefResolve(RefResolveOp resolve) {
    auto resWidth = getBitWidth(resolve.getType());
    if (resWidth.has_value() && *resWidth == 0) {
      // Donot emit 0 width XMRs, replace it with constant 0.
      ImplicitLocOpBuilder builder(resolve.getLoc(), resolve);
      auto zeroUintType = UIntType::get(builder.getContext(), 0);
      auto zeroC = builder.createOrFold<BitCastOp>(
          resolve.getType(), builder.create<ConstantOp>(
                                 zeroUintType, getIntZerosAttr(zeroUintType)));
      resolve.getResult().replaceAllUsesWith(zeroC);
      return success();
    }
    Value result;
    if (failed(resolveReference(resolve.getRef(), resolve.getType(),
                                resolve.getLoc(), resolve, result)))
      return failure();
    resolve.getResult().replaceAllUsesWith(result);
    return success();
  }

  void setPortToRemove(Operation *op, size_t index, size_t numPorts) {
    if (refPortsToRemoveMap[op].size() < numPorts)
      refPortsToRemoveMap[op].resize(numPorts);
    refPortsToRemoveMap[op].set(index);
  }

  // Propagate the reachable RefSendOp across modules.
  LogicalResult handleInstanceOp(InstanceOp inst,
                                 InstanceGraph &instanceGraph) {
    Operation *mod = instanceGraph.getReferencedModule(inst);
    if (auto extRefMod = dyn_cast<FExtModuleOp>(mod)) {
      // Extern modules can generate RefType ports, they have an attached
      // attribute which specifies the internal path into the extern module.
      // This string attribute will be used to generate the final xmr.
      auto internalPaths = extRefMod.getInternalPaths();
      size_t pathsIndex = 0;
      auto numPorts = inst.getNumResults();
      SmallString<128> circuitRefPrefix;

      /// Get the resolution string for this ref-type port.
      auto getPath = [&](size_t portNo) {
        // If there's an internalPaths array, grab the next element.
        if (!internalPaths.empty())
          return cast<StringAttr>(internalPaths[pathsIndex++]);

        // Otherwise, we're using the ref ABI.  Generate the prefix string
        // and return the macro for the specified port.
        if (circuitRefPrefix.empty())
          getRefABIPrefix(extRefMod, circuitRefPrefix);

        return getRefABIMacroForPort(extRefMod, portNo, circuitRefPrefix, true);
      };

      for (const auto &res : llvm::enumerate(inst.getResults())) {
        if (!isa<RefType>(inst.getResult(res.index()).getType()))
          continue;

        auto inRef = getInnerRefTo(inst);
        auto ind = addReachingSendsEntry(res.value(), inRef);

        xmrPathSuffix[ind] = getPath(res.index());
        // The instance result and module port must be marked for removal.
        setPortToRemove(inst, res.index(), numPorts);
        setPortToRemove(extRefMod, res.index(), numPorts);
      }
      return success();
    }
    auto refMod = dyn_cast<FModuleOp>(mod);
    bool multiplyInstantiated = !visitedModules.insert(refMod).second;
    for (size_t portNum = 0, numPorts = inst.getNumResults();
         portNum < numPorts; ++portNum) {
      auto instanceResult = inst.getResult(portNum);
      if (!isa<RefType>(instanceResult.getType()))
        continue;
      if (!refMod)
        return inst.emitOpError("cannot lower ext modules with RefType ports");
      // Reference ports must be removed.
      setPortToRemove(inst, portNum, numPorts);
      // Drop the dead-instance-ports.
      if (instanceResult.use_empty() ||
          isZeroWidth(type_cast<RefType>(instanceResult.getType()).getType()))
        continue;
      auto refModuleArg = refMod.getArgument(portNum);
      if (inst.getPortDirection(portNum) == Direction::Out) {
        // For output instance ports, the dataflow is into this module.
        // Get the remote RefSendOp, that flows through the module ports.
        // If dataflow at remote module argument does not exist, error out.
        auto remoteOpPath = getRemoteRefSend(refModuleArg);
        if (!remoteOpPath)
          return failure();
        // Get the path to reaching refSend at the referenced module argument.
        // Now append this instance to the path to the reaching refSend.
        addReachingSendsEntry(instanceResult, getInnerRefTo(inst),
                              remoteOpPath);
      } else {
        // For input instance ports, the dataflow is into the referenced module.
        // Input RefType port implies, generating an upward scoped XMR.
        // No need to add the instance context, since downward reference must be
        // through single instantiated modules.
        if (multiplyInstantiated)
          return refMod.emitOpError(
                     "multiply instantiated module with input RefType port '")
                 << refMod.getPortName(portNum) << "'";
        dataFlowClasses->unionSets(
            dataFlowClasses->getOrInsertLeaderValue(refModuleArg),
            dataFlowClasses->getOrInsertLeaderValue(instanceResult));
      }
    }
    return success();
  }

  LogicalResult handlePublicModuleRefPorts(FModuleOp module) {
    auto builder = ImplicitLocOpBuilder::atBlockBegin(
        module.getLoc(), getOperation().getBodyBlock());

    SmallString<128> circuitRefPrefix;
    for (size_t portIndex = 0, numPorts = module.getNumPorts();
         portIndex != numPorts; ++portIndex) {
      auto refType = type_dyn_cast<RefType>(module.getPortType(portIndex));
      if (!refType || isZeroWidth(refType.getType()) ||
          module.getPortDirection(portIndex) != Direction::Out)
        continue;
      auto portValue =
          module.getArgument(portIndex).cast<mlir::TypedValue<RefType>>();

      mlir::FlatSymbolRefAttr ref;
      SmallString<128> stringLeaf;
      if (failed(resolveReferencePath(portValue, builder, ref, stringLeaf)))
        return failure();

      SmallString<128> formatString;
      if (ref)
        formatString += "{{0}}";
      formatString += stringLeaf;

      // Insert a macro with the format:
      // ref_<circuit-name>_<module-name>_<ref-name> <path>
      if (circuitRefPrefix.empty())
        getRefABIPrefix(module, circuitRefPrefix);
      auto macroName =
          getRefABIMacroForPort(module, portIndex, circuitRefPrefix);
      builder.create<sv::MacroDeclOp>(macroName, ArrayAttr(), StringAttr());

      auto macroDefOp = builder.create<sv::MacroDefOp>(
          FlatSymbolRefAttr::get(macroName),
          builder.getStringAttr(formatString),
          builder.getArrayAttr(ref ? ref : ArrayRef<Attribute>{}));

      // The macro will be exported to a file with the format:
      // ref_<circuit-name>_<module-name>.sv
      macroDefOp->setAttr("output_file",
                          hw::OutputFileAttr::getFromFilename(
                              &getContext(), circuitRefPrefix + ".sv"));
    }

    return success();
  }

  /// Get the cached namespace for a module.
  ModuleNamespace &getModuleNamespace(FModuleLike module) {
    auto it = moduleNamespaces.find(module);
    if (it != moduleNamespaces.end())
      return it->second;
    return moduleNamespaces.insert({module, ModuleNamespace(module)})
        .first->second;
  }

  InnerRefAttr getInnerRefTo(Value val) {
    if (auto arg = dyn_cast<BlockArgument>(val))
      return ::getInnerRefTo(
          cast<FModuleLike>(arg.getParentBlock()->getParentOp()),
          arg.getArgNumber(), [&](FModuleLike mod) -> ModuleNamespace & {
            return getModuleNamespace(mod);
          });
    return getInnerRefTo(val.getDefiningOp());
  }

  InnerRefAttr getInnerRefTo(Operation *op) {
    return ::getInnerRefTo(op, [&](FModuleOp mod) -> ModuleNamespace & {
      return getModuleNamespace(mod);
    });
  }

  void markForRemoval(Operation *op) { opsToRemove.push_back(op); }

  std::optional<size_t> getRemoteRefSend(Value val,
                                         bool errorIfNotFound = true) {
    auto iter = dataflowAt.find(dataFlowClasses->getOrInsertLeaderValue(val));
    if (iter != dataflowAt.end())
      return iter->getSecond();
    if (!errorIfNotFound)
      return std::nullopt;
    // The referenced module must have already been analyzed, error out if the
    // dataflow at the child module is not resolved.
    if (BlockArgument arg = dyn_cast<BlockArgument>(val))
      arg.getOwner()->getParentOp()->emitError(
          "reference dataflow cannot be traced back to the remote read op "
          "for module port '")
          << dyn_cast<FModuleOp>(arg.getOwner()->getParentOp())
                 .getPortName(arg.getArgNumber())
          << "'";
    else
      val.getDefiningOp()->emitOpError(
          "reference dataflow cannot be traced back to the remote read op");
    signalPassFailure();
    return std::nullopt;
  }

  size_t
  addReachingSendsEntry(Value atRefVal, XMRNode::SymOrIndexOp info,
                        std::optional<size_t> continueFrom = std::nullopt) {
    auto leader = dataFlowClasses->getOrInsertLeaderValue(atRefVal);
    auto indx = refSendPathList.size();
    dataflowAt[leader] = indx;
    refSendPathList.push_back({info, continueFrom});
    return indx;
  }

  void garbageCollect() {
    // Now erase all the Ops and ports of RefType.
    // This needs to be done as the last step to ensure uses are erased before
    // the def is erased.
    for (Operation *op : llvm::reverse(opsToRemove))
      op->erase();
    for (auto iter : refPortsToRemoveMap)
      if (auto mod = dyn_cast<FModuleOp>(iter.getFirst()))
        mod.erasePorts(iter.getSecond());
      else if (auto mod = dyn_cast<FExtModuleOp>(iter.getFirst()))
        mod.erasePorts(iter.getSecond());
      else if (auto inst = dyn_cast<InstanceOp>(iter.getFirst())) {
        ImplicitLocOpBuilder b(inst.getLoc(), inst);
        inst.erasePorts(b, iter.getSecond());
        inst.erase();
      } else if (auto mem = dyn_cast<MemOp>(iter.getFirst())) {
        // Remove all debug ports of the memory.
        ImplicitLocOpBuilder builder(mem.getLoc(), mem);
        SmallVector<Attribute, 4> resultNames;
        SmallVector<Type, 4> resultTypes;
        SmallVector<Attribute, 4> portAnnotations;
        SmallVector<Value, 4> oldResults;
        for (const auto &res : llvm::enumerate(mem.getResults())) {
          if (isa<RefType>(mem.getResult(res.index()).getType()))
            continue;
          resultNames.push_back(mem.getPortName(res.index()));
          resultTypes.push_back(res.value().getType());
          portAnnotations.push_back(mem.getPortAnnotation(res.index()));
          oldResults.push_back(res.value());
        }
        auto newMem = builder.create<MemOp>(
            resultTypes, mem.getReadLatency(), mem.getWriteLatency(),
            mem.getDepth(), RUWAttr::Undefined,
            builder.getArrayAttr(resultNames), mem.getNameAttr(),
            mem.getNameKind(), mem.getAnnotations(),
            builder.getArrayAttr(portAnnotations), mem.getInnerSymAttr(),
            mem.getInitAttr(), mem.getPrefixAttr());
        for (const auto &res : llvm::enumerate(oldResults))
          res.value().replaceAllUsesWith(newMem.getResult(res.index()));
        mem.erase();
      }
    opsToRemove.clear();
    refPortsToRemoveMap.clear();
    dataflowAt.clear();
    refSendPathList.clear();
  }

  bool isZeroWidth(FIRRTLBaseType t) { return t.getBitWidthOrSentinel() == 0; }

  /// Return a HierPathOp for the provided pathArray.  This will either return
  /// an existing HierPathOp or it will create and return a new one.
  hw::HierPathOp getOrCreatePath(ArrayAttr pathArray,
                                 ImplicitLocOpBuilder &builder) {
    assert(pathArray && !pathArray.empty());
    // Return an existing HierPathOp if one exists with the same path.
    auto pathIter = pathCache.find(pathArray);
    if (pathIter != pathCache.end())
      return pathIter->second;

    // Reset the insertion point after this function returns.
    OpBuilder::InsertionGuard guard(builder);

    // Set the insertion point to either the known location where the pass
    // inserts HierPathOps or to the start of the circuit.
    if (pathInsertPoint.isSet())
      builder.restoreInsertionPoint(pathInsertPoint);
    else
      builder.setInsertionPointToStart(getOperation().getBodyBlock());

    // Create the new HierPathOp and insert it into the pathCache.
    hw::HierPathOp path =
        pathCache
            .insert({pathArray,
                     builder.create<hw::HierPathOp>(
                         circuitNamespace->newName("xmrPath"), pathArray)})
            .first->second;
    path.setVisibility(SymbolTable::Visibility::Private);

    // Save the insertion point so other unique HierPathOps will be created
    // after this one.
    pathInsertPoint = builder.saveInsertionPoint();

    // Return the new path.
    return path;
  }

private:
  /// Cached module namespaces.
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;

  DenseSet<Operation *> visitedModules;
  /// Map of a reference value to an entry into refSendPathList. Each entry in
  /// refSendPathList represents the path to RefSend.
  /// The path is required since there can be multiple paths to the RefSend and
  /// we need to identify a unique path.
  DenseMap<Value, size_t> dataflowAt;

  /// refSendPathList is used to construct a path to the RefSendOp. Each entry
  /// is an XMRNode, with an InnerRefAttr or indexing op, and a pointer to the
  /// next node in the path. The InnerRefAttr can be to an InstanceOp or to the
  /// XMR defining op, the index op records narrowing along path. All the nodes
  /// representing an InstanceOp or indexing operation must have a valid
  /// NextNodeOnPath. Only the node representing the final XMR defining op has
  /// no NextNodeOnPath, which denotes a leaf node on the path.
  SmallVector<XMRNode> refSendPathList;

  /// llvm::EquivalenceClasses wants comparable elements. This comparator uses
  /// uses pointer comparison on the Impl.
  struct ValueComparator {
    bool operator()(const Value &lhs, const Value &rhs) const {
      return lhs.getImpl() < rhs.getImpl();
    }
  };

  llvm::EquivalenceClasses<Value, ValueComparator> *dataFlowClasses;
  // Instance and module ref ports that needs to be removed.
  DenseMap<Operation *, llvm::BitVector> refPortsToRemoveMap;

  /// RefResolve, RefSend, and Connects involving them that will be removed.
  SmallVector<Operation *> opsToRemove;

  /// Record the internal path to an external module or a memory.
  DenseMap<size_t, SmallString<128>> xmrPathSuffix;

  CircuitNamespace *circuitNamespace;

  /// A cache of already created HierPathOps.  This is used to avoid repeatedly
  /// creating the same HierPathOp.
  DenseMap<Attribute, hw::HierPathOp> pathCache;

  /// The insertion point where the pass inserts HierPathOps.
  OpBuilder::InsertPoint pathInsertPoint = {};
};

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerXMRPass() {
  return std::make_unique<LowerXMRPass>();
}
