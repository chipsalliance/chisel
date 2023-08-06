//===- FIRRTLAnnotationHelper.cpp - FIRRTL Annotation Lookup ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helpers mapping annotations to operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-annos"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using llvm::StringRef;

// Some types have been expanded so the first layer of aggregate path is
// a return value.
static LogicalResult updateExpandedPort(StringRef field, AnnoTarget &ref) {
  if (auto mem = dyn_cast<MemOp>(ref.getOp()))
    for (size_t p = 0, pe = mem.getPortNames().size(); p < pe; ++p)
      if (mem.getPortNameStr(p) == field) {
        ref = PortAnnoTarget(mem, p);
        return success();
      }
  ref.getOp()->emitError("Cannot find port with name ") << field;
  return failure();
}

/// Try to resolve an non-array aggregate name from a target given the type and
/// operation of the resolved target.  This needs to deal with places where we
/// represent bundle returns as split into constituent parts.
static FailureOr<unsigned> findBundleElement(Operation *op, Type type,
                                             StringRef field) {
  auto bundle = type_dyn_cast<BundleType>(type);
  if (!bundle) {
    op->emitError("field access '")
        << field << "' into non-bundle type '" << type << "'";
    return failure();
  }
  auto idx = bundle.getElementIndex(field);
  if (!idx) {
    op->emitError("cannot resolve field '")
        << field << "' in subtype '" << bundle << "'";
    return failure();
  }
  return *idx;
}

/// Try to resolve an array index from a target given the type of the resolved
/// target.
static FailureOr<unsigned> findVectorElement(Operation *op, Type type,
                                             StringRef indexStr) {
  size_t index;
  if (indexStr.getAsInteger(10, index)) {
    op->emitError("Cannot convert '") << indexStr << "' to an integer";
    return failure();
  }
  auto vec = type_dyn_cast<FVectorType>(type);
  if (!vec) {
    op->emitError("index access '")
        << index << "' into non-vector type '" << type << "'";
    return failure();
  }
  return index;
}

static FailureOr<unsigned> findFieldID(AnnoTarget &ref,
                                       ArrayRef<TargetToken> tokens) {
  if (tokens.empty())
    return 0;

  auto *op = ref.getOp();
  auto fieldIdx = 0;
  // The first field for some ops refers to expanded return values.
  if (isa<MemOp>(ref.getOp())) {
    if (failed(updateExpandedPort(tokens.front().name, ref)))
      return {};
    tokens = tokens.drop_front();
  }

  auto type = ref.getType();
  for (auto token : tokens) {
    if (token.isIndex) {
      auto result = findVectorElement(op, type, token.name);
      if (failed(result))
        return failure();
      auto vector = type_cast<FVectorType>(type);
      type = vector.getElementType();
      fieldIdx += vector.getFieldID(*result);
    } else {
      auto result = findBundleElement(op, type, token.name);
      if (failed(result))
        return failure();
      auto bundle = type_cast<BundleType>(type);
      type = bundle.getElementType(*result);
      fieldIdx += bundle.getFieldID(*result);
    }
  }
  return fieldIdx;
}

void TokenAnnoTarget::toVector(SmallVectorImpl<char> &out) const {
  out.push_back('~');
  out.append(circuit.begin(), circuit.end());
  out.push_back('|');
  for (auto modInstPair : instances) {
    out.append(modInstPair.first.begin(), modInstPair.first.end());
    out.push_back('/');
    out.append(modInstPair.second.begin(), modInstPair.second.end());
    out.push_back(':');
  }
  out.append(module.begin(), module.end());
  if (name.empty())
    return;
  out.push_back('>');
  out.append(name.begin(), name.end());
  for (auto comp : component) {
    out.push_back(comp.isIndex ? '[' : '.');
    out.append(comp.name.begin(), comp.name.end());
    if (comp.isIndex)
      out.push_back(']');
  }
}

std::string firrtl::canonicalizeTarget(StringRef target) {

  if (target.empty())
    return target.str();

  // If this is a normal Target (not a Named), erase that field in the JSON
  // object and return that Target.
  if (target[0] == '~')
    return target.str();

  // This is a legacy target using the firrtl.annotations.Named type.  This
  // can be trivially canonicalized to a non-legacy target, so we do it with
  // the following three mappings:
  //   1. CircuitName => CircuitTarget, e.g., A -> ~A
  //   2. ModuleName => ModuleTarget, e.g., A.B -> ~A|B
  //   3. ComponentName => ReferenceTarget, e.g., A.B.C -> ~A|B>C
  std::string newTarget = ("~" + target).str();
  auto n = newTarget.find('.');
  if (n != std::string::npos)
    newTarget[n] = '|';
  n = newTarget.find('.');
  if (n != std::string::npos)
    newTarget[n] = '>';
  return newTarget;
}

std::optional<AnnoPathValue>
firrtl::resolveEntities(TokenAnnoTarget path, CircuitOp circuit,
                        SymbolTable &symTbl, CircuitTargetCache &cache) {
  // Validate circuit name.
  if (!path.circuit.empty() && circuit.getName() != path.circuit) {
    mlir::emitError(circuit.getLoc())
        << "circuit name doesn't match annotation '" << path.circuit << '\'';
    return {};
  }
  // Circuit only target.
  if (path.module.empty()) {
    assert(path.name.empty() && path.instances.empty() &&
           path.component.empty());
    return AnnoPathValue(circuit);
  }

  // Resolve all instances for non-local paths.
  SmallVector<InstanceOp> instances;
  for (auto p : path.instances) {
    auto mod = symTbl.lookup<FModuleOp>(p.first);
    if (!mod) {
      mlir::emitError(circuit.getLoc())
          << "module doesn't exist '" << p.first << '\'';
      return {};
    }
    auto resolved = cache.lookup(mod, p.second);
    if (!resolved || !isa<InstanceOp>(resolved.getOp())) {
      mlir::emitError(circuit.getLoc()) << "cannot find instance '" << p.second
                                        << "' in '" << mod.getName() << "'";
      return {};
    }
    instances.push_back(cast<InstanceOp>(resolved.getOp()));
  }
  // The final module is where the named target is (or is the named target).
  auto mod = symTbl.lookup<FModuleLike>(path.module);
  if (!mod) {
    mlir::emitError(circuit.getLoc())
        << "module doesn't exist '" << path.module << '\'';
    return {};
  }

  // ClassOps may not participate in annotation targeting. Neither the class
  // itself, nor any "named thing" defined under it, may be targeted by an anno.
  if (isa<ClassOp>(mod)) {
    mlir::emitError(mod.getLoc()) << "annotations cannot target classes";
    return {};
  }

  AnnoTarget ref;
  if (path.name.empty()) {
    assert(path.component.empty());
    ref = OpAnnoTarget(mod);
  } else {
    ref = cache.lookup(mod, path.name);
    if (!ref) {
      mlir::emitError(circuit.getLoc()) << "cannot find name '" << path.name
                                        << "' in " << mod.getModuleName();
      return {};
    }
    // AnnoTarget::getType() is not safe (CHIRRTL ops crash, null if instance),
    // avoid. For now, only references in ports can be targets, check that.
    // TODO: containsReference().
    if (ref.isa<PortAnnoTarget>() && isa<RefType>(ref.getType())) {
      mlir::emitError(circuit.getLoc())
          << "cannot target reference-type '" << path.name << "' in "
          << mod.getModuleName();
      return {};
    }
  }

  // If the reference is pointing to an instance op, we have to move the target
  // to the module.  This is done both because it is logical to have one
  // representation (this effectively canonicalizes a reference target on an
  // instance into an instance target) and because the SFC has a pass that does
  // this conversion.  E.g., this is converting (where "bar" is an instance):
  //   ~Foo|Foo>bar
  // Into:
  //   ~Foo|Foo/bar:Bar
  ArrayRef<TargetToken> component(path.component);
  if (auto instance = dyn_cast<InstanceOp>(ref.getOp())) {
    instances.push_back(instance);
    auto target = instance.getReferencedModule(symTbl);
    if (component.empty()) {
      ref = OpAnnoTarget(instance.getReferencedModule(symTbl));
    } else if (component.front().isIndex) {
      mlir::emitError(circuit.getLoc())
          << "illegal target '" << path.str() << "' indexes into an instance";
      return {};
    } else {
      auto field = component.front().name;
      ref = AnnoTarget();
      for (size_t p = 0, pe = getNumPorts(target); p < pe; ++p)
        if (target.getPortName(p) == field) {
          ref = PortAnnoTarget(target, p);
          break;
        }
      if (!ref) {
        mlir::emitError(circuit.getLoc())
            << "cannot find port '" << field << "' in module "
            << target.getModuleName();
        return {};
      }
      // TODO: containsReference().
      if (isa<RefType>(ref.getType())) {
        mlir::emitError(circuit.getLoc())
            << "annotation cannot target reference-type port '" << field
            << "' in module " << target.getModuleName();
        return {};
      }
      component = component.drop_front();
    }
  }

  // If we have aggregate specifiers, resolve those now. This call can update
  // the ref to target a port of a memory.
  auto result = findFieldID(ref, component);
  if (failed(result))
    return {};
  auto fieldIdx = *result;

  return AnnoPathValue(instances, ref, fieldIdx);
}

/// split a target string into it constituent parts.  This is the primary parser
/// for targets.
std::optional<TokenAnnoTarget> firrtl::tokenizePath(StringRef origTarget) {
  // An empty string is not a legal target.
  if (origTarget.empty())
    return {};
  StringRef target = origTarget;
  TokenAnnoTarget retval;
  std::tie(retval.circuit, target) = target.split('|');
  if (!retval.circuit.empty() && retval.circuit[0] == '~')
    retval.circuit = retval.circuit.drop_front();
  while (target.count(':')) {
    StringRef nla;
    std::tie(nla, target) = target.split(':');
    StringRef inst, mod;
    std::tie(mod, inst) = nla.split('/');
    retval.instances.emplace_back(mod, inst);
  }
  // remove aggregate
  auto targetBase =
      target.take_until([](char c) { return c == '.' || c == '['; });
  auto aggBase = target.drop_front(targetBase.size());
  std::tie(retval.module, retval.name) = targetBase.split('>');
  while (!aggBase.empty()) {
    if (aggBase[0] == '.') {
      aggBase = aggBase.drop_front();
      StringRef field = aggBase.take_front(aggBase.find_first_of("[."));
      aggBase = aggBase.drop_front(field.size());
      retval.component.push_back({field, false});
    } else if (aggBase[0] == '[') {
      aggBase = aggBase.drop_front();
      StringRef index = aggBase.take_front(aggBase.find_first_of(']'));
      aggBase = aggBase.drop_front(index.size() + 1);
      retval.component.push_back({index, true});
    } else {
      return {};
    }
  }

  return retval;
}

std::optional<AnnoPathValue> firrtl::resolvePath(StringRef rawPath,
                                                 CircuitOp circuit,
                                                 SymbolTable &symTbl,
                                                 CircuitTargetCache &cache) {
  auto pathStr = canonicalizeTarget(rawPath);
  StringRef path{pathStr};

  auto tokens = tokenizePath(path);
  if (!tokens) {
    mlir::emitError(circuit.getLoc())
        << "Cannot tokenize annotation path " << rawPath;
    return {};
  }

  return resolveEntities(*tokens, circuit, symTbl, cache);
}

InstanceOp firrtl::addPortsToModule(FModuleLike mod, InstanceOp instOnPath,
                                    FIRRTLType portType, Direction dir,
                                    StringRef newName,
                                    InstancePathCache &instancePathcache,
                                    CircuitTargetCache *targetCaches) {
  // To store the cloned version of `instOnPath`.
  InstanceOp clonedInstOnPath;
  // Get a new port name from the Namespace.
  auto portName = StringAttr::get(mod.getContext(), newName);
  // The port number for the new port.
  unsigned portNo = getNumPorts(mod);
  PortInfo portInfo = {portName, portType, dir, {}, mod.getLoc()};
  mod.insertPorts({{portNo, portInfo}});
  // Now update all the instances of `mod`.
  for (auto *use : instancePathcache.instanceGraph.lookup(mod)->uses()) {
    InstanceOp useInst = cast<InstanceOp>(use->getInstance());
    auto clonedInst = useInst.cloneAndInsertPorts({{portNo, portInfo}});
    if (useInst == instOnPath)
      clonedInstOnPath = clonedInst;
    // Update all occurences of old instance.
    instancePathcache.replaceInstance(useInst, clonedInst);
    if (targetCaches)
      targetCaches->replaceOp(useInst, clonedInst);
    useInst->replaceAllUsesWith(clonedInst.getResults().drop_back());
    useInst->erase();
  }
  return clonedInstOnPath;
}

//===----------------------------------------------------------------------===//
// AnnoTargetCache
//===----------------------------------------------------------------------===//

void AnnoTargetCache::gatherTargets(FModuleLike mod) {
  // Add ports
  for (const auto &p : llvm::enumerate(mod.getPorts()))
    insertPort(mod, p.index());

  // And named things
  mod.walk([&](Operation *op) { insertOp(op); });
}

//===----------------------------------------------------------------------===//
// Code related to handling Grand Central Data/Mem Taps annotations
//===----------------------------------------------------------------------===//

static Value lowerInternalPathAnno(AnnoPathValue &srcTarget,
                                   const AnnoPathValue &moduleTarget,
                                   const AnnoPathValue &target,
                                   StringAttr internalPathAttr,
                                   FIRRTLBaseType targetType,
                                   ApplyState &state) {
  Value sendVal;
  FModuleLike mod = cast<FModuleLike>(moduleTarget.ref.getOp());
  InstanceOp modInstance;
  if (!moduleTarget.instances.empty()) {
    modInstance = moduleTarget.instances.back();
  } else {
    auto *node = state.instancePathCache.instanceGraph.lookup(mod);
    if (!node->hasOneUse()) {
      mod->emitOpError(
          "cannot be used for DataTaps, it is instantiated multiple times");
      return nullptr;
    }
    modInstance = cast<InstanceOp>((*node->uses().begin())->getInstance());
  }
  ImplicitLocOpBuilder builder(modInstance.getLoc(), modInstance);
  builder.setInsertionPointAfter(modInstance);
  auto portRefType = RefType::get(targetType);
  SmallString<32> refName;
  for (auto c : internalPathAttr.getValue()) {
    switch (c) {
    case '.':
    case '[':
      refName.push_back('_');
      break;
    case ']':
      break;
    default:
      refName.push_back(c);
      break;
    }
  }

  // Add RefType ports corresponding to this "internalPath" to the external
  // module. This also updates all the instances of the external module.
  // This removes and replaces the instance, and returns the updated
  // instance.
  if (!state.wiringProblemInstRefs.contains(modInstance)) {
    modInstance =
        addPortsToModule(mod, modInstance, portRefType, Direction::Out, refName,
                         state.instancePathCache, &state.targetCaches);
  } else {
    // As a current limitation, mixing legacy Wiring and Data Taps is forbidden
    // to prevent invalidating Values used later
    mod->emitOpError(
        "cannot be used for both legacy Wiring and DataTaps simultaneously");
    return nullptr;
  }

  // Since the instance op generates the RefType output, no need of another
  // RefSendOp.  Store into an op to ensure we have stable reference,
  // so future tapping won't invalidate this Value.
  sendVal = modInstance.getResults().back();
  sendVal =
      builder
          .create<mlir::UnrealizedConversionCastOp>(sendVal.getType(), sendVal)
          ->getResult(0);

  // Now set the instance as the source for the final datatap xmr.
  srcTarget = AnnoPathValue(modInstance);
  if (auto extMod = dyn_cast<FExtModuleOp>((Operation *)mod)) {
    // The extern module can have other internal paths attached to it,
    // append this to them.
    SmallVector<Attribute> paths(extMod.getInternalPathsAttr().getValue());
    paths.push_back(internalPathAttr);
    extMod.setInternalPathsAttr(builder.getArrayAttr(paths));
  } else if (auto intMod = dyn_cast<FModuleOp>((Operation *)mod)) {
    auto builder = ImplicitLocOpBuilder::atBlockEnd(
        intMod.getLoc(), &intMod.getBody().getBlocks().back());
    auto pathStr = builder.create<VerbatimExprOp>(
        portRefType.getType(), internalPathAttr.getValue(), ValueRange{});
    auto sendPath = builder.create<RefSendOp>(pathStr);
    emitConnect(builder, intMod.getArguments().back(), sendPath.getResult());
  }

  if (!moduleTarget.instances.empty())
    srcTarget.instances = moduleTarget.instances;
  else {
    auto path = state.instancePathCache
                    .getAbsolutePaths(modInstance->getParentOfType<FModuleOp>())
                    .back();
    srcTarget.instances.append(path.begin(), path.end());
  }
  return sendVal;
}

// Describes tap points into the design.  This has the following structure:
//   keys: Seq[DataTapKey]
// DataTapKey has multiple implementations:
//   - ReferenceDataTapKey: (tapping a point which exists in the FIRRTL)
//       sink: ReferenceTarget
//       source: ReferenceTarget
//   - DataTapModuleSignalKey: (tapping a point, by name, in a blackbox)
//       module: IsModule
//       internalPath: String
//       sink: ReferenceTarget
//   - DeletedDataTapKey: (not implemented here)
//       sink: ReferenceTarget
//   - LiteralDataTapKey: (not implemented here)
//       literal: Literal
//       sink: ReferenceTarget
// A Literal is a FIRRTL IR literal serialized to a string.  For now, just
// store the string.
// TODO: Parse the literal string into a UInt or SInt literal.
LogicalResult circt::firrtl::applyGCTDataTaps(const AnnoPathValue &target,
                                              DictionaryAttr anno,
                                              ApplyState &state) {
  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();

  // Process all the taps.
  auto keyAttr = tryGetAs<ArrayAttr>(anno, anno, "keys", loc, dataTapsClass);
  if (!keyAttr)
    return failure();
  for (size_t i = 0, e = keyAttr.size(); i != e; ++i) {
    auto b = keyAttr[i];
    auto path = ("keys[" + Twine(i) + "]").str();
    auto bDict = cast<DictionaryAttr>(b);
    auto classAttr =
        tryGetAs<StringAttr>(bDict, anno, "class", loc, dataTapsClass, path);
    if (!classAttr)
      return failure();
    // Can only handle ReferenceDataTapKey and DataTapModuleSignalKey
    if (classAttr.getValue() != referenceKeyClass &&
        classAttr.getValue() != internalKeyClass)
      return mlir::emitError(loc, "Annotation '" + Twine(dataTapsClass) +
                                      "' with path '" +
                                      (Twine(path) + ".class") +
                                      "' contained an unknown/unimplemented "
                                      "DataTapKey class '" +
                                      classAttr.getValue() + "'.")
                 .attachNote()
             << "The full Annotation is reproduced here: " << anno << "\n";

    auto sinkNameAttr =
        tryGetAs<StringAttr>(bDict, anno, "sink", loc, dataTapsClass, path);
    std::string wirePathStr;
    if (sinkNameAttr)
      wirePathStr = canonicalizeTarget(sinkNameAttr.getValue());
    if (!wirePathStr.empty())
      if (!tokenizePath(wirePathStr))
        wirePathStr.clear();
    std::optional<AnnoPathValue> wireTarget;
    if (!wirePathStr.empty())
      wireTarget = resolvePath(wirePathStr, state.circuit, state.symTbl,
                               state.targetCaches);
    if (!wireTarget)
      return mlir::emitError(loc, "Annotation '" + Twine(dataTapsClass) +
                                      "' with wire path '" + wirePathStr +
                                      "' could not be resolved.");
    if (!wireTarget->ref.getImpl().isOp())
      return mlir::emitError(loc, "Annotation '" + Twine(dataTapsClass) +
                                      "' with path '" +
                                      (Twine(path) + ".class") +
                                      "' cannot specify a port for sink.");
    // Extract the name of the wire, used for datatap.
    auto tapName = StringAttr::get(
        context, wirePathStr.substr(wirePathStr.find_last_of('>') + 1));
    std::optional<AnnoPathValue> srcTarget;
    Value sendVal;
    if (classAttr.getValue() == internalKeyClass) {
      // For DataTapModuleSignalKey, the source is encoded as a string, that
      // should exist inside the specified module. This source string is used as
      // a suffix to the instance name for the module inside a VerbatimExprOp.
      // This verbatim represents an intermediate xmr, which is then used by a
      // ref.send to be read remotely.
      auto internalPathAttr = tryGetAs<StringAttr>(bDict, anno, "internalPath",
                                                   loc, dataTapsClass, path);
      auto moduleAttr =
          tryGetAs<StringAttr>(bDict, anno, "module", loc, dataTapsClass, path);
      if (!internalPathAttr || !moduleAttr)
        return failure();
      auto moduleTargetStr = canonicalizeTarget(moduleAttr.getValue());
      if (!tokenizePath(moduleTargetStr))
        return failure();
      std::optional<AnnoPathValue> moduleTarget = resolvePath(
          moduleTargetStr, state.circuit, state.symTbl, state.targetCaches);
      if (!moduleTarget)
        return failure();
      AnnoPathValue internalPathSrc;
      auto targetType =
          firrtl::type_cast<FIRRTLBaseType>(wireTarget->ref.getType());
      if (wireTarget->fieldIdx)
        targetType = firrtl::type_cast<FIRRTLBaseType>(
            targetType.getFinalTypeByFieldID(wireTarget->fieldIdx));
      sendVal = lowerInternalPathAnno(internalPathSrc, *moduleTarget, target,
                                      internalPathAttr, targetType, state);
      if (!sendVal)
        return failure();
      srcTarget = internalPathSrc;
    } else {
      // Now handle ReferenceDataTapKey. Get the source from annotation.
      auto sourceAttr =
          tryGetAs<StringAttr>(bDict, anno, "source", loc, dataTapsClass, path);
      if (!sourceAttr)
        return failure();
      auto sourcePathStr = canonicalizeTarget(sourceAttr.getValue());
      if (!tokenizePath(sourcePathStr))
        return failure();
      LLVM_DEBUG(llvm::dbgs() << "\n Drill xmr path from :" << sourcePathStr
                              << " to " << wirePathStr);
      srcTarget = resolvePath(sourcePathStr, state.circuit, state.symTbl,
                              state.targetCaches);
    }
    if (!srcTarget)
      return mlir::emitError(loc, "Annotation '" + Twine(dataTapsClass) +
                                      "' source path could not be resolved.");

    auto wireModule =
        cast<FModuleOp>(wireTarget->ref.getModule().getOperation());

    if (auto extMod = dyn_cast<FExtModuleOp>(srcTarget->ref.getOp())) {
      // If the source is a port on extern module, then move the source to the
      // instance port for the ext module.
      auto portNo = srcTarget->ref.getImpl().getPortNo();
      auto lastInst = srcTarget->instances.pop_back_val();
      auto builder = ImplicitLocOpBuilder::atBlockEnd(lastInst.getLoc(),
                                                      lastInst->getBlock());
      builder.setInsertionPointAfter(lastInst);
      // Instance port cannot be used as an annotation target, so use a NodeOp.
      auto node = builder.create<NodeOp>(lastInst.getResult(portNo));
      AnnotationSet::addDontTouch(node);
      srcTarget->ref = AnnoTarget(circt::firrtl::detail::AnnoTargetImpl(node));
    }

    // The RefSend value can be either generated by the instance of an external
    // module or a RefSendOp.
    if (!sendVal) {
      auto srcModule =
          dyn_cast<FModuleOp>(srcTarget->ref.getModule().getOperation());

      ImplicitLocOpBuilder sendBuilder(srcModule.getLoc(), srcModule);
      // Set the insertion point for the RefSend, it should be dominated by the
      // srcTarget value. If srcTarget is a port, then insert the RefSend
      // at the beggining of the module, else define the RefSend at the end of
      // the block that contains the srcTarget Op.
      if (srcTarget->ref.getImpl().isOp()) {
        sendVal = srcTarget->ref.getImpl().getOp()->getResult(0);
        sendBuilder.setInsertionPointAfter(srcTarget->ref.getOp());
      } else if (srcTarget->ref.getImpl().isPort()) {
        sendVal = srcModule.getArgument(srcTarget->ref.getImpl().getPortNo());
        sendBuilder.setInsertionPointToStart(srcModule.getBodyBlock());
      }
      // If the target value is a field of an aggregate create the
      // subfield/subaccess into it.
      sendVal = getValueByFieldID(sendBuilder, sendVal, srcTarget->fieldIdx);
      // Note: No DontTouch added to sendVal, it can be constantprop'ed or
      // CSE'ed.
    }

    auto *targetOp = wireTarget->ref.getOp();
    auto sinkBuilder = ImplicitLocOpBuilder::atBlockEnd(wireModule.getLoc(),
                                                        targetOp->getBlock());
    auto wireType = type_cast<FIRRTLBaseType>(targetOp->getResult(0).getType());
    // Get type of sent value, if already a RefType, the base type.
    auto valType = getBaseType(type_cast<FIRRTLType>(sendVal.getType()));
    Value sink = getValueByFieldID(sinkBuilder, targetOp->getResult(0),
                                   wireTarget->fieldIdx);

    // For resets, sometimes inject a cast between sink and target 'sink'.
    // Introduced a dummy wire and cast that, dummy wire will be 'sink'.
    if (valType.isResetType() &&
        valType.getWidthlessType() != wireType.getWidthlessType()) {
      // Helper: create a wire, cast it with callback, connect cast to sink.
      auto addWireWithCast = [&](auto createCast) {
        auto wire =
            sinkBuilder.create<WireOp>(valType, tapName.getValue()).getResult();
        emitConnect(sinkBuilder, sink, createCast(wire));
        sink = wire;
      };
      if (isa<IntType>(wireType))
        addWireWithCast(
            [&](auto v) { return sinkBuilder.create<AsUIntPrimOp>(v); });
      else if (isa<AsyncResetType>(wireType))
        addWireWithCast(
            [&](auto v) { return sinkBuilder.create<AsAsyncResetPrimOp>(v); });
    }

    state.wiringProblems.push_back(
        {sendVal, sink, "", WiringProblem::RefTypeUsage::Prefer});
  }

  return success();
}

LogicalResult circt::firrtl::applyGCTMemTaps(const AnnoPathValue &target,
                                             DictionaryAttr anno,
                                             ApplyState &state) {
  auto loc = state.circuit.getLoc();

  auto sourceAttr =
      tryGetAs<StringAttr>(anno, anno, "source", loc, memTapClass);
  if (!sourceAttr)
    return failure();
  auto sourceTargetStr = canonicalizeTarget(sourceAttr.getValue());

  Value memDbgPort;
  std::optional<AnnoPathValue> srcTarget = resolvePath(
      sourceTargetStr, state.circuit, state.symTbl, state.targetCaches);
  if (!srcTarget)
    return mlir::emitError(loc, "cannot resolve source target path '")
           << sourceTargetStr << "'";
  auto tapsAttr = tryGetAs<ArrayAttr>(anno, anno, "sink", loc, memTapClass);
  if (!tapsAttr || tapsAttr.empty())
    return mlir::emitError(loc, "sink must have at least one entry");
  if (auto combMem = dyn_cast<chirrtl::CombMemOp>(srcTarget->ref.getOp())) {
    if (!combMem.getType().getElementType().isGround())
      return combMem.emitOpError(
          "cannot generate MemTap to a memory with aggregate data type");
    ImplicitLocOpBuilder builder(combMem->getLoc(), combMem);
    builder.setInsertionPointAfter(combMem);
    // Construct the type for the debug port.
    auto debugType =
        RefType::get(FVectorType::get(combMem.getType().getElementType(),
                                      combMem.getType().getNumElements()));

    auto debugPort = builder.create<chirrtl::MemoryDebugPortOp>(
        debugType, combMem,
        state.getNamespace(srcTarget->ref.getModule()).newName("memTap"));

    memDbgPort = debugPort.getResult();
    if (srcTarget->instances.empty()) {
      auto path = state.instancePathCache.getAbsolutePaths(
          combMem->getParentOfType<FModuleOp>());
      if (path.size() > 1)
        return combMem.emitOpError(
            "cannot be resolved as source for MemTap, multiple paths from top "
            "exist and unique instance cannot be resolved");
      srcTarget->instances.append(path.back().begin(), path.back().end());
    }
    if (tapsAttr.size() != combMem.getType().getNumElements())
      return mlir::emitError(
          loc, "sink cannot specify more taps than the depth of the memory");
  } else
    return srcTarget->ref.getOp()->emitOpError(
        "unsupported operation, only CombMem can be used as the source of "
        "MemTap");

  auto tap = tapsAttr[0].dyn_cast_or_null<StringAttr>();
  if (!tap) {
    return mlir::emitError(
               loc, "Annotation '" + Twine(memTapClass) +
                        "' with path '.taps[0" +
                        "]' contained an unexpected type (expected a string).")
               .attachNote()
           << "The full Annotation is reprodcued here: " << anno << "\n";
  }
  auto wireTargetStr = canonicalizeTarget(tap.getValue());
  if (!tokenizePath(wireTargetStr))
    return failure();
  std::optional<AnnoPathValue> wireTarget = resolvePath(
      wireTargetStr, state.circuit, state.symTbl, state.targetCaches);
  if (!wireTarget)
    return mlir::emitError(loc, "Annotation '" + Twine(memTapClass) +
                                    "' with path '.taps[0]' contains target '" +
                                    wireTargetStr +
                                    "' that cannot be resolved.")
               .attachNote()
           << "The full Annotation is reproduced here: " << anno << "\n";

  auto sendVal = memDbgPort;
  if (wireTarget->ref.getOp()->getResult(0).getType() !=
      type_cast<RefType>(sendVal.getType()).getType())
    return wireTarget->ref.getOp()->emitError(
        "cannot generate the MemTap, wiretap Type does not match the memory "
        "type");
  auto sink = wireTarget->ref.getOp()->getResult(0);
  state.wiringProblems.push_back(
      {sendVal, sink, "memTap", WiringProblem::RefTypeUsage::Prefer});
  return success();
}
