//===- FIRRTLAnnotationHelper.h - FIRRTL Annotation Lookup ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helpers mapping annotations to operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLANNOTATIONHELPER_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLANNOTATIONHELPER_H

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace firrtl {

/// Stores an index into an aggregate.
struct TargetToken {
  StringRef name;
  bool isIndex;
};

/// The parsed annotation path.
struct TokenAnnoTarget {
  StringRef circuit;
  SmallVector<std::pair<StringRef, StringRef>> instances;
  StringRef module;
  // The final name of the target
  StringRef name;
  // Any aggregates indexed.
  SmallVector<TargetToken> component;

  /// Append the annotation path to the given `SmallString` or `SmallVector`.
  void toVector(SmallVectorImpl<char> &out) const;

  /// Convert the annotation path to a string.
  std::string str() const {
    SmallString<32> out;
    toVector(out);
    return std::string(out);
  }
};

// The potentially non-local resolved annotation.
struct AnnoPathValue {
  SmallVector<InstanceOp> instances;
  AnnoTarget ref;
  unsigned fieldIdx = 0;

  AnnoPathValue() = default;
  AnnoPathValue(CircuitOp op) : ref(OpAnnoTarget(op)) {}
  AnnoPathValue(Operation *op) : ref(OpAnnoTarget(op)) {}
  AnnoPathValue(const SmallVectorImpl<InstanceOp> &insts, AnnoTarget b,
                unsigned fieldIdx)
      : instances(insts.begin(), insts.end()), ref(b), fieldIdx(fieldIdx) {}

  bool isLocal() const { return instances.empty(); }

  template <typename... T>
  bool isOpOfType() const {
    if (auto opRef = ref.dyn_cast<OpAnnoTarget>())
      return isa<T...>(opRef.getOp());
    return false;
  }
};

template <typename T>
static T &operator<<(T &os, const AnnoPathValue &path) {
  os << "~" << path.ref.getModule()->getParentOfType<CircuitOp>().getName()
     << "|";

  if (path.isLocal()) {
    os << path.ref.getModule().getModuleName();
  } else {
    os << path.instances.front()
              ->getParentOfType<FModuleLike>()
              .getModuleName();
  }
  for (auto inst : path.instances)
    os << "/" << inst.getName() << ":" << inst.getModuleName();
  if (!path.isOpOfType<FModuleOp, FExtModuleOp, InstanceOp>()) {
    os << ">" << path.ref;
    auto type = dyn_cast<FIRRTLBaseType>(path.ref.getType());
    if (!type)
      return os;
    auto targetFieldID = path.fieldIdx;
    while (targetFieldID) {
      FIRRTLTypeSwitch<FIRRTLBaseType>(type)
          .Case<FVectorType>([&](FVectorType vector) {
            auto index = vector.getIndexForFieldID(targetFieldID);
            os << "[" << index << "]";
            type = vector.getElementType();
            targetFieldID -= vector.getFieldID(index);
          })
          .template Case<BundleType>([&](BundleType bundle) {
            auto index = bundle.getIndexForFieldID(targetFieldID);
            os << "." << bundle.getElementName(index);
            type = bundle.getElementType(index);
            targetFieldID -= bundle.getFieldID(index);
          })
          .Default([&](auto) { targetFieldID = 0; });
    }
  }
  return os;
}

template <typename T>
static T &operator<<(T &os, const OpAnnoTarget &target) {
  os << target.getOp()->getAttrOfType<StringAttr>("name").getValue();
  return os;
}

template <typename T>
static T &operator<<(T &os, const PortAnnoTarget &target) {
  os << target.getModule().getPortName(target.getPortNo());
  return os;
}

template <typename T>
static T &operator<<(T &os, const AnnoTarget &target) {
  if (auto op = target.dyn_cast<OpAnnoTarget>())
    os << op;
  else if (auto port = target.dyn_cast<PortAnnoTarget>())
    os << port;
  else
    os << "<<Unknown Anno Target>>";
  return os;
}

/// Cache AnnoTargets for a module's named things.
struct AnnoTargetCache {
  AnnoTargetCache() = delete;
  AnnoTargetCache(const AnnoTargetCache &other) = default;
  AnnoTargetCache(AnnoTargetCache &&other)
      : targets(std::move(other.targets)){};

  AnnoTargetCache(FModuleLike mod) { gatherTargets(mod); };

  /// Lookup the target for 'name', empty if not found.
  /// (check for validity using operator bool()).
  AnnoTarget getTargetForName(StringRef name) const {
    return targets.lookup(name);
  }

  void insertOp(Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<InstanceOp, MemOp, NodeOp, RegOp, RegResetOp, WireOp,
              chirrtl::CombMemOp, chirrtl::SeqMemOp, chirrtl::MemoryPortOp,
              chirrtl::MemoryDebugPortOp, PrintFOp>([&](auto op) {
          // To be safe, check attribute and non-empty name before adding.
          if (auto name = op.getNameAttr(); name && !name.getValue().empty())
            targets.insert({name, OpAnnoTarget(op)});
        });
  }

  /// Replace `oldOp` with `newOp` in the target cache. The new and old ops can
  /// have different names.
  void replaceOp(Operation *oldOp, Operation *newOp) {
    if (auto name = oldOp->getAttrOfType<StringAttr>("name");
        name && !name.getValue().empty())
      targets.erase(name);
    insertOp(newOp);
  }

  /// Add a new module port to the target cache.
  void insertPort(FModuleLike mod, size_t portNo) {
    targets.insert({mod.getPortNameAttr(portNo), PortAnnoTarget(mod, portNo)});
  }

private:
  /// Walk the module and add named things to 'targets'.
  void gatherTargets(FModuleLike mod);

  llvm::DenseMap<StringRef, AnnoTarget> targets;
};

/// Cache AnnoTargets for a circuit's modules, walked as needed.
struct CircuitTargetCache {
  /// Get cache for specified module, creating it as needed.
  /// Returned reference may become invalidated by future calls.
  const AnnoTargetCache &getOrCreateCacheFor(FModuleLike module) {
    auto it = targetCaches.find(module);
    if (it == targetCaches.end())
      it = targetCaches.try_emplace(module, module).first;
    return it->second;
  }

  /// Lookup the target for 'name' in 'module'.
  AnnoTarget lookup(FModuleLike module, StringRef name) {
    return getOrCreateCacheFor(module).getTargetForName(name);
  }

  /// Clear the cache completely.
  void invalidate() { targetCaches.clear(); }

  /// Replace `oldOp` with `newOp` in the target cache. The new and old ops can
  /// have different names.
  void replaceOp(Operation *oldOp, Operation *newOp) {
    auto mod = newOp->getParentOfType<FModuleOp>();
    auto it = targetCaches.find(mod);
    if (it == targetCaches.end())
      return;
    it->getSecond().replaceOp(oldOp, newOp);
  }

  /// Add a new module port to the target cache.
  void insertPort(FModuleLike mod, size_t portNo) {
    auto it = targetCaches.find(mod);
    if (it == targetCaches.end())
      return;
    it->getSecond().insertPort(mod, portNo);
  }

  /// Add a new op to the target cache.
  void insertOp(Operation *op) {
    auto mod = op->getParentOfType<FModuleOp>();
    auto it = targetCaches.find(mod);
    if (it == targetCaches.end())
      return;
    it->getSecond().insertOp(op);
  }

private:
  DenseMap<Operation *, AnnoTargetCache> targetCaches;
};

/// Return an input \p target string in canonical form.  This converts a Legacy
/// Annotation (e.g., A.B.C) into a modern annotation (e.g., ~A|B>C).  Trailing
/// subfield/subindex references are preserved.
std::string canonicalizeTarget(StringRef target);

/// Parse a FIRRTL annotation path into its constituent parts.
std::optional<TokenAnnoTarget> tokenizePath(StringRef origTarget);

/// Convert a parsed target string to a resolved target structure.  This
/// resolves all names and aggregates from a parsed target.
std::optional<AnnoPathValue> resolveEntities(TokenAnnoTarget path,
                                             CircuitOp circuit,
                                             SymbolTable &symTbl,
                                             CircuitTargetCache &cache);

/// Resolve a string path to a named item inside a circuit.
std::optional<AnnoPathValue> resolvePath(StringRef rawPath, CircuitOp circuit,
                                         SymbolTable &symTbl,
                                         CircuitTargetCache &cache);

/// Return true if an Annotation's class name is handled by the LowerAnnotations
/// pass.
bool isAnnoClassLowered(StringRef className);

/// A representation of a deferred Wiring problem consisting of a source that
/// should be connected to a sink.
struct WiringProblem {
  enum class RefTypeUsage { Prefer, Never };

  /// A source to wire from.
  Value source;

  /// A sink to wire to.
  Value sink;

  /// A base name to use when generating new signals associated with this wiring
  /// problem.
  std::string newNameHint;

  /// The usage of ref type ports when solving this problem.
  RefTypeUsage refTypeUsage;
};

/// A representation of a legacy Wiring problem consisting of a signal source
/// that should be connected to one or many sinks.
struct LegacyWiringProblem {
  /// A source to wire from.
  Value source;

  /// Sink(s) to wire to.
  SmallVector<Value> sinks;
};

/// A store of pending modifications to a FIRRTL module associated with solving
/// one or more WiringProblems.
struct ModuleModifications {
  /// A pair of Wiring Problem index and port information.
  using portInfoPair = std::pair<size_t, PortInfo>;

  /// A pair of Wiring Problem index and a U-turn Value that should be
  /// connected.
  using uturnPair = std::pair<size_t, Value>;

  /// Ports that should be added to a module.
  SmallVector<portInfoPair> portsToAdd;

  /// A mapping of a Value that should be connected to either a new port or a
  /// U-turn, for a specific Wiring Problem.  This is pre-populated with the
  /// source and sink.
  DenseMap<size_t, Value> connectionMap;

  /// A secondary value that _may_ need to be hooked up.  This is always set
  /// after the Value in the connectionMap.
  SmallVector<uturnPair> uturns;
};

/// State threaded through functions for resolving and applying annotations.
struct ApplyState {
  using AddToWorklistFn = llvm::function_ref<void(DictionaryAttr)>;
  ApplyState(CircuitOp circuit, SymbolTable &symTbl,
             AddToWorklistFn addToWorklistFn,
             InstancePathCache &instancePathCache)
      : circuit(circuit), symTbl(symTbl), addToWorklistFn(addToWorklistFn),
        instancePathCache(instancePathCache) {}

  CircuitOp circuit;
  SymbolTable &symTbl;
  CircuitTargetCache targetCaches;
  AddToWorklistFn addToWorklistFn;
  InstancePathCache &instancePathCache;
  DenseMap<Attribute, FlatSymbolRefAttr> instPathToNLAMap;
  size_t numReusedHierPaths = 0;

  DenseSet<InstanceOp> wiringProblemInstRefs;
  DenseMap<StringAttr, LegacyWiringProblem> legacyWiringProblems;
  SmallVector<WiringProblem> wiringProblems;

  ModuleNamespace &getNamespace(FModuleLike module) {
    auto &ptr = namespaces[module];
    if (!ptr)
      ptr = std::make_unique<ModuleNamespace>(module);
    return *ptr;
  }

  IntegerAttr newID() {
    return IntegerAttr::get(IntegerType::get(circuit.getContext(), 64),
                            annotationID++);
  };

private:
  DenseMap<Operation *, std::unique_ptr<ModuleNamespace>> namespaces;
  unsigned annotationID = 0;
};

LogicalResult applyGCTView(const AnnoPathValue &target, DictionaryAttr anno,
                           ApplyState &state);

LogicalResult applyGCTDataTaps(const AnnoPathValue &target, DictionaryAttr anno,
                               ApplyState &state);

LogicalResult applyGCTMemTaps(const AnnoPathValue &target, DictionaryAttr anno,
                              ApplyState &state);

LogicalResult applyOMIR(const AnnoPathValue &target, DictionaryAttr anno,
                        ApplyState &state);

LogicalResult applyTraceName(const AnnoPathValue &target, DictionaryAttr anno,
                             ApplyState &state);

LogicalResult applyWiring(const AnnoPathValue &target, DictionaryAttr anno,
                          ApplyState &state);

/// Implements the same behavior as DictionaryAttr::getAs<A> to return the
/// value of a specific type associated with a key in a dictionary. However,
/// this is specialized to print a useful error message, specific to custom
/// annotation process, on failure.
template <typename A>
A tryGetAs(DictionaryAttr &dict, const Attribute &root, StringRef key,
           Location loc, Twine className, Twine path = Twine()) {
  // Check that the key exists.
  auto value = dict.get(key);
  if (!value) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className + "' did not contain required key '" +
             key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain required key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  // Check that the value has the correct type.
  auto valueA = dyn_cast_or_null<A>(value);
  if (!valueA) {
    SmallString<128> msg;
    if (path.isTriviallyEmpty())
      msg = ("Annotation '" + className +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    else
      msg = ("Annotation '" + className + "' with path '" + path +
             "' did not contain the correct type for key '" + key + "'.")
                .str();
    mlir::emitError(loc, msg).attachNote()
        << "The full Annotation is reproduced here: " << root << "\n";
    return nullptr;
  }
  return valueA;
}

/// Add ports to the module and all its instances and return the clone for
/// `instOnPath`. This does not connect the new ports to anything. Replace
/// the old instances with the new cloned instance in all the caches.
InstanceOp addPortsToModule(FModuleLike mod, InstanceOp instOnPath,
                            FIRRTLType portType, Direction dir,
                            StringRef newName,
                            InstancePathCache &instancePathcache,
                            CircuitTargetCache *targetCaches = nullptr);

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLANNOTATIONHELPER_H
