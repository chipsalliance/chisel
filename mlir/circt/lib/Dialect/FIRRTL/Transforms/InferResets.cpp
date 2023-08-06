//===- InferResets.cpp - Infer resets and add async reset -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferResets pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "infer-resets"

using llvm::BumpPtrAllocator;
using llvm::MapVector;
using llvm::SmallDenseSet;
using llvm::SmallSetVector;
using mlir::FailureOr;
using mlir::InferTypeOpInterface;
using mlir::WalkOrder;

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// An absolute instance path.
using InstanceLike = circt::hw::HWInstanceLike;
using InstancePathRef = ArrayRef<InstanceLike>;
using InstancePathVec = SmallVector<InstanceLike>;

template <typename T>
static T &operator<<(T &os, InstancePathRef path) {
  os << "$root";
  for (InstanceLike inst : path)
    os << "/" << inst.getInstanceName() << ":"
       << inst.getReferencedModuleName();
  return os;
}

#ifndef NDEBUG
static StringRef getTail(InstancePathRef path) {
  if (path.empty())
    return "$root";
  auto last = path.back();
  return last.getInstanceName();
}
#endif

namespace {
/// A reset domain.
struct ResetDomain {
  /// Whether this is the root of the reset domain.
  bool isTop = false;
  /// The reset signal for this domain. A null value indicates that this domain
  /// explicitly has no reset.
  Value reset;

  // Implementation details for this domain.
  Value existingValue;
  std::optional<unsigned> existingPort;
  StringAttr newPortName;

  ResetDomain(Value reset) : reset(reset) {}
};
} // namespace

inline bool operator==(const ResetDomain &a, const ResetDomain &b) {
  return (a.isTop == b.isTop && a.reset == b.reset);
}
inline bool operator!=(const ResetDomain &a, const ResetDomain &b) {
  return !(a == b);
}

/// Return the name and parent module of a reset. The reset value must either be
/// a module port or a wire/node operation.
static std::pair<StringAttr, FModuleOp> getResetNameAndModule(Value reset) {
  if (auto arg = dyn_cast<BlockArgument>(reset)) {
    auto module = cast<FModuleOp>(arg.getParentRegion()->getParentOp());
    return {module.getPortNameAttr(arg.getArgNumber()), module};
  } else {
    auto op = reset.getDefiningOp();
    return {op->getAttrOfType<StringAttr>("name"),
            op->getParentOfType<FModuleOp>()};
  }
}

/// Return the name of a reset. The reset value must either be a module port or
/// a wire/node operation.
static inline StringAttr getResetName(Value reset) {
  return getResetNameAndModule(reset).first;
}

/// Construct a zero value of the given type using the given builder.
static Value createZeroValue(ImplicitLocOpBuilder &builder, FIRRTLBaseType type,
                             SmallDenseMap<FIRRTLBaseType, Value> &cache) {
  // The zero value's type is a const version of `type`.
  type = type.getConstType(true);
  auto it = cache.find(type);
  if (it != cache.end())
    return it->second;
  auto nullBit = [&]() {
    return createZeroValue(
        builder, UIntType::get(builder.getContext(), 1, /*isConst=*/true),
        cache);
  };
  auto value =
      FIRRTLTypeSwitch<FIRRTLBaseType, Value>(type)
          .Case<ClockType>([&](auto type) {
            return builder.create<AsClockPrimOp>(nullBit());
          })
          .Case<AsyncResetType>([&](auto type) {
            return builder.create<AsAsyncResetPrimOp>(nullBit());
          })
          .Case<SIntType, UIntType>([&](auto type) {
            return builder.create<ConstantOp>(
                type, APInt::getZero(type.getWidth().value_or(1)));
          })
          .Case<BundleType>([&](auto type) {
            auto wireOp = builder.create<WireOp>(type);
            for (unsigned i = 0, e = type.getNumElements(); i < e; ++i) {
              auto fieldType = type.getElementTypePreservingConst(i);
              auto zero = createZeroValue(builder, fieldType, cache);
              auto acc =
                  builder.create<SubfieldOp>(fieldType, wireOp.getResult(), i);
              builder.create<StrictConnectOp>(acc, zero);
            }
            return wireOp.getResult();
          })
          .Case<FVectorType>([&](auto type) {
            auto wireOp = builder.create<WireOp>(type);
            auto zero = createZeroValue(
                builder, type.getElementTypePreservingConst(), cache);
            for (unsigned i = 0, e = type.getNumElements(); i < e; ++i) {
              auto acc = builder.create<SubindexOp>(zero.getType(),
                                                    wireOp.getResult(), i);
              builder.create<StrictConnectOp>(acc, zero);
            }
            return wireOp.getResult();
          })
          .Case<ResetType, AnalogType>(
              [&](auto type) { return builder.create<InvalidValueOp>(type); })
          .Default([](auto) {
            llvm_unreachable("switch handles all types");
            return Value{};
          });
  cache.insert({type, value});
  return value;
}

/// Construct a null value of the given type using the given builder.
static Value createZeroValue(ImplicitLocOpBuilder &builder,
                             FIRRTLBaseType type) {
  SmallDenseMap<FIRRTLBaseType, Value> cache;
  return createZeroValue(builder, type, cache);
}

/// Helper function that inserts reset multiplexer into all `ConnectOp`s
/// with the given target. Looks through `SubfieldOp`, `SubindexOp`,
/// and `SubaccessOp`, and inserts multiplexers into connects to
/// these subaccesses as well. Modifies the insertion location of the builder.
/// Returns true if the `resetValue` was used in any way, false otherwise.
static bool insertResetMux(ImplicitLocOpBuilder &builder, Value target,
                           Value reset, Value resetValue) {
  // Indicates whether the `resetValue` was assigned to in some way. We use this
  // to erase unused subfield/subindex/subaccess ops on the reset value if they
  // end up unused.
  bool resetValueUsed = false;

  for (auto &use : target.getUses()) {
    Operation *useOp = use.getOwner();
    builder.setInsertionPoint(useOp);
    TypeSwitch<Operation *>(useOp)
        // Insert a mux on the value connected to the target:
        // connect(dst, src) -> connect(dst, mux(reset, resetValue, src))
        .Case<ConnectOp, StrictConnectOp>([&](auto op) {
          if (op.getDest() != target)
            return;
          LLVM_DEBUG(llvm::dbgs() << "  - Insert mux into " << op << "\n");
          auto muxOp =
              builder.create<MuxPrimOp>(reset, resetValue, op.getSrc());
          op.getSrcMutable().assign(muxOp);
          resetValueUsed = true;
        })
        // Look through subfields.
        .Case<SubfieldOp>([&](auto op) {
          auto resetSubValue =
              builder.create<SubfieldOp>(resetValue, op.getFieldIndexAttr());
          if (insertResetMux(builder, op, reset, resetSubValue))
            resetValueUsed = true;
          else
            resetSubValue.erase();
        })
        // Look through subindices.
        .Case<SubindexOp>([&](auto op) {
          auto resetSubValue =
              builder.create<SubindexOp>(resetValue, op.getIndexAttr());
          if (insertResetMux(builder, op, reset, resetSubValue))
            resetValueUsed = true;
          else
            resetSubValue.erase();
        })
        // Look through subaccesses.
        .Case<SubaccessOp>([&](auto op) {
          if (op.getInput() != target)
            return;
          auto resetSubValue =
              builder.create<SubaccessOp>(resetValue, op.getIndex());
          if (insertResetMux(builder, op, reset, resetSubValue))
            resetValueUsed = true;
          else
            resetSubValue.erase();
        });
  }
  return resetValueUsed;
}

//===----------------------------------------------------------------------===//
// Reset Network
//===----------------------------------------------------------------------===//

namespace {

/// A reset signal.
///
/// This essentially combines the exact `FieldRef` of the signal in question
/// with a type to be used for error reporting and inferring the reset kind.
struct ResetSignal {
  ResetSignal(FieldRef field, FIRRTLBaseType type) : field(field), type(type) {}
  bool operator<(const ResetSignal &other) const { return field < other.field; }
  bool operator==(const ResetSignal &other) const {
    return field == other.field;
  }
  bool operator!=(const ResetSignal &other) const { return !(*this == other); }

  FieldRef field;
  FIRRTLBaseType type;
};

/// A connection made to or from a reset network.
///
/// These drives are tracked for each reset network, and are used for error
/// reporting to the user.
struct ResetDrive {
  /// What's being driven.
  ResetSignal dst;
  /// What's driving.
  ResetSignal src;
  /// The location to use for diagnostics.
  Location loc;
};

/// A list of connections to a reset network.
using ResetDrives = SmallVector<ResetDrive, 1>;

/// All signals connected together into a reset network.
using ResetNetwork = llvm::iterator_range<
    llvm::EquivalenceClasses<ResetSignal>::member_iterator>;

/// Whether a reset is sync or async.
enum class ResetKind { Async, Sync };

} // namespace

namespace llvm {
template <>
struct DenseMapInfo<ResetSignal> {
  static inline ResetSignal getEmptyKey() {
    return ResetSignal{DenseMapInfo<FieldRef>::getEmptyKey(), {}};
  }
  static inline ResetSignal getTombstoneKey() {
    return ResetSignal{DenseMapInfo<FieldRef>::getTombstoneKey(), {}};
  }
  static unsigned getHashValue(const ResetSignal &x) {
    return circt::hash_value(x.field);
  }
  static bool isEqual(const ResetSignal &lhs, const ResetSignal &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

template <typename T>
static T &operator<<(T &os, const ResetKind &kind) {
  switch (kind) {
  case ResetKind::Async:
    return os << "async";
  case ResetKind::Sync:
    return os << "sync";
  }
  return os;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
/// Infer concrete reset types and insert full async reset.
///
/// This pass replaces `reset` types in the IR with a concrete `asyncreset` or
/// `uint<1>` depending on how the reset is used, and adds async resets to
/// registers in modules marked with the corresponding
/// `FullAsyncResetAnnotation`.
///
/// On a high level, the first stage of the pass that deals with reset inference
/// operates as follows:
///
/// 1. Build a global graph of the resets in the design by tracing reset signals
///    through instances. This uses the `ResetNetwork` utilities and boils down
///    to finding  groups of values in the IR that are part of the same reset
///    network (i.e., somehow attached together through ports, wires, instances,
///    and connects). We use LLVM's `EquivalenceClasses` data structure to do
///    this efficiently.
///
/// 2. Infer the type of each reset network found in step 1 by looking at the
///    type of values connected to the network. This results in the network
///    being declared a sync (`uint<1>`) or async (`asyncreset`) network. If the
///    reset is never driven by a concrete type, an error is emitted.
///
/// 3. Walk the IR and update the type of wires and ports with the reset types
///    found in step 2. This will replace all `reset` types in the IR with
///    a concrete type.
///
/// The second stage that deals with the addition of async resets operates as
/// follows:
///
/// 4. Visit every module in the design and determine if it has an explicit
///    reset annotated. Ports of and wires in the module can have a
///    `FullAsyncResetAnnotation`, which marks that port or wire as the async
///    reset for the module. A module may also carry a
///    `IgnoreFullAsyncResetAnnotation`, which marks it as being explicitly not
///    in a reset domain. These annotations are sparse; it is very much possible
///    that just the top-level module in the design has a full async reset
///    annotation. A module can only ever carry one of these annotations, which
///    puts it into one of three categories from an async reset inference
///    perspective:
///
///      a. unambiguously marks a port or wire as the module's async reset
///      b. explicitly marks it as not to have any async resets added
///      c. inherit reset
///
/// 5. For every module in the design, determine the full async reset domain it
///    is in. Note that this very narrowly deals with the inference of a
///    "default" async reset, which basically goes through the IR and attaches
///    all non-reset registers to a default async reset signal. If a module
///    carries one of the annotations mentioned in (4), the annotated port or
///    wire is used as its reset domain. Otherwise, it inherits the reset domain
///    from parent modules. This conceptually involves looking at all the places
///    where a module is instantiated, and recursively determining the reset
///    domain at the instantiation site. A module can only ever be in one reset
///    domain. In case it is inferred to lie in multiple ones, e.g., if it is
///    instantiated in different reset domains, an error is emitted. If
///    successful, every module is associated with a reset signal, either one of
///    its local ports or wires, or a port or wire within one of its parent
///    modules.
///
/// 6. For every module in the design, determine how async resets shall be
///    implemented. This step handles the following distinct cases:
///
///      a. Skip a module because it is marked as having no reset domain.
///      b. Use a port or wire in the module itself as reset. This is possible
///         if the module is at the "top" of its reset domain, which means that
///         it itself carried a reset annotation, and the reset value is either
///         a port or wire of the module itself.
///      c. Route a parent module's reset through a module port and use that
///         port as the reset. This happens if the module is *not* at the "top"
///         of its reset domain, but rather refers to a value in a parent module
///         as its reset.
///
///    As a result, a module's reset domain is annotated with the existing local
///    value to reuse (port or wire), the index of an existing port to reuse,
///    and the name of an additional port to insert into its port list.
///
/// 7. For every module in the design, async resets are implemented. This
///    determines the local value to use as the reset signal and updates the
///    `reg` and `regreset` operations in the design. If the register already
///    has an async reset, it is left unchanged. If it has a sync reset, the
///    sync reset is moved into a `mux` operation on all `connect`s to the
///    register (which the Scala code base called the `RemoveResets` pass).
///    Finally the register is replaced with a `regreset` operation, with the
///    reset signal determined earlier, and a "zero" value constructed for the
///    register's type.
///
///    Determining the local reset value is trivial if step 6 found a module to
///    be of case a or b. Case c is the non-trivial one, because it requires
///    modifying the port list of the module. This is done by first determining
///    the name of the reset signal in the parent module, which is either the
///    name of the port or wire declaration. We then look for an existing
///    `asyncreset` port in the port list and reuse that as reset. If no port
///    with that name was found, or the existing port is of the wrong type, a
///    new port is inserted into the port list.
///
///    TODO: This logic is *very* brittle and error-prone. It may make sense to
///    just add an additional port for the inferred async reset in any case,
///    with an optimization to use an existing `asyncreset` port if all of the
///    module's instantiations have that port connected to the desired signal
///    already.
///
struct InferResetsPass : public InferResetsBase<InferResetsPass> {
  void runOnOperation() override;
  void runOnOperationInner();

  // Copy creates a new empty pass (because ResetMap has no copy constructor).
  using InferResetsBase::InferResetsBase;
  InferResetsPass(const InferResetsPass &other) : InferResetsBase(other) {}

  //===--------------------------------------------------------------------===//
  // Reset type inference

  void traceResets(CircuitOp circuit);
  void traceResets(InstanceOp inst);
  void traceResets(Value dst, Value src, Location loc);
  void traceResets(Value value);
  void traceResets(Type dstType, Value dst, unsigned dstID, Type srcType,
                   Value src, unsigned srcID, Location loc);

  LogicalResult inferAndUpdateResets();
  FailureOr<ResetKind> inferReset(ResetNetwork net);
  LogicalResult updateReset(ResetNetwork net, ResetKind kind);
  bool updateReset(FieldRef field, FIRRTLBaseType resetType);

  //===--------------------------------------------------------------------===//
  // Async reset implementation

  LogicalResult collectAnnos(CircuitOp circuit);
  // Collect async reset annotations in the module and return a reset signal.
  // Return `failure()` if there was an error in the annotation processing.
  // Return `std::nullopt` if there was no async reset annotation.
  // Return `nullptr` if there was `ignore` annotation.
  // Return a non-null Value if the reset was actually provided.
  FailureOr<std::optional<Value>> collectAnnos(FModuleOp module);

  LogicalResult buildDomains(CircuitOp circuit);
  void buildDomains(FModuleOp module, const InstancePathVec &instPath,
                    Value parentReset, InstanceGraph &instGraph,
                    unsigned indent = 0);

  void determineImpl();
  void determineImpl(FModuleOp module, ResetDomain &domain);

  LogicalResult implementAsyncReset();
  LogicalResult implementAsyncReset(FModuleOp module, ResetDomain &domain);
  void implementAsyncReset(Operation *op, FModuleOp module, Value actualReset);

  LogicalResult verifyNoAbstractReset();

  //===--------------------------------------------------------------------===//
  // Utilities

  /// Get the reset network a signal belongs to.
  ResetNetwork getResetNetwork(ResetSignal signal) {
    return llvm::make_range(resetClasses.findLeader(signal),
                            resetClasses.member_end());
  }

  /// Get the drives of a reset network.
  ResetDrives &getResetDrives(ResetNetwork net) {
    return resetDrives[*net.begin()];
  }

  /// Guess the root node of a reset network, such that we have something for
  /// the user to make sense of.
  ResetSignal guessRoot(ResetNetwork net);
  ResetSignal guessRoot(ResetSignal signal) {
    return guessRoot(getResetNetwork(signal));
  }

  //===--------------------------------------------------------------------===//
  // Analysis data

  /// A map of all traced reset networks in the circuit.
  llvm::EquivalenceClasses<ResetSignal> resetClasses;

  /// A map of all connects to and from a reset.
  DenseMap<ResetSignal, ResetDrives> resetDrives;

  /// The annotated reset for a module. A null value indicates that the module
  /// is explicitly annotated with `ignore`. Otherwise the port/wire/node
  /// annotated as reset within the module is stored.
  DenseMap<Operation *, Value> annotatedResets;

  /// The reset domain for a module. In case of conflicting domain membership,
  /// the vector for a module contains multiple elements.
  MapVector<FModuleOp, SmallVector<std::pair<ResetDomain, InstancePathVec>, 1>>
      domains;

  /// Cache of modules symbols
  InstanceGraph *instanceGraph;
};
} // namespace

void InferResetsPass::runOnOperation() {
  runOnOperationInner();
  resetClasses = llvm::EquivalenceClasses<ResetSignal>();
  resetDrives.clear();
  annotatedResets.clear();
  domains.clear();
  markAnalysesPreserved<InstanceGraph>();
}

void InferResetsPass::runOnOperationInner() {
  instanceGraph = &getAnalysis<InstanceGraph>();

  // Trace the uninferred reset networks throughout the design.
  traceResets(getOperation());

  // Infer the type of the traced resets and update the IR.
  if (failed(inferAndUpdateResets()))
    return signalPassFailure();

  // Gather the reset annotations throughout the modules.
  if (failed(collectAnnos(getOperation())))
    return signalPassFailure();

  // Build the reset domains in the design.
  if (failed(buildDomains(getOperation())))
    return signalPassFailure();

  // Determine how each reset shall be implemented.
  determineImpl();

  // Implement the async resets.
  if (failed(implementAsyncReset()))
    return signalPassFailure();

  // Require that no Abstract Resets exist on ports in the design.
  if (failed(verifyNoAbstractReset()))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInferResetsPass() {
  return std::make_unique<InferResetsPass>();
}

ResetSignal InferResetsPass::guessRoot(ResetNetwork net) {
  ResetDrives &drives = getResetDrives(net);
  ResetSignal bestSignal = *net.begin();
  unsigned bestNumDrives = -1;

  for (auto signal : net) {
    // Don't consider `invalidvalue` for reporting as a root.
    if (isa_and_nonnull<InvalidValueOp>(
            signal.field.getValue().getDefiningOp()))
      continue;

    // Count the number of times this particular signal in the reset network is
    // assigned to.
    unsigned numDrives = 0;
    for (auto &drive : drives)
      if (drive.dst == signal)
        ++numDrives;

    // Keep track of the signal with the lowest number of assigns. These tend to
    // be the signals further up the reset tree. This will usually resolve to
    // the root of the reset tree far up in the design hierarchy.
    if (numDrives < bestNumDrives) {
      bestNumDrives = numDrives;
      bestSignal = signal;
    }
  }
  return bestSignal;
}

//===----------------------------------------------------------------------===//
// Custom Field IDs
//===----------------------------------------------------------------------===//

// The following functions implement custom field IDs specifically for the use
// in reset inference. They look much more like tracking fields on types than
// individual values. For example, vectors don't carry separate IDs for each of
// their elements. Instead they have one set of IDs for the entire vector, since
// the element type is uniform across all elements.

static unsigned getMaxFieldID(FIRRTLBaseType type) {
  return FIRRTLTypeSwitch<FIRRTLBaseType, unsigned>(type)
      .Case<BundleType>([](auto type) {
        unsigned id = 0;
        for (auto e : type.getElements())
          id += getMaxFieldID(e.type) + 1;
        return id;
      })
      .Case<FVectorType>(
          [](auto type) { return getMaxFieldID(type.getElementType()) + 1; })
      .Default([](auto) { return 0; });
}

static unsigned getFieldID(BundleType type, unsigned index) {
  assert(index < type.getNumElements());
  unsigned id = 1;
  for (unsigned i = 0; i < index; ++i)
    id += getMaxFieldID(type.getElementType(i)) + 1;
  return id;
}

static unsigned getFieldID(FVectorType type) { return 1; }

static unsigned getIndexForFieldID(BundleType type, unsigned fieldID) {
  assert(type.getNumElements() && "Bundle must have >0 fields");
  --fieldID;
  for (const auto &e : llvm::enumerate(type.getElements())) {
    auto numSubfields = getMaxFieldID(e.value().type) + 1;
    if (fieldID < numSubfields)
      return e.index();
    fieldID -= numSubfields;
  }
  assert(false && "field id outside bundle");
  return 0;
}

// If a field is pointing to a child of a zero-length vector, it is useless.
static bool isUselessVec(FIRRTLBaseType oldType, unsigned fieldID) {
  if (oldType.isGround()) {
    assert(fieldID == 0);
    return false;
  }

  // If this is a bundle type, recurse.
  if (auto bundleType = type_dyn_cast<BundleType>(oldType)) {
    unsigned index = getIndexForFieldID(bundleType, fieldID);
    return isUselessVec(bundleType.getElementType(index),
                        fieldID - getFieldID(bundleType, index));
  }

  // If this is a vector type, check if it is zero length.  Anything in a
  // zero-length vector is useless.
  if (auto vectorType = type_dyn_cast<FVectorType>(oldType)) {
    if (vectorType.getNumElements() == 0)
      return true;
    return isUselessVec(vectorType.getElementType(),
                        fieldID - getFieldID(vectorType));
  }

  return false;
}

// If a field is pointing to a child of a zero-length vector, it is useless.
static bool isUselessVec(FieldRef field) {
  return isUselessVec(
      getBaseType(type_cast<FIRRTLType>(field.getValue().getType())),
      field.getFieldID());
}

static bool getDeclName(Value value, SmallString<32> &string) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto module = cast<FModuleOp>(arg.getOwner()->getParentOp());
    string += module.getPortName(arg.getArgNumber());
    return true;
  }

  auto *op = value.getDefiningOp();
  return TypeSwitch<Operation *, bool>(op)
      .Case<InstanceOp, MemOp>([&](auto op) {
        string += op.getName();
        string += ".";
        string +=
            op.getPortName(cast<OpResult>(value).getResultNumber()).getValue();
        return true;
      })
      .Case<WireOp, RegOp, RegResetOp>([&](auto op) {
        string += op.getName();
        return true;
      })
      .Default([](auto) { return false; });
}

static bool getFieldName(const FieldRef &fieldRef, SmallString<32> &string) {
  SmallString<64> name;
  auto value = fieldRef.getValue();
  if (!getDeclName(value, string))
    return false;

  auto type = value.getType();
  auto localID = fieldRef.getFieldID();
  while (localID) {
    if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      auto index = getIndexForFieldID(bundleType, localID);
      // Add the current field string, and recurse into a subfield.
      auto &element = bundleType.getElements()[index];
      if (!string.empty())
        string += ".";
      string += element.name.getValue();
      // Recurse in to the element type.
      type = element.type;
      localID = localID - getFieldID(bundleType, index);
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      string += "[]";
      // Recurse in to the element type.
      type = vecType.getElementType();
      localID = localID - getFieldID(vecType);
    } else {
      // If we reach here, the field ref is pointing inside some aggregate type
      // that isn't a bundle or a vector. If the type is a ground type, then the
      // localID should be 0 at this point, and we should have broken from the
      // loop.
      llvm_unreachable("unsupported type");
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Reset Tracing
//===----------------------------------------------------------------------===//

/// Check whether a type contains a `ResetType`.
static bool typeContainsReset(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<FIRRTLType>([](auto type) {
        return type.getRecursiveTypeProperties().hasUninferredReset;
      })
      .Default([](auto) { return false; });
}

/// Iterate over a circuit and follow all signals with `ResetType`, aggregating
/// them into reset nets. After this function returns, the `resetMap` is
/// populated with the reset networks in the circuit, alongside information on
/// drivers and their types that contribute to the reset.
void InferResetsPass::traceResets(CircuitOp circuit) {
  LLVM_DEBUG(
      llvm::dbgs() << "\n===----- Tracing uninferred resets -----===\n\n");

  SmallVector<std::pair<FModuleOp, SmallVector<Operation *>>> moduleToOps;

  for (auto module : circuit.getOps<FModuleOp>())
    moduleToOps.push_back({module, {}});

  mlir::parallelForEach(circuit.getContext(), moduleToOps, [](auto &e) {
    e.first.walk([&](Operation *op) {
      // We are only interested in operations which are related to abstract
      // reset.
      if (llvm::any_of(
              op->getResultTypes(),
              [](mlir::Type type) { return typeContainsReset(type); }) ||
          llvm::any_of(op->getOperandTypes(), typeContainsReset))
        e.second.push_back(op);
    });
  });

  for (auto &[_, ops] : moduleToOps)
    for (auto *op : ops) {
      TypeSwitch<Operation *>(op)
          .Case<FConnectLike>([&](auto op) {
            traceResets(op.getDest(), op.getSrc(), op.getLoc());
          })
          .Case<InstanceOp>([&](auto op) { traceResets(op); })
          .Case<RefSendOp>([&](auto op) {
            // Trace using base types.
            traceResets(op.getType().getType(), op.getResult(), 0,
                        op.getBase().getType().getPassiveType(), op.getBase(),
                        0, op.getLoc());
          })
          .Case<RefResolveOp>([&](auto op) {
            // Trace using base types.
            traceResets(op.getType(), op.getResult(), 0,
                        op.getRef().getType().getType(), op.getRef(), 0,
                        op.getLoc());
          })
          .Case<Forceable>([&](Forceable op) {
            // Trace reset into rwprobe.  Avoid invalid IR.
            if (op.isForceable())
              traceResets(op.getDataType(), op.getData(), 0, op.getDataType(),
                          op.getDataRef(), 0, op.getLoc());
          })
          .Case<UninferredResetCastOp, ConstCastOp, RefCastOp>([&](auto op) {
            traceResets(op.getResult(), op.getInput(), op.getLoc());
          })
          .Case<InvalidValueOp>([&](auto op) {
            // Uniquify `InvalidValueOp`s that are contributing to multiple
            // reset networks. These are tricky to handle because passes
            // like CSE will generally ensure that there is only a single
            // `InvalidValueOp` per type. However, a `reset` invalid value
            // may be connected to two reset networks that end up being
            // inferred as `asyncreset` and `uint<1>`. In that case, we need
            // a distinct `InvalidValueOp` for each reset network in order
            // to assign it the correct type.
            auto type = op.getType();
            if (!typeContainsReset(type) || op->hasOneUse() || op->use_empty())
              return;
            LLVM_DEBUG(llvm::dbgs() << "Uniquify " << op << "\n");
            ImplicitLocOpBuilder builder(op->getLoc(), op);
            for (auto &use :
                 llvm::make_early_inc_range(llvm::drop_begin(op->getUses()))) {
              // - `make_early_inc_range` since `getUses()` is invalidated
              // upon
              //   `use.set(...)`.
              // - `drop_begin` such that the first use can keep the
              // original op.
              auto newOp = builder.create<InvalidValueOp>(type);
              use.set(newOp);
            }
          })

          .Case<SubfieldOp>([&](auto op) {
            // Associate the input bundle's resets with the output field's
            // resets.
            BundleType bundleType = op.getInput().getType();
            auto index = op.getFieldIndex();
            traceResets(op.getType(), op.getResult(), 0,
                        bundleType.getElements()[index].type, op.getInput(),
                        getFieldID(bundleType, index), op.getLoc());
          })

          .Case<SubindexOp, SubaccessOp>([&](auto op) {
            // Associate the input vector's resets with the output field's
            // resets.
            //
            // This collapses all elements in vectors into one shared
            // element which will ensure that reset inference provides a
            // uniform result for all elements.
            //
            // CAVEAT: This may infer reset networks that are too big, since
            // unrelated resets in the same vector end up looking as if they
            // were connected. However for the sake of type inference, this
            // is indistinguishable from them having to share the same type
            // (namely the vector element type).
            FVectorType vectorType = op.getInput().getType();
            traceResets(op.getType(), op.getResult(), 0,
                        vectorType.getElementType(), op.getInput(),
                        getFieldID(vectorType), op.getLoc());
          })

          .Case<RefSubOp>([&](RefSubOp op) {
            // Trace through ref.sub.
            auto aggType = op.getInput().getType().getType();
            uint64_t fieldID = TypeSwitch<FIRRTLBaseType, uint64_t>(aggType)
                                   .Case<FVectorType>([](auto type) {
                                     return getFieldID(type);
                                   })
                                   .Case<BundleType>([&](auto type) {
                                     return getFieldID(type, op.getIndex());
                                   });
            traceResets(op.getType(), op.getResult(), 0,
                        op.getResult().getType(), op.getInput(), fieldID,
                        op.getLoc());
          });
    }
}

/// Trace reset signals through an instance. This essentially associates the
/// instance's port values with the target module's port values.
void InferResetsPass::traceResets(InstanceOp inst) {
  // Lookup the referenced module. Nothing to do if its an extmodule.
  auto module = dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(inst));
  if (!module)
    return;
  LLVM_DEBUG(llvm::dbgs() << "Visiting instance " << inst.getName() << "\n");

  // Establish a connection between the instance ports and module ports.
  auto dirs = module.getPortDirections();
  for (const auto &it : llvm::enumerate(inst.getResults())) {
    auto dir = module.getPortDirection(it.index());
    Value dstPort = module.getArgument(it.index());
    Value srcPort = it.value();
    if (dir == Direction::Out)
      std::swap(dstPort, srcPort);
    traceResets(dstPort, srcPort, it.value().getLoc());
  }
}

/// Analyze a connect of one (possibly aggregate) value to another.
/// Each drive involving a `ResetType` is recorded.
void InferResetsPass::traceResets(Value dst, Value src, Location loc) {
  // Analyze the actual connection.
  traceResets(dst.getType(), dst, 0, src.getType(), src, 0, loc);
}

/// Analyze a connect of one (possibly aggregate) value to another.
/// Each drive involving a `ResetType` is recorded.
void InferResetsPass::traceResets(Type dstType, Value dst, unsigned dstID,
                                  Type srcType, Value src, unsigned srcID,
                                  Location loc) {
  if (auto dstBundle = type_dyn_cast<BundleType>(dstType)) {
    auto srcBundle = type_cast<BundleType>(srcType);
    for (unsigned dstIdx = 0, e = dstBundle.getNumElements(); dstIdx < e;
         ++dstIdx) {
      auto dstField = dstBundle.getElements()[dstIdx].name;
      auto srcIdx = srcBundle.getElementIndex(dstField);
      if (!srcIdx)
        continue;
      auto &dstElt = dstBundle.getElements()[dstIdx];
      auto &srcElt = srcBundle.getElements()[*srcIdx];
      if (dstElt.isFlip) {
        traceResets(srcElt.type, src, srcID + getFieldID(srcBundle, *srcIdx),
                    dstElt.type, dst, dstID + getFieldID(dstBundle, dstIdx),
                    loc);
      } else {
        traceResets(dstElt.type, dst, dstID + getFieldID(dstBundle, dstIdx),
                    srcElt.type, src, srcID + getFieldID(srcBundle, *srcIdx),
                    loc);
      }
    }
    return;
  }

  if (auto dstVector = type_dyn_cast<FVectorType>(dstType)) {
    auto srcVector = type_cast<FVectorType>(srcType);
    auto srcElType = srcVector.getElementType();
    auto dstElType = dstVector.getElementType();
    // Collapse all elements into one shared element. See comment in traceResets
    // above for some context. Note that we are directly passing on the field ID
    // of the vector itself as a stand-in for its element type. This is not
    // really what `FieldRef` is designed to do, but tends to work since all the
    // places that need to reason about the resulting weird IDs are inside this
    // file. Normally you would pick a specific index from the vector, which
    // would also move the field ID forward by some amount. However, we can't
    // distinguish individual elements for the sake of type inference *and* we
    // have to support zero-length vectors for which the only available ID is
    // the vector itself. Therefore we always just pick the vector itself for
    // the field ID and make sure in `updateType` that we handle vectors
    // accordingly.
    traceResets(dstElType, dst, dstID + getFieldID(dstVector), srcElType, src,
                srcID + getFieldID(srcVector), loc);
    return;
  }

  // Handle connecting ref's.  Other uses trace using base type.
  if (auto dstRef = type_dyn_cast<RefType>(dstType)) {
    auto srcRef = type_cast<RefType>(srcType);
    return traceResets(dstRef.getType(), dst, dstID, srcRef.getType(), src,
                       srcID, loc);
  }

  // Handle reset connections.
  auto dstBase = type_dyn_cast<FIRRTLBaseType>(dstType);
  auto srcBase = type_dyn_cast<FIRRTLBaseType>(srcType);
  if (!dstBase || !srcBase)
    return;
  if (!type_isa<ResetType>(dstBase) && !type_isa<ResetType>(srcBase))
    return;

  FieldRef dstField(dst, dstID);
  FieldRef srcField(src, srcID);
  LLVM_DEBUG(llvm::dbgs() << "Visiting driver '" << dstField << "' = '"
                          << srcField << "' (" << dstType << " = " << srcType
                          << ")\n");

  // Determine the leaders for the dst and src reset networks before we make
  // the connection. This will allow us to later detect if dst got merged
  // into src, or src into dst.
  ResetSignal dstLeader =
      *resetClasses.findLeader(resetClasses.insert({dstField, dstBase}));
  ResetSignal srcLeader =
      *resetClasses.findLeader(resetClasses.insert({srcField, srcBase}));

  // Unify the two reset networks.
  ResetSignal unionLeader = *resetClasses.unionSets(dstLeader, srcLeader);
  assert(unionLeader == dstLeader || unionLeader == srcLeader);

  // If dst got merged into src, append dst's drives to src's, or vice
  // versa. Also, remove dst's or src's entry in resetDrives, because they
  // will never come up as a leader again.
  if (dstLeader != srcLeader) {
    auto &unionDrives = resetDrives[unionLeader]; // needed before finds
    auto mergedDrivesIt =
        resetDrives.find(unionLeader == dstLeader ? srcLeader : dstLeader);
    if (mergedDrivesIt != resetDrives.end()) {
      unionDrives.append(mergedDrivesIt->second);
      resetDrives.erase(mergedDrivesIt);
    }
  }

  // Keep note of this drive so we can point the user at the right location
  // in case something goes wrong.
  resetDrives[unionLeader].push_back(
      {{dstField, dstBase}, {srcField, srcBase}, loc});
}

//===----------------------------------------------------------------------===//
// Reset Inference
//===----------------------------------------------------------------------===//

LogicalResult InferResetsPass::inferAndUpdateResets() {
  LLVM_DEBUG(llvm::dbgs() << "\n===----- Infer reset types -----===\n\n");
  for (auto it = resetClasses.begin(), end = resetClasses.end(); it != end;
       ++it) {
    if (!it->isLeader())
      continue;
    ResetNetwork net = llvm::make_range(resetClasses.member_begin(it),
                                        resetClasses.member_end());

    // Infer whether this should be a sync or async reset.
    auto kind = inferReset(net);
    if (failed(kind))
      return failure();

    // Update the types in the IR to match the inferred kind.
    if (failed(updateReset(net, *kind)))
      return failure();
  }
  return success();
}

FailureOr<ResetKind> InferResetsPass::inferReset(ResetNetwork net) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring reset network with "
                          << std::distance(net.begin(), net.end())
                          << " nodes\n");

  // Go through the nodes and track the involved types.
  unsigned asyncDrives = 0;
  unsigned syncDrives = 0;
  unsigned invalidDrives = 0;
  for (ResetSignal signal : net) {
    // Keep track of whether this signal contributes a vote for async or sync.
    if (type_isa<AsyncResetType>(signal.type))
      ++asyncDrives;
    else if (type_isa<UIntType>(signal.type))
      ++syncDrives;
    else if (isUselessVec(signal.field) ||
             isa_and_nonnull<InvalidValueOp>(
                 signal.field.getValue().getDefiningOp()))
      ++invalidDrives;
  }
  LLVM_DEBUG(llvm::dbgs() << "- Found " << asyncDrives << " async, "
                          << syncDrives << " sync, " << invalidDrives
                          << " invalid drives\n");

  // Handle the case where we have no votes for either kind.
  if (asyncDrives == 0 && syncDrives == 0 && invalidDrives == 0) {
    ResetSignal root = guessRoot(net);
    auto diag = mlir::emitError(root.field.getValue().getLoc())
                << "reset network never driven with concrete type";
    for (ResetSignal signal : net)
      diag.attachNote(signal.field.getLoc()) << "here: ";
    return failure();
  }

  // Handle the case where we have votes for both kinds.
  if (asyncDrives > 0 && syncDrives > 0) {
    ResetSignal root = guessRoot(net);
    bool majorityAsync = asyncDrives >= syncDrives;
    auto diag = mlir::emitError(root.field.getValue().getLoc())
                << "reset network";
    SmallString<32> fieldName;
    if (getFieldName(root.field, fieldName))
      diag << " \"" << fieldName << "\"";
    diag << " simultaneously connected to async and sync resets";
    diag.attachNote(root.field.getValue().getLoc())
        << "majority of connections to this reset are "
        << (majorityAsync ? "async" : "sync");
    for (auto &drive : getResetDrives(net)) {
      if ((type_isa<AsyncResetType>(drive.dst.type) && !majorityAsync) ||
          (type_isa<AsyncResetType>(drive.src.type) && !majorityAsync) ||
          (type_isa<UIntType>(drive.dst.type) && majorityAsync) ||
          (type_isa<UIntType>(drive.src.type) && majorityAsync))
        diag.attachNote(drive.loc)
            << (type_isa<AsyncResetType>(drive.src.type) ? "async" : "sync")
            << " drive here:";
    }
    return failure();
  }

  // At this point we know that the type of the reset is unambiguous. If there
  // are any votes for async, we make the reset async. Otherwise we make it
  // sync.
  auto kind = (asyncDrives ? ResetKind::Async : ResetKind::Sync);
  LLVM_DEBUG(llvm::dbgs() << "- Inferred as " << kind << "\n");
  return kind;
}

//===----------------------------------------------------------------------===//
// Reset Updating
//===----------------------------------------------------------------------===//

LogicalResult InferResetsPass::updateReset(ResetNetwork net, ResetKind kind) {
  LLVM_DEBUG(llvm::dbgs() << "Updating reset network with "
                          << std::distance(net.begin(), net.end())
                          << " nodes to " << kind << "\n");

  // Determine the final type the reset should have.
  FIRRTLBaseType resetType;
  if (kind == ResetKind::Async)
    resetType = AsyncResetType::get(&getContext());
  else
    resetType = UIntType::get(&getContext(), 1);

  // Update all those values in the network that cannot be inferred from
  // operands. If we change the type of a module port (i.e. BlockArgument), add
  // the module to a module worklist since we need to update its function type.
  SmallSetVector<Operation *, 16> worklist;
  SmallDenseSet<Operation *> moduleWorklist;
  SmallDenseSet<std::pair<Operation *, Operation *>> extmoduleWorklist;
  for (auto signal : net) {
    Value value = signal.field.getValue();
    if (!isa<BlockArgument>(value) &&
        !isa_and_nonnull<WireOp, RegOp, RegResetOp, InstanceOp, InvalidValueOp,
                         ConstCastOp, RefCastOp, UninferredResetCastOp>(
            value.getDefiningOp()))
      continue;
    if (updateReset(signal.field, resetType)) {
      for (auto user : value.getUsers())
        worklist.insert(user);
      if (auto blockArg = dyn_cast<BlockArgument>(value))
        moduleWorklist.insert(blockArg.getOwner()->getParentOp());
      else if (auto instOp = value.getDefiningOp<InstanceOp>()) {
        if (auto extmodule = dyn_cast<FExtModuleOp>(
                *instanceGraph->getReferencedModule(instOp)))
          extmoduleWorklist.insert({extmodule, instOp});
      } else if (auto uncast = value.getDefiningOp<UninferredResetCastOp>()) {
        uncast.replaceAllUsesWith(uncast.getInput());
        uncast.erase();
      }
    }
  }

  // Process the worklist of operations that have their type changed, pushing
  // types down the SSA dataflow graph. This is important because we change the
  // reset types in aggregates, and then need all the subindex, subfield, and
  // subaccess operations to be updated as appropriate.
  while (!worklist.empty()) {
    auto *wop = worklist.pop_back_val();
    SmallVector<Type, 2> types;
    if (auto op = dyn_cast<InferTypeOpInterface>(wop)) {
      // Determine the new result types.
      SmallVector<Type, 2> types;
      if (failed(op.inferReturnTypes(op->getContext(), op->getLoc(),
                                     op->getOperands(), op->getAttrDictionary(),
                                     op->getPropertiesStorage(),
                                     op->getRegions(), types)))
        return failure();

      // Update the results and add the changed ones to the
      // worklist.
      for (auto it : llvm::zip(op->getResults(), types)) {
        auto newType = std::get<1>(it);
        if (std::get<0>(it).getType() == newType)
          continue;
        std::get<0>(it).setType(newType);
        for (auto *user : std::get<0>(it).getUsers())
          worklist.insert(user);
      }
      LLVM_DEBUG(llvm::dbgs() << "- Inferred " << *op << "\n");
    } else if (auto uop = dyn_cast<UninferredResetCastOp>(wop)) {
      for (auto *user : uop.getResult().getUsers())
        worklist.insert(user);
      uop.replaceAllUsesWith(uop.getInput());
      LLVM_DEBUG(llvm::dbgs() << "- Inferred " << uop << "\n");
      uop.erase();
    }
  }

  // Update module types based on the type of the block arguments.
  for (auto *op : moduleWorklist) {
    auto module = dyn_cast<FModuleOp>(op);
    if (!module)
      continue;

    SmallVector<Attribute> argTypes;
    argTypes.reserve(module.getNumPorts());
    for (auto arg : module.getArguments())
      argTypes.push_back(TypeAttr::get(arg.getType()));

    module->setAttr(FModuleLike::getPortTypesAttrName(),
                    ArrayAttr::get(op->getContext(), argTypes));
    LLVM_DEBUG(llvm::dbgs()
               << "- Updated type of module '" << module.getName() << "'\n");
  }

  // Update extmodule types based on their instantiation.
  for (auto pair : extmoduleWorklist) {
    auto module = cast<FExtModuleOp>(pair.first);
    auto instOp = cast<InstanceOp>(pair.second);

    SmallVector<Attribute> types;
    for (auto type : instOp.getResultTypes())
      types.push_back(TypeAttr::get(type));

    module->setAttr(FModuleLike::getPortTypesAttrName(),
                    ArrayAttr::get(module->getContext(), types));
    LLVM_DEBUG(llvm::dbgs()
               << "- Updated type of extmodule '" << module.getName() << "'\n");
  }

  return success();
}

/// Update the type of a single field within a type.
static FIRRTLBaseType updateType(FIRRTLBaseType oldType, unsigned fieldID,
                                 FIRRTLBaseType fieldType) {
  // If this is a ground type, simply replace it, preserving constness.
  if (oldType.isGround()) {
    assert(fieldID == 0);
    return fieldType.getConstType(oldType.isConst());
  }

  // If this is a bundle type, update the corresponding field.
  if (auto bundleType = type_dyn_cast<BundleType>(oldType)) {
    unsigned index = getIndexForFieldID(bundleType, fieldID);
    SmallVector<BundleType::BundleElement> fields(bundleType.begin(),
                                                  bundleType.end());
    fields[index].type = updateType(
        fields[index].type, fieldID - getFieldID(bundleType, index), fieldType);
    return BundleType::get(oldType.getContext(), fields, bundleType.isConst());
  }

  // If this is a vector type, update the element type.
  if (auto vectorType = type_dyn_cast<FVectorType>(oldType)) {
    auto newType = updateType(vectorType.getElementType(),
                              fieldID - getFieldID(vectorType), fieldType);
    return FVectorType::get(newType, vectorType.getNumElements(),
                            vectorType.isConst());
  }

  llvm_unreachable("unknown aggregate type");
  return oldType;
}

/// Update the reset type of a specific field.
bool InferResetsPass::updateReset(FieldRef field, FIRRTLBaseType resetType) {
  // Compute the updated type.
  auto oldType = type_cast<FIRRTLType>(field.getValue().getType());
  FIRRTLType newType = mapBaseType(oldType, [&](auto base) {
    return updateType(base, field.getFieldID(), resetType);
  });

  // Update the type if necessary.
  if (oldType == newType)
    return false;
  LLVM_DEBUG(llvm::dbgs() << "- Updating '" << field << "' from " << oldType
                          << " to " << newType << "\n");
  field.getValue().setType(newType);
  return true;
}

//===----------------------------------------------------------------------===//
// Reset Annotations
//===----------------------------------------------------------------------===//

LogicalResult InferResetsPass::collectAnnos(CircuitOp circuit) {
  LLVM_DEBUG(
      llvm::dbgs() << "\n===----- Gather async reset annotations -----===\n\n");
  SmallVector<std::pair<FModuleOp, std::optional<Value>>> results;
  for (auto module : circuit.getOps<FModuleOp>())
    results.push_back({module, {}});
  // Collect annotations parallelly.
  if (failed(mlir::failableParallelForEach(
          circuit.getContext(), results, [&](auto &moduleAndResult) {
            auto result = collectAnnos(moduleAndResult.first);
            if (failed(result))
              return failure();
            moduleAndResult.second = *result;
            return success();
          })))
    return failure();

  for (auto [module, reset] : results)
    if (reset.has_value())
      annotatedResets.insert({module, *reset});
  return success();
}

FailureOr<std::optional<Value>>
InferResetsPass::collectAnnos(FModuleOp module) {
  bool anyFailed = false;
  SmallSetVector<std::pair<Annotation, Location>, 4> conflictingAnnos;

  // Consume a possible "ignore" annotation on the module itself, which
  // explicitly assigns it no reset domain.
  bool ignore = false;
  AnnotationSet moduleAnnos(module);
  if (!moduleAnnos.empty()) {
    moduleAnnos.removeAnnotations([&](Annotation anno) {
      if (anno.isClass(ignoreFullAsyncResetAnnoClass)) {
        ignore = true;
        conflictingAnnos.insert({anno, module.getLoc()});
        return true;
      }
      if (anno.isClass(fullAsyncResetAnnoClass)) {
        anyFailed = true;
        module.emitError("'FullAsyncResetAnnotation' cannot target module; "
                         "must target port or wire/node instead");
        return true;
      }
      return false;
    });
    moduleAnnos.applyToOperation(module);
  }
  if (anyFailed)
    return failure();

  // Consume any reset annotations on module ports.
  Value reset;
  AnnotationSet::removePortAnnotations(module, [&](unsigned argNum,
                                                   Annotation anno) {
    Value arg = module.getArgument(argNum);
    if (anno.isClass(fullAsyncResetAnnoClass)) {
      reset = arg;
      conflictingAnnos.insert({anno, reset.getLoc()});
      return true;
    }
    if (anno.isClass(ignoreFullAsyncResetAnnoClass)) {
      anyFailed = true;
      mlir::emitError(arg.getLoc(),
                      "'IgnoreFullAsyncResetAnnotation' cannot target port; "
                      "must target module instead");
      return true;
    }
    return false;
  });
  if (anyFailed)
    return failure();

  // Consume any reset annotations on wires in the module body.
  module.walk([&](Operation *op) {
    AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
      // Reset annotations must target wire/node ops.
      if (!isa<WireOp, NodeOp>(op)) {
        if (anno.isClass(fullAsyncResetAnnoClass,
                         ignoreFullAsyncResetAnnoClass)) {
          anyFailed = true;
          op->emitError(
              "reset annotations must target module, port, or wire/node");
          return true;
        }
        return false;
      }

      // At this point we know that we have a WireOp/NodeOp. Process the reset
      // annotations.
      if (anno.isClass(fullAsyncResetAnnoClass)) {
        reset = op->getResult(0);
        conflictingAnnos.insert({anno, reset.getLoc()});
        return true;
      }
      if (anno.isClass(ignoreFullAsyncResetAnnoClass)) {
        anyFailed = true;
        op->emitError(
            "'IgnoreFullAsyncResetAnnotation' cannot target wire/node; must "
            "target module instead");
        return true;
      }
      return false;
    });
  });
  if (anyFailed)
    return failure();

  // If we have found no annotations, there is nothing to do. We just leave
  // this module unannotated, which will cause it to inherit a reset domain
  // from its instantiation sites.
  if (!ignore && !reset) {
    LLVM_DEBUG(llvm::dbgs()
               << "No reset annotation for " << module.getName() << "\n");
    return std::optional<Value>();
  }

  // If we have found multiple annotations, emit an error and abort.
  if (conflictingAnnos.size() > 1) {
    auto diag = module.emitError("multiple reset annotations on module '")
                << module.getName() << "'";
    for (auto &annoAndLoc : conflictingAnnos)
      diag.attachNote(annoAndLoc.second)
          << "conflicting " << annoAndLoc.first.getClassAttr() << ":";
    return failure();
  }

  // Dump some information in debug builds.
  LLVM_DEBUG({
    llvm::dbgs() << "Annotated reset for " << module.getName() << ": ";
    if (ignore)
      llvm::dbgs() << "no domain\n";
    else if (auto arg = dyn_cast<BlockArgument>(reset))
      llvm::dbgs() << "port " << module.getPortName(arg.getArgNumber()) << "\n";
    else
      llvm::dbgs() << "wire "
                   << reset.getDefiningOp()->getAttrOfType<StringAttr>("name")
                   << "\n";
  });

  // Store the annotated reset for this module.
  assert(ignore || reset);
  return std::optional<Value>(reset);
}

//===----------------------------------------------------------------------===//
// Domain Construction
//===----------------------------------------------------------------------===//

/// Gather the reset domains present in a circuit. This traverses the instance
/// hierarchy of the design, making instances either live in a new reset
/// domain if so annotated, or inherit their parent's domain. This can go
/// wrong in some cases, mainly when a module is instantiated multiple times
/// within different reset domains.
LogicalResult InferResetsPass::buildDomains(CircuitOp circuit) {
  LLVM_DEBUG(
      llvm::dbgs() << "\n===----- Build async reset domains -----===\n\n");

  // Gather the domains.
  auto &instGraph = getAnalysis<InstanceGraph>();
  auto module = dyn_cast<FModuleOp>(*instGraph.getTopLevelNode()->getModule());
  if (!module) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping circuit because main module is no `firrtl.module`");
    return success();
  }
  buildDomains(module, InstancePathVec{}, Value{}, instGraph);

  // Report any domain conflicts among the modules.
  bool anyFailed = false;
  for (auto &it : domains) {
    auto module = cast<FModuleOp>(it.first);
    auto &domainConflicts = it.second;
    if (domainConflicts.size() <= 1)
      continue;

    anyFailed = true;
    SmallDenseSet<Value> printedDomainResets;
    auto diag = module.emitError("module '")
                << module.getName()
                << "' instantiated in different reset domains";
    for (auto &it : domainConflicts) {
      ResetDomain &domain = it.first;
      InstancePathRef path = it.second;
      auto inst = path.back();
      auto loc = path.empty() ? module.getLoc() : inst.getLoc();
      auto &note = diag.attachNote(loc);

      // Describe the instance itself.
      if (path.empty())
        note << "root instance";
      else {
        note << "instance '";
        llvm::interleave(
            path, [&](InstanceLike inst) { note << inst.getInstanceName(); },
            [&]() { note << "/"; });
        note << "'";
      }

      // Describe the reset domain the instance is in.
      note << " is in";
      if (domain.reset) {
        auto nameAndModule = getResetNameAndModule(domain.reset);
        note << " reset domain rooted at '" << nameAndModule.first.getValue()
             << "' of module '" << nameAndModule.second.getName() << "'";

        // Show where the domain reset is declared (once per reset).
        if (printedDomainResets.insert(domain.reset).second) {
          diag.attachNote(domain.reset.getLoc())
              << "reset domain '" << nameAndModule.first.getValue()
              << "' of module '" << nameAndModule.second.getName()
              << "' declared here:";
        }
      } else
        note << " no reset domain";
    }
  }
  return failure(anyFailed);
}

void InferResetsPass::buildDomains(FModuleOp module,
                                   const InstancePathVec &instPath,
                                   Value parentReset, InstanceGraph &instGraph,
                                   unsigned indent) {
  LLVM_DEBUG(llvm::dbgs().indent(indent * 2)
             << "Visiting " << getTail(instPath) << " (" << module.getName()
             << ")\n");

  // Assemble the domain for this module.
  ResetDomain domain(parentReset);
  auto it = annotatedResets.find(module);
  if (it != annotatedResets.end()) {
    domain.isTop = true;
    domain.reset = it->second;
  }

  // Associate the domain with this module. If the module already has an
  // associated domain, it must be identical. Otherwise we'll have to report
  // the conflicting domains to the user.
  auto &entries = domains[module];
  if (llvm::all_of(entries,
                   [&](const auto &entry) { return entry.first != domain; }))
    entries.push_back({domain, instPath});

  // Traverse the child instances.
  InstancePathVec childPath = instPath;
  for (auto *record : *instGraph[module]) {
    auto submodule = dyn_cast<FModuleOp>(*record->getTarget()->getModule());
    if (!submodule)
      continue;
    childPath.push_back(cast<InstanceLike>(*record->getInstance()));
    buildDomains(submodule, childPath, domain.reset, instGraph, indent + 1);
    childPath.pop_back();
  }
}

/// Determine how the reset for each module shall be implemented.
void InferResetsPass::determineImpl() {
  LLVM_DEBUG(
      llvm::dbgs() << "\n===----- Determine implementation -----===\n\n");
  for (auto &it : domains) {
    auto module = cast<FModuleOp>(it.first);
    auto &domain = it.second.back().first;
    determineImpl(module, domain);
  }
}

/// Determine how the reset for a module shall be implemented. This function
/// fills in the `existingValue`, `existingPort`, and `newPortName` fields of
/// the given reset domain.
///
/// Generally it does the following:
/// - If the domain has explicitly no reset ("ignore"), leaves everything
/// empty.
/// - If the domain is the place where the reset is defined ("top"), fills in
///   the existing port/wire/node as reset.
/// - If the module already has a port with the reset's name:
///   - If the type is `asyncreset`, reuses that port.
///   - Otherwise appends a `_N` suffix with increasing N to create a
///   yet-unused
///     port name, and marks that as to be created.
/// - Otherwise indicates that a port with the reset's name should be created.
///
void InferResetsPass::determineImpl(FModuleOp module, ResetDomain &domain) {
  if (!domain.reset)
    return; // nothing to do if the module needs no reset
  LLVM_DEBUG(llvm::dbgs() << "Planning reset for " << module.getName() << "\n");

  // If this is the root of a reset domain, we don't need to add any ports
  // and can just simply reuse the existing values.
  if (domain.isTop) {
    LLVM_DEBUG(llvm::dbgs() << "- Rooting at local value "
                            << getResetName(domain.reset) << "\n");
    domain.existingValue = domain.reset;
    if (auto blockArg = dyn_cast<BlockArgument>(domain.reset))
      domain.existingPort = blockArg.getArgNumber();
    return;
  }

  // Otherwise, check if a port with this name and type already exists and
  // reuse that where possible.
  auto neededName = getResetName(domain.reset);
  auto neededType = domain.reset.getType();
  LLVM_DEBUG(llvm::dbgs() << "- Looking for existing port " << neededName
                          << "\n");
  auto portNames = module.getPortNames();
  auto ports = llvm::zip(portNames, module.getArguments());
  auto portIt = llvm::find_if(
      ports, [&](auto port) { return std::get<0>(port) == neededName; });
  if (portIt != ports.end() && std::get<1>(*portIt).getType() == neededType) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Reusing existing port " << neededName << "\n");
    domain.existingValue = std::get<1>(*portIt);
    domain.existingPort = std::distance(ports.begin(), portIt);
    return;
  }

  // If we have found a port but the types don't match, pick a new name for
  // the reset port.
  //
  // CAVEAT: The Scala FIRRTL compiler just throws an error in this case. This
  // seems unnecessary though, since the compiler can just insert a new reset
  // signal as needed.
  if (portIt != ports.end()) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Existing " << neededName << " has incompatible type "
               << std::get<1>(*portIt).getType() << "\n");
    StringAttr newName;
    unsigned suffix = 0;
    do {
      newName =
          StringAttr::get(&getContext(), Twine(neededName.getValue()) +
                                             Twine("_") + Twine(suffix++));
    } while (llvm::is_contained(portNames, newName));
    LLVM_DEBUG(llvm::dbgs()
               << "- Creating uniquified port " << newName << "\n");
    domain.newPortName = newName;
    return;
  }

  // At this point we know that there is no such port, and we can safely
  // create one as needed.
  LLVM_DEBUG(llvm::dbgs() << "- Creating new port " << neededName << "\n");
  domain.newPortName = neededName;
}

//===----------------------------------------------------------------------===//
// Async Reset Implementation
//===----------------------------------------------------------------------===//

/// Implement the async resets gathered in the pass' `domains` map.
LogicalResult InferResetsPass::implementAsyncReset() {
  LLVM_DEBUG(llvm::dbgs() << "\n===----- Implement async resets -----===\n\n");
  for (auto &it : domains)
    if (failed(implementAsyncReset(cast<FModuleOp>(it.first),
                                   it.second.back().first)))
      return failure();
  return success();
}

/// Implement the async resets for a specific module.
///
/// This will add ports to the module as appropriate, update the register ops
/// in the module, and update any instantiated submodules with their
/// corresponding reset implementation details.
LogicalResult InferResetsPass::implementAsyncReset(FModuleOp module,
                                                   ResetDomain &domain) {
  LLVM_DEBUG(llvm::dbgs() << "Implementing async reset for " << module.getName()
                          << "\n");

  // Nothing to do if the module was marked explicitly with no reset domain.
  if (!domain.reset) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Skipping because module explicitly has no domain\n");
    return success();
  }

  // If needed, add a reset port to the module.
  Value actualReset = domain.existingValue;
  if (domain.newPortName) {
    PortInfo portInfo{domain.newPortName,
                      AsyncResetType::get(&getContext()),
                      Direction::In,
                      {},
                      domain.reset.getLoc()};
    module.insertPorts({{0, portInfo}});
    actualReset = module.getArgument(0);
    LLVM_DEBUG(llvm::dbgs()
               << "- Inserted port " << domain.newPortName << "\n");
  }
  assert(actualReset);
  LLVM_DEBUG({
    llvm::dbgs() << "- Using ";
    if (auto blockArg = dyn_cast<BlockArgument>(actualReset))
      llvm::dbgs() << "port #" << blockArg.getArgNumber() << " ";
    else
      llvm::dbgs() << "wire/node ";
    llvm::dbgs() << getResetName(actualReset) << "\n";
  });

  // Gather a list of operations in the module that need to be updated with
  // the new reset.
  SmallVector<Operation *> opsToUpdate;
  module.walk([&](Operation *op) {
    if (isa<InstanceOp, RegOp, RegResetOp>(op))
      opsToUpdate.push_back(op);
  });

  // If the reset is a local wire or node, move it upwards such that it
  // dominates all the operations that it will need to attach to. In the case
  // of a node this might not be easily possible, so we just spill into a wire
  // in that case.
  if (!isa<BlockArgument>(actualReset)) {
    mlir::DominanceInfo dom(module);
    // The first op in `opsToUpdate` is the top-most op in the module, since
    // the ops and blocks are traversed in a depth-first, top-to-bottom order
    // in `walk`. So we can simply check if the local reset declaration is
    // before the first op to find out if we need to move anything.
    auto *resetOp = actualReset.getDefiningOp();
    if (!opsToUpdate.empty() && !dom.dominates(resetOp, opsToUpdate[0])) {
      LLVM_DEBUG(llvm::dbgs()
                 << "- Reset doesn't dominate all uses, needs to be moved\n");

      // If the node can't be moved because its input doesn't dominate the
      // target location, convert it to a wire.
      auto nodeOp = dyn_cast<NodeOp>(resetOp);
      if (nodeOp && !dom.dominates(nodeOp.getInput(), opsToUpdate[0])) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Promoting node to wire for move: " << nodeOp << "\n");
        ImplicitLocOpBuilder builder(nodeOp.getLoc(), nodeOp);
        auto wireOp = builder.create<WireOp>(
            nodeOp.getResult().getType(), nodeOp.getNameAttr(),
            nodeOp.getNameKindAttr(), nodeOp.getAnnotationsAttr(),
            nodeOp.getInnerSymAttr(), nodeOp.getForceableAttr());
        builder.create<StrictConnectOp>(wireOp.getResult(), nodeOp.getInput());
        nodeOp->replaceAllUsesWith(wireOp);
        nodeOp.erase();
        resetOp = wireOp;
        actualReset = wireOp.getResult();
        domain.existingValue = wireOp.getResult();
      }

      // Determine the block into which the reset declaration needs to be
      // moved.
      Block *targetBlock = dom.findNearestCommonDominator(
          resetOp->getBlock(), opsToUpdate[0]->getBlock());
      LLVM_DEBUG({
        if (targetBlock != resetOp->getBlock())
          llvm::dbgs() << "- Needs to be moved to different block\n";
      });

      // At this point we have to figure out in front of which operation in
      // the target block the reset declaration has to be moved. The reset
      // declaration and the first op it needs to dominate may be buried
      // inside blocks of other operations (e.g. `WhenOp`), so we have to look
      // through their parent operations until we find the one that lies
      // within the target block.
      auto getParentInBlock = [](Operation *op, Block *block) {
        while (op && op->getBlock() != block)
          op = op->getParentOp();
        return op;
      };
      auto *resetOpInTarget = getParentInBlock(resetOp, targetBlock);
      auto *firstOpInTarget = getParentInBlock(opsToUpdate[0], targetBlock);

      // Move the operation upwards. Since there are situations where the
      // reset declaration does not dominate the first use, but the `WhenOp`
      // it is nested within actually *does* come before that use, we have to
      // consider moving the reset declaration in front of its parent op.
      if (resetOpInTarget->isBeforeInBlock(firstOpInTarget))
        resetOp->moveBefore(resetOpInTarget);
      else
        resetOp->moveBefore(firstOpInTarget);
    }
  }

  // Update the operations.
  for (auto *op : opsToUpdate)
    implementAsyncReset(op, module, actualReset);

  return success();
}

/// Modify an operation in a module to implement an async reset for that
/// module.
void InferResetsPass::implementAsyncReset(Operation *op, FModuleOp module,
                                          Value actualReset) {
  ImplicitLocOpBuilder builder(op->getLoc(), op);

  // Handle instances.
  if (auto instOp = dyn_cast<InstanceOp>(op)) {
    // Lookup the reset domain of the instantiated module. If there is no
    // reset domain associated with that module, or the module is explicitly
    // marked as being in no domain, simply skip.
    auto refModule =
        dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instOp));
    if (!refModule)
      return;
    auto domainIt = domains.find(refModule);
    if (domainIt == domains.end())
      return;
    auto &domain = domainIt->second.back().first;
    if (!domain.reset)
      return;
    LLVM_DEBUG(llvm::dbgs()
               << "- Update instance '" << instOp.getName() << "'\n");

    // If needed, add a reset port to the instance.
    Value instReset;
    if (domain.newPortName) {
      LLVM_DEBUG(llvm::dbgs() << "  - Adding new result as reset\n");

      auto newInstOp = instOp.cloneAndInsertPorts(
          {{/*portIndex=*/0,
            {domain.newPortName,
             type_cast<FIRRTLBaseType>(actualReset.getType()),
             Direction::In}}});
      instReset = newInstOp.getResult(0);

      // Update the uses over to the new instance and drop the old instance.
      instOp.replaceAllUsesWith(newInstOp.getResults().drop_front());
      instanceGraph->replaceInstance(instOp, newInstOp);
      instOp->erase();
      instOp = newInstOp;
    } else if (domain.existingPort.has_value()) {
      auto idx = *domain.existingPort;
      instReset = instOp.getResult(idx);
      LLVM_DEBUG(llvm::dbgs() << "  - Using result #" << idx << " as reset\n");
    }

    // If there's no reset port on the instance to connect, we're done. This
    // can happen if the instantiated module has a reset domain, but that
    // domain is e.g. rooted at an internal wire.
    if (!instReset)
      return;

    // Connect the instance's reset to the actual reset.
    assert(instReset && actualReset);
    builder.setInsertionPointAfter(instOp);
    builder.create<StrictConnectOp>(instReset, actualReset);
    return;
  }

  // Handle reset-less registers.
  if (auto regOp = dyn_cast<RegOp>(op)) {
    if (AnnotationSet::removeAnnotations(regOp, excludeMemToRegAnnoClass))
      return;

    LLVM_DEBUG(llvm::dbgs() << "- Adding async reset to " << regOp << "\n");
    auto zero = createZeroValue(builder, regOp.getResult().getType());
    auto newRegOp = builder.create<RegResetOp>(
        regOp.getResult().getType(), regOp.getClockVal(), actualReset, zero,
        regOp.getNameAttr(), regOp.getNameKindAttr(), regOp.getAnnotations(),
        regOp.getInnerSymAttr(), regOp.getForceableAttr());
    regOp.getResult().replaceAllUsesWith(newRegOp.getResult());
    if (regOp.getForceable())
      regOp.getRef().replaceAllUsesWith(newRegOp.getRef());
    regOp->erase();
    return;
  }

  // Handle registers with reset.
  if (auto regOp = dyn_cast<RegResetOp>(op)) {
    // If the register already has an async reset, leave it untouched.
    if (type_isa<AsyncResetType>(regOp.getResetSignal().getType())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "- Skipping (has async reset) " << regOp << "\n");
      // The following performs the logic of `CheckResets` in the original
      // Scala source code.
      if (failed(regOp.verifyInvariants()))
        signalPassFailure();
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "- Updating reset of " << regOp << "\n");

    auto reset = regOp.getResetSignal();
    auto value = regOp.getResetValue();

    // If we arrive here, the register has a sync reset. In order to add an
    // async reset, we have to move the sync reset into a mux in front of the
    // register.
    insertResetMux(builder, regOp.getResult(), reset, value);
    builder.setInsertionPointAfterValue(regOp.getResult());
    auto mux = builder.create<MuxPrimOp>(reset, value, regOp.getResult());
    emitConnect(builder, regOp.getResult(), mux);

    // Replace the existing reset with the async reset.
    builder.setInsertionPoint(regOp);
    auto zero = createZeroValue(builder, regOp.getResult().getType());
    regOp.getResetSignalMutable().assign(actualReset);
    regOp.getResetValueMutable().assign(zero);
  }
}

LogicalResult InferResetsPass::verifyNoAbstractReset() {
  bool hasAbstractResetPorts = false;
  for (FModuleLike module :
       getOperation().getBodyBlock()->getOps<FModuleLike>()) {
    for (PortInfo port : module.getPorts()) {
      if (getBaseOfType<ResetType>(port.type)) {
        auto diag = emitError(port.loc)
                    << "a port \"" << port.getName()
                    << "\" with abstract reset type was unable to be "
                       "inferred by InferResets (is this a top-level port?)";
        diag.attachNote(module->getLoc())
            << "the module with this uninferred reset port was defined here";
        hasAbstractResetPorts = true;
      }
    }
  }

  if (hasAbstractResetPorts)
    return failure();
  return success();
}
