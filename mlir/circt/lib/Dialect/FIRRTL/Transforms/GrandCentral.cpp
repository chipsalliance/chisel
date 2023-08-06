//===- GrandCentral.cpp - Ingest black box sources --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Implement SiFive's Grand Central transform.  Currently, this supports
// SystemVerilog Interface generation.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/YAMLTraits.h"
#include <variant>

#define DEBUG_TYPE "gct"

using namespace circt;
using namespace firrtl;
using hw::HWModuleLike;

//===----------------------------------------------------------------------===//
// Collateral for generating a YAML representation of a SystemVerilog interface
//===----------------------------------------------------------------------===//

namespace {

// These macros are used to provide hard-errors if a user tries to use the YAML
// infrastructure improperly.  We only implement conversion to YAML and not
// conversion from YAML.  The LLVM YAML infrastructure doesn't provide the
// ability to differentitate this and we don't need it for the purposes of
// Grand Central.
#define UNIMPLEMENTED_DEFAULT(clazz)                                           \
  llvm_unreachable("default '" clazz                                           \
                   "' construction is an intentionally *NOT* implemented "     \
                   "YAML feature (you should never be using this)");
#define UNIMPLEMENTED_DENORM(clazz)                                            \
  llvm_unreachable("conversion from YAML to a '" clazz                         \
                   "' is intentionally *NOT* implemented (you should not be "  \
                   "converting from YAML to an interface)");

// This namespace provides YAML-related collateral that is specific to Grand
// Central and should not be placed in the `llvm::yaml` namespace.
namespace yaml {

/// Context information necessary for YAML generation.
struct Context {
  /// A symbol table consisting of _only_ the interfaces construted by the Grand
  /// Central pass.  This is not a symbol table because we do not have an
  /// up-to-date symbol table that includes interfaces at the time the Grand
  /// Central pass finishes.  This structure is easier to build up and is only
  /// the information we need.
  DenseMap<Attribute, sv::InterfaceOp> &interfaceMap;
};

/// A representation of an `sv::InterfaceSignalOp` that includes additional
/// description information.
///
/// TODO: This could be removed if we add `firrtl.DocStringAnnotation` support
/// or if FIRRTL dialect included support for ops to specify "comment"
/// information.
struct DescribedSignal {
  /// The comment associated with this signal.
  StringAttr description;

  /// The signal.
  sv::InterfaceSignalOp signal;
};

/// This exist to work around the fact that no interface can be instantiated
/// inside another interface.  This serves to represent an op like this for the
/// purposes of conversion to YAML.
///
/// TODO: Fix this once we have a solution for #1464.
struct DescribedInstance {
  StringAttr name;

  /// A comment associated with the interface instance.
  StringAttr description;

  /// The dimensionality of the interface instantiation.
  ArrayAttr dimensions;

  /// The symbol associated with the interface.
  FlatSymbolRefAttr interface;
};

} // namespace yaml
} // namespace

// These macros tell the YAML infrastructure that these are types which can
// show up in vectors and provides implementations of how to serialize these.
// Each of these macros puts the resulting class into the `llvm::yaml` namespace
// (which is why these are outside the `llvm::yaml` namespace below).
LLVM_YAML_IS_SEQUENCE_VECTOR(::yaml::DescribedSignal)
LLVM_YAML_IS_SEQUENCE_VECTOR(::yaml::DescribedInstance)
LLVM_YAML_IS_SEQUENCE_VECTOR(sv::InterfaceOp)

// This `llvm::yaml` namespace contains implementations of classes that enable
// conversion from an `sv::InterfaceOp` to a YAML representation of that
// interface using [LLVM's YAML I/O library](https://llvm.org/docs/YamlIO.html).
namespace llvm {
namespace yaml {

using namespace ::yaml;

/// Convert newlines and comments to remove the comments.  This produces better
/// looking YAML output.  E.g., this will convert the following:
///
///   // foo
///   // bar
///
/// Into the following:
///
///   foo
///   bar
std::string static stripComment(StringRef str) {
  std::string descriptionString;
  llvm::raw_string_ostream stream(descriptionString);
  SmallVector<StringRef> splits;
  str.split(splits, "\n");
  llvm::interleave(
      splits,
      [&](auto substr) {
        substr.consume_front("//");
        stream << substr.drop_while([](auto c) { return c == ' '; });
      },
      [&]() { stream << "\n"; });
  return descriptionString;
}

/// Conversion from a `DescribedSignal` to YAML.  This is
/// implemented using YAML normalization to first convert this to an internal
/// `Field` structure which has a one-to-one mapping to the YAML represntation.
template <>
struct MappingContextTraits<DescribedSignal, Context> {
  /// A one-to-one representation with a YAML representation of a signal/field.
  struct Field {
    /// The name of the field.
    StringRef name;

    /// An optional, textual description of what the field is.
    std::optional<std::string> description;

    /// The dimensions of the field.
    SmallVector<unsigned, 2> dimensions;

    /// The width of the underlying type.
    unsigned width;

    /// Construct a `Field` from a `DescribedSignal` (an `sv::InterfaceSignalOp`
    /// with an optional description).
    Field(IO &io, DescribedSignal &op)
        : name(op.signal.getSymNameAttr().getValue()) {

      // Convert the description from a `StringAttr` (which may be null) to an
      // `optional<StringRef>`.  This aligns exactly with the YAML
      // representation.
      if (op.description)
        description = stripComment(op.description.getValue());

      // Unwrap the type of the field into an array of dimensions and a width.
      // By example, this is going from the following hardware type:
      //
      //     !hw.uarray<1xuarray<2xuarray<3xi8>>>
      //
      // To the following representation:
      //
      //     dimensions: [ 3, 2, 1 ]
      //     width: 8
      //
      // Note that the above is equivalenet to the following Verilog
      // specification.
      //
      //     wire [7:0] foo [2:0][1:0][0:0]
      //
      // Do this by repeatedly unwrapping unpacked array types until you get to
      // the underlying type.  The dimensions need to be reversed as this
      // unwrapping happens in reverse order of the final representation.
      auto tpe = op.signal.getType();
      while (auto vector = tpe.dyn_cast<hw::UnpackedArrayType>()) {
        dimensions.push_back(vector.getSize());
        tpe = vector.getElementType();
      }
      dimensions = SmallVector<unsigned>(llvm::reverse(dimensions));

      // The final non-array type must be an integer.  Leave this as an assert
      // with a blind cast because we generated this type in this pass (and we
      // therefore cannot fail this cast).
      assert(isa<IntegerType>(tpe));
      width = type_cast<IntegerType>(tpe).getWidth();
    }

    /// A no-argument constructor is necessary to work with LLVM's YAML library.
    Field(IO &io){UNIMPLEMENTED_DEFAULT("Field")}

    /// This cannot be denomralized back to an interface op.
    DescribedSignal denormalize(IO &) {
      UNIMPLEMENTED_DENORM("DescribedSignal")
    }
  };

  static void mapping(IO &io, DescribedSignal &op, Context &ctx) {
    MappingNormalization<Field, DescribedSignal> keys(io, op);
    io.mapRequired("name", keys->name);
    io.mapOptional("description", keys->description);
    io.mapRequired("dimensions", keys->dimensions);
    io.mapRequired("width", keys->width);
  }
};

/// Conversion from a `DescribedInstance` to YAML.  This is implemented using
/// YAML normalization to first convert the `DescribedInstance` to an internal
/// `Instance` struct which has a one-to-one representation with the final YAML
/// representation.
template <>
struct MappingContextTraits<DescribedInstance, Context> {
  /// A YAML-serializable representation of an interface instantiation.
  struct Instance {
    /// The name of the interface.
    StringRef name;

    /// An optional textual description of the interface.
    std::optional<std::string> description = std::nullopt;

    /// An array describing the dimnensionality of the interface.
    SmallVector<int64_t, 2> dimensions;

    /// The underlying interface.
    FlatSymbolRefAttr interface;

    Instance(IO &io, DescribedInstance &op)
        : name(op.name.getValue()), interface(op.interface) {

      // Convert the description from a `StringAttr` (which may be null) to an
      // `optional<StringRef>`.  This aligns exactly with the YAML
      // representation.
      if (op.description)
        description = stripComment(op.description.getValue());

      for (auto &d : op.dimensions) {
        auto dimension = dyn_cast<IntegerAttr>(d);
        dimensions.push_back(dimension.getInt());
      }
    }

    Instance(IO &io){UNIMPLEMENTED_DEFAULT("Instance")}

    DescribedInstance denormalize(IO &) {
      UNIMPLEMENTED_DENORM("DescribedInstance")
    }
  };

  static void mapping(IO &io, DescribedInstance &op, Context &ctx) {
    MappingNormalization<Instance, DescribedInstance> keys(io, op);
    io.mapRequired("name", keys->name);
    io.mapOptional("description", keys->description);
    io.mapRequired("dimensions", keys->dimensions);
    io.mapRequired("interface", ctx.interfaceMap[keys->interface], ctx);
  }
};

/// Conversion from an `sv::InterfaceOp` to YAML.  This is implemented using
/// YAML normalization to first convert the interface to an internal `Interface`
/// which reformats the Grand Central-generated interface into the YAML format.
template <>
struct MappingContextTraits<sv::InterfaceOp, Context> {
  /// A YAML-serializable representation of an interface.  This consists of
  /// fields (vector or ground types) and nested interfaces.
  struct Interface {
    /// The name of the interface.
    StringRef name;

    /// All ground or vectors that make up the interface.
    std::vector<DescribedSignal> fields;

    /// Instantiations of _other_ interfaces.
    std::vector<DescribedInstance> instances;

    /// Construct an `Interface` from an `sv::InterfaceOp`.  This is tuned to
    /// "parse" the structure of an interface that the Grand Central pass
    /// generates.  The structure of `Field`s and `Instance`s is documented
    /// below.
    ///
    /// A field will look like the following.  The verbatim description is
    /// optional:
    ///
    ///     sv.verbatim "// <description>" {
    ///       firrtl.grandcentral.yaml.type = "description",
    ///       symbols = []}
    ///     sv.interface.signal @<name> : <type>
    ///
    /// An interface instanctiation will look like the following.  The verbatim
    /// description is optional.
    ///
    ///     sv.verbatim "// <description>" {
    ///       firrtl.grandcentral.type = "description",
    ///       symbols = []}
    ///     sv.verbatim "<name> <symbol>();" {
    ///       firrtl.grandcentral.yaml.name = "<name>",
    ///       firrtl.grandcentral.yaml.dimensions = [<first dimension>, ...],
    ///       firrtl.grandcentral.yaml.symbol = @<symbol>,
    ///       firrtl.grandcentral.yaml.type = "instance",
    ///       symbols = []}
    ///
    Interface(IO &io, sv::InterfaceOp &op) : name(op.getName()) {
      // A mutable store of the description.  This occurs in the op _before_ the
      // field or instance, so we need someplace to put it until we use it.
      StringAttr description = {};

      for (auto &op : op.getBodyBlock()->getOperations()) {
        TypeSwitch<Operation *>(&op)
            // A verbatim op is either a description or an interface
            // instantiation.
            .Case<sv::VerbatimOp>([&](sv::VerbatimOp op) {
              auto tpe = op->getAttrOfType<StringAttr>(
                  "firrtl.grandcentral.yaml.type");

              // This is a descripton.  Update the mutable description and
              // continue;
              if (tpe.getValue() == "description") {
                description = op.getFormatStringAttr();
                return;
              }

              // This is an unsupported construct. Just drop it.
              if (tpe.getValue() == "unsupported") {
                description = {};
                return;
              }

              // This is an instance of another interface.  Add the symbol to
              // the vector of instances.
              auto name = op->getAttrOfType<StringAttr>(
                  "firrtl.grandcentral.yaml.name");
              auto dimensions = op->getAttrOfType<ArrayAttr>(
                  "firrtl.grandcentral.yaml.dimensions");
              auto symbol = op->getAttrOfType<FlatSymbolRefAttr>(
                  "firrtl.grandcentral.yaml.symbol");
              instances.push_back(
                  DescribedInstance({name, description, dimensions, symbol}));
              description = {};
            })
            // An interface signal op is a field.
            .Case<sv::InterfaceSignalOp>([&](sv::InterfaceSignalOp op) {
              fields.push_back(DescribedSignal({description, op}));
              description = {};
            });
      }
    }

    /// A no-argument constructor is necessary to work with LLVM's YAML library.
    Interface(IO &io){UNIMPLEMENTED_DEFAULT("Interface")}

    /// This cannot be denomralized back to an interface op.
    sv::InterfaceOp denormalize(IO &) {
      UNIMPLEMENTED_DENORM("sv::InterfaceOp")
    }
  };

  static void mapping(IO &io, sv::InterfaceOp &op, Context &ctx) {
    MappingNormalization<Interface, sv::InterfaceOp> keys(io, op);
    io.mapRequired("name", keys->name);
    io.mapRequired("fields", keys->fields, ctx);
    io.mapRequired("instances", keys->instances, ctx);
  }
};

} // namespace yaml
} // namespace llvm

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

/// A helper to build verbatim strings with symbol placeholders. Provides a
/// mechanism to snapshot the current string and symbols and restore back to
/// this state after modifications. These snapshots are particularly useful when
/// the string is assembled through hierarchical traversal of some sort, which
/// populates the string with a prefix common to all children of a hierarchy
/// (like the interface field traversal in the `GrandCentralPass`).
///
/// The intended use is as follows:
///
///     void baz(VerbatimBuilder &v) {
///       foo(v.snapshot().append("bar"));
///     }
///
/// The function `baz` takes a snapshot of the current verbatim text `v`, adds
/// "bar" to it and calls `foo` with that appended verbatim text. After the call
/// to `foo` returns, any changes made by `foo` as well as the "bar" are dropped
/// from the verbatim text `v`, as the temporary snapshot goes out of scope.
struct VerbatimBuilder {
  struct Base {
    SmallString<128> string;
    SmallVector<Attribute> symbols;
    VerbatimBuilder builder() { return VerbatimBuilder(*this); }
    operator VerbatimBuilder() { return builder(); }
  };

  /// Constructing a builder will snapshot the `Base` which holds the actual
  /// string and symbols.
  VerbatimBuilder(Base &base)
      : base(base), stringBaseSize(base.string.size()),
        symbolsBaseSize(base.symbols.size()) {}

  /// Destroying a builder will reset the `Base` to the original string and
  /// symbols.
  ~VerbatimBuilder() {
    base.string.resize(stringBaseSize);
    base.symbols.resize(symbolsBaseSize);
  }

  // Disallow copying.
  VerbatimBuilder(const VerbatimBuilder &) = delete;
  VerbatimBuilder &operator=(const VerbatimBuilder &) = delete;

  /// Take a snapshot of the current string and symbols. This returns a new
  /// `VerbatimBuilder` that will reset to the current state of the string once
  /// destroyed.
  VerbatimBuilder snapshot() { return VerbatimBuilder(base); }

  /// Get the current string.
  StringRef getString() const { return base.string; }
  /// Get the current symbols;
  ArrayRef<Attribute> getSymbols() const { return base.symbols; }

  /// Append to the string.
  VerbatimBuilder &append(char c) {
    base.string.push_back(c);
    return *this;
  }

  /// Append to the string.
  VerbatimBuilder &append(const Twine &twine) {
    twine.toVector(base.string);
    return *this;
  }

  /// Append a placeholder and symbol to the string.
  VerbatimBuilder &append(Attribute symbol) {
    unsigned id = base.symbols.size();
    base.symbols.push_back(symbol);
    append("{{" + Twine(id) + "}}");
    return *this;
  }

  VerbatimBuilder &operator+=(char c) { return append(c); }
  VerbatimBuilder &operator+=(const Twine &twine) { return append(twine); }
  VerbatimBuilder &operator+=(Attribute symbol) { return append(symbol); }

private:
  Base &base;
  size_t stringBaseSize;
  size_t symbolsBaseSize;
};

/// A wrapper around a string that is used to encode a type which cannot be
/// represented by an mlir::Type for some reason.  This is currently used to
/// represent either an interface, a n-dimensional vector of interfaces, or a
/// tombstone for an actually unsupported type (e.g., an AugmentedBooleanType).
struct VerbatimType {
  /// The textual representation of the type.
  std::string str;

  /// True if this is a type which must be "instatiated" and requires a trailing
  /// "()".
  bool instantiation;

  /// A vector storing the width of each dimension of the type.
  SmallVector<int32_t, 4> dimensions = {};

  /// Serialize this type to a string.
  std::string toStr(StringRef name) {
    SmallString<64> stringType(str);
    stringType.append(" ");
    stringType.append(name);
    for (auto d : llvm::reverse(dimensions)) {
      stringType.append("[");
      stringType.append(Twine(d).str());
      stringType.append("]");
    }
    if (instantiation)
      stringType.append("()");
    stringType.append(";");
    return std::string(stringType);
  }
};

/// A sum type representing either a type encoded as a string (VerbatimType)
/// or an actual mlir::Type.
typedef std::variant<VerbatimType, Type> TypeSum;

/// Stores the information content of an ExtractGrandCentralAnnotation.
struct ExtractionInfo {
  /// The directory where Grand Central generated collateral (modules,
  /// interfaces, etc.) will be written.
  StringAttr directory = {};

  /// The name of the file where any binds will be written.  This will be placed
  /// in the same output area as normal compilation output, e.g., output
  /// Verilog.  This has no relation to the `directory` member.
  StringAttr bindFilename = {};
};

/// Stores information about the companion module of a GrandCentral view.
struct CompanionInfo {
  StringRef name;

  FModuleOp companion;
  bool isNonlocal;
};

/// Stores a reference to a ground type and an optional NLA associated with
/// that field.
struct FieldAndNLA {
  FieldRef field;
  FlatSymbolRefAttr nlaSym;
};

/// Stores the arguments required to construct the verbatim xmr assignment.
struct VerbatimXMRbuilder {
  Value val;
  StringAttr str;
  ArrayAttr syms;
  FModuleOp companionMod;
  VerbatimXMRbuilder(Value val, StringAttr str, ArrayAttr syms,
                     FModuleOp companionMod)
      : val(val), str(str), syms(syms), companionMod(companionMod) {}
};

/// Stores the arguments required to construct the InterfaceOps and
/// InterfaceSignalOps.
struct InterfaceElemsBuilder {
  StringAttr iFaceName;
  IntegerAttr id;
  struct Properties {
    StringAttr description;
    StringAttr elemName;
    TypeSum elemType;
    Properties(StringAttr des, StringAttr name, TypeSum &elemType)
        : description(des), elemName(name), elemType(elemType) {}
  };
  SmallVector<Properties> elementsList;
  InterfaceElemsBuilder(StringAttr iFaceName, IntegerAttr id)
      : iFaceName(iFaceName), id(id) {}
};

/// Generate SystemVerilog interfaces from Grand Central annotations.  This pass
/// roughly works in the following three phases:
///
/// 1. Extraction information is determined.
///
/// 2. The circuit is walked to find all scattered annotations related to Grand
///    Central interfaces.  These are: (a) the companion module and (b) all
///    leaves that are to be connected to the interface.
///
/// 3. The circuit-level Grand Central annotation is walked to both generate and
///    instantiate interfaces and to generate the "mappings" file that produces
///    cross-module references (XMRs) to drive the interface.
struct GrandCentralPass : public GrandCentralBase<GrandCentralPass> {
  GrandCentralPass(bool instantiateCompanionOnlyFlag) {
    instantiateCompanionOnly = instantiateCompanionOnlyFlag;
  }

  void runOnOperation() override;

private:
  /// Optionally build an AugmentedType from an attribute.  Return none if the
  /// attribute is not a dictionary or if it does not match any of the known
  /// templates for AugmentedTypes.
  std::optional<Attribute> fromAttr(Attribute attr);

  /// Mapping of ID to leaf ground type and an optional non-local annotation
  /// associated with that ID.
  DenseMap<Attribute, FieldAndNLA> leafMap;

  /// Mapping of ID to companion module.
  DenseMap<Attribute, CompanionInfo> companionIDMap;

  /// An optional prefix applied to all interfaces in the design.  This is set
  /// based on a PrefixInterfacesAnnotation.
  StringRef interfacePrefix;

  NLATable *nlaTable;

  /// The design-under-test (DUT) as determined by the presence of a
  /// "sifive.enterprise.firrtl.MarkDUTAnnotation".  This will be null if no DUT
  /// was found.
  FModuleOp dut;

  /// An optional directory for testbench-related files.  This is null if no
  /// "TestBenchDirAnnotation" is found.
  StringAttr testbenchDir;

  /// Return a string containing the name of an interface.  Apply correct
  /// prefixing from the interfacePrefix and module-level prefix parameter.
  std::string getInterfaceName(StringAttr prefix,
                               AugmentedBundleTypeAttr bundleType) {

    if (prefix)
      return (prefix.getValue() + interfacePrefix +
              bundleType.getDefName().getValue())
          .str();
    return (interfacePrefix + bundleType.getDefName().getValue()).str();
  }

  /// Recursively examine an AugmentedType to populate the "mappings" file
  /// (generate XMRs) for this interface.  This does not build new interfaces.
  bool traverseField(Attribute field, IntegerAttr id, VerbatimBuilder &path,
                     SmallVector<VerbatimXMRbuilder> &xmrElems,
                     SmallVector<InterfaceElemsBuilder> &interfaceBuilder);

  /// Recursively examine an AugmentedType to both build new interfaces and
  /// populate a "mappings" file (generate XMRs) using `traverseField`.  Return
  /// the type of the field exmained.
  std::optional<TypeSum>
  computeField(Attribute field, IntegerAttr id, StringAttr prefix,
               VerbatimBuilder &path, SmallVector<VerbatimXMRbuilder> &xmrElems,
               SmallVector<InterfaceElemsBuilder> &interfaceBuilder);

  /// Recursively examine an AugmentedBundleType to both build new interfaces
  /// and populate a "mappings" file (generate XMRs).  Return none if the
  /// interface is invalid.
  std::optional<StringAttr>
  traverseBundle(AugmentedBundleTypeAttr bundle, IntegerAttr id,
                 StringAttr prefix, VerbatimBuilder &path,
                 SmallVector<VerbatimXMRbuilder> &xmrElems,
                 SmallVector<InterfaceElemsBuilder> &interfaceBuilder);

  /// Return the module associated with this value.
  HWModuleLike getEnclosingModule(Value value, FlatSymbolRefAttr sym = {});

  /// Inforamtion about how the circuit should be extracted.  This will be
  /// non-empty if an extraction annotation is found.
  std::optional<ExtractionInfo> maybeExtractInfo = std::nullopt;

  /// A filename describing where to put a YAML representation of the
  /// interfaces generated by this pass.
  std::optional<StringAttr> maybeHierarchyFileYAML = std::nullopt;

  StringAttr getOutputDirectory() {
    if (maybeExtractInfo)
      return maybeExtractInfo->directory;
    return {};
  }

  /// Store of an instance paths analysis.  This is constructed inside
  /// `runOnOperation`, to work around the deleted copy constructor of
  /// `InstancePathCache`'s internal `BumpPtrAllocator`.
  ///
  /// TODO: Investigate a way to not use a pointer here like how `getNamespace`
  /// works below.
  InstancePathCache *instancePaths = nullptr;

  /// The namespace associated with the circuit.  This is lazily constructed
  /// using `getNamesapce`.
  std::optional<CircuitNamespace> circuitNamespace;

  /// The module namespaces. These are lazily constructed by
  /// `getModuleNamespace`.
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;

  /// Return a reference to the circuit namespace.  This will lazily construct a
  /// namespace if one does not exist.
  CircuitNamespace &getNamespace() {
    if (!circuitNamespace)
      circuitNamespace = CircuitNamespace(getOperation());
    return *circuitNamespace;
  }

  /// Get the cached namespace for a module.
  ModuleNamespace &getModuleNamespace(FModuleLike module) {
    auto it = moduleNamespaces.find(module);
    if (it != moduleNamespaces.end())
      return it->second;
    return moduleNamespaces.insert({module, ModuleNamespace(module)})
        .first->second;
  }

  /// A symbol table associated with the circuit.  This is lazily constructed by
  /// `getSymbolTable`.
  std::optional<SymbolTable *> symbolTable;

  /// Return a reference to a circuit-level symbol table.  Lazily construct one
  /// if such a symbol table does not already exist.
  SymbolTable &getSymbolTable() {
    if (!symbolTable)
      symbolTable = &getAnalysis<SymbolTable>();
    return **symbolTable;
  }

  // Utility that acts like emitOpError, but does _not_ include a note.  The
  // note in emitOpError includes the entire op which means the **ENTIRE**
  // FIRRTL circuit.  This doesn't communicate anything useful to the user
  // other than flooding their terminal.
  InFlightDiagnostic emitCircuitError(StringRef message = {}) {
    return emitError(getOperation().getLoc(), "'firrtl.circuit' op " + message);
  }

  // Insert comment delimiters ("// ") after newlines in the description string.
  // This is necessary to prevent introducing invalid verbatim Verilog.
  //
  // TODO: Add a comment op and lower the description to that.
  // TODO: Tracking issue: https://github.com/llvm/circt/issues/1677
  std::string cleanupDescription(StringRef description) {
    StringRef head;
    SmallString<64> out;
    do {
      std::tie(head, description) = description.split("\n");
      out.append(head);
      if (!description.empty())
        out.append("\n// ");
    } while (!description.empty());
    return std::string(out);
  }

  /// A store of the YAML representation of interfaces.
  DenseMap<Attribute, sv::InterfaceOp> interfaceMap;

  /// Returns an operation's `inner_sym`, adding one if necessary.
  StringAttr getOrAddInnerSym(Operation *op);

  /// Returns a port's `inner_sym`, adding one if necessary.
  StringAttr getOrAddInnerSym(FModuleLike module, size_t portIdx);
};

} // namespace

//===----------------------------------------------------------------------===//
// Code related to handling Grand Central View annotations
//===----------------------------------------------------------------------===//

/// Recursively walk a sifive.enterprise.grandcentral.AugmentedType to extract
/// any annotations it may contain.  This is going to generate two types of
/// annotations:
///   1) Annotations necessary to build interfaces and store them at "~"
///   2) Scattered annotations for how components bind to interfaces
static std::optional<DictionaryAttr>
parseAugmentedType(ApplyState &state, DictionaryAttr augmentedType,
                   DictionaryAttr root, StringRef companion, StringAttr name,
                   StringAttr defName, std::optional<IntegerAttr> id,
                   std::optional<StringAttr> description, Twine clazz,
                   StringAttr companionAttr, Twine path = {}) {

  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();

  /// Optionally unpack a ReferenceTarget encoded as a DictionaryAttr.  Return
  /// either a pair containing the Target string (up to the reference) and an
  /// array of components or none if the input is malformed.  The input
  /// DicionaryAttr encoding is a JSON object of a serialized ReferenceTarget
  /// Scala class.  By example, this is converting:
  ///   ~Foo|Foo>a.b[0]
  /// To:
  ///   {"~Foo|Foo>a", {".b", "[0]"}}
  /// The format of a ReferenceTarget object like:
  ///   circuit: String
  ///   module: String
  ///   path: Seq[(Instance, OfModule)]
  ///   ref: String
  ///   component: Seq[TargetToken]
  auto refToTarget =
      [&](DictionaryAttr refTarget) -> std::optional<std::string> {
    auto circuitAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "circuit", loc, clazz, path);
    auto moduleAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "module", loc, clazz, path);
    auto pathAttr =
        tryGetAs<ArrayAttr>(refTarget, refTarget, "path", loc, clazz, path);
    auto componentAttr = tryGetAs<ArrayAttr>(refTarget, refTarget, "component",
                                             loc, clazz, path);
    if (!circuitAttr || !moduleAttr || !pathAttr || !componentAttr)
      return {};

    // Parse non-local annotations.
    SmallString<32> strpath;
    for (auto p : pathAttr) {
      auto dict = dyn_cast_or_null<DictionaryAttr>(p);
      if (!dict) {
        mlir::emitError(loc, "annotation '" + clazz +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      auto instHolder =
          tryGetAs<DictionaryAttr>(dict, dict, "_1", loc, clazz, path);
      auto modHolder =
          tryGetAs<DictionaryAttr>(dict, dict, "_2", loc, clazz, path);
      if (!instHolder || !modHolder) {
        mlir::emitError(loc, "annotation '" + clazz +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      auto inst = tryGetAs<StringAttr>(instHolder, instHolder, "value", loc,
                                       clazz, path);
      auto mod =
          tryGetAs<StringAttr>(modHolder, modHolder, "value", loc, clazz, path);
      if (!inst || !mod) {
        mlir::emitError(loc, "annotation '" + clazz +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      strpath += "/" + inst.getValue().str() + ":" + mod.getValue().str();
    }

    SmallVector<Attribute> componentAttrs;
    SmallString<32> componentStr;
    for (size_t i = 0, e = componentAttr.size(); i != e; ++i) {
      auto cPath = (path + ".component[" + Twine(i) + "]").str();
      auto component = componentAttr[i];
      auto dict = dyn_cast_or_null<DictionaryAttr>(component);
      if (!dict) {
        mlir::emitError(loc, "annotation '" + clazz + "' with path '" + cPath +
                                 " has invalid type (expected DictionaryAttr)");
        return {};
      }
      auto classAttr =
          tryGetAs<StringAttr>(dict, refTarget, "class", loc, clazz, cPath);
      if (!classAttr)
        return {};

      auto value = dict.get("value");

      // A subfield like "bar" in "~Foo|Foo>foo.bar".
      if (auto field = dyn_cast<StringAttr>(value)) {
        assert(classAttr.getValue() == "firrtl.annotations.TargetToken$Field" &&
               "A StringAttr target token must be found with a subfield target "
               "token.");
        componentStr.append((Twine(".") + field.getValue()).str());
        continue;
      }

      // A subindex like "42" in "~Foo|Foo>foo[42]".
      if (auto index = dyn_cast<IntegerAttr>(value)) {
        assert(classAttr.getValue() == "firrtl.annotations.TargetToken$Index" &&
               "An IntegerAttr target token must be found with a subindex "
               "target token.");
        componentStr.append(
            (Twine("[") + Twine(index.getValue().getZExtValue()) + "]").str());
        continue;
      }

      mlir::emitError(loc,
                      "Annotation '" + clazz + "' with path '" + cPath +
                          ".value has unexpected type (should be StringAttr "
                          "for subfield  or IntegerAttr for subindex).")
              .attachNote()
          << "The value received was: " << value << "\n";
      return {};
    }

    auto refAttr =
        tryGetAs<StringAttr>(refTarget, refTarget, "ref", loc, clazz, path);

    return (Twine("~" + circuitAttr.getValue() + "|" + moduleAttr.getValue() +
                  strpath + ">" + refAttr.getValue()) +
            componentStr)
        .str();
  };

  auto classAttr =
      tryGetAs<StringAttr>(augmentedType, root, "class", loc, clazz, path);
  if (!classAttr)
    return std::nullopt;
  StringRef classBase = classAttr.getValue();
  if (!classBase.consume_front("sifive.enterprise.grandcentral.Augmented")) {
    mlir::emitError(loc,
                    "the 'class' was expected to start with "
                    "'sifive.enterprise.grandCentral.Augmented*', but was '" +
                        classAttr.getValue() + "' (Did you misspell it?)")
            .attachNote()
        << "see annotation: " << augmentedType;
    return std::nullopt;
  }

  // An AugmentedBundleType looks like:
  //   "defName": String
  //   "elements": Seq[AugmentedField]
  if (classBase == "BundleType") {
    defName =
        tryGetAs<StringAttr>(augmentedType, root, "defName", loc, clazz, path);
    if (!defName)
      return std::nullopt;

    // Each element is an AugmentedField with members:
    //   "name": String
    //   "description": Option[String]
    //   "tpe": AugmenetedType
    SmallVector<Attribute> elements;
    auto elementsAttr =
        tryGetAs<ArrayAttr>(augmentedType, root, "elements", loc, clazz, path);
    if (!elementsAttr)
      return std::nullopt;
    for (size_t i = 0, e = elementsAttr.size(); i != e; ++i) {
      auto field = dyn_cast_or_null<DictionaryAttr>(elementsAttr[i]);
      if (!field) {
        mlir::emitError(
            loc,
            "Annotation '" + Twine(clazz) + "' with path '.elements[" +
                Twine(i) +
                "]' contained an unexpected type (expected a DictionaryAttr).")
                .attachNote()
            << "The received element was: " << elementsAttr[i] << "\n";
        return std::nullopt;
      }
      auto ePath = (path + ".elements[" + Twine(i) + "]").str();
      auto name = tryGetAs<StringAttr>(field, root, "name", loc, clazz, ePath);
      auto tpe =
          tryGetAs<DictionaryAttr>(field, root, "tpe", loc, clazz, ePath);
      std::optional<StringAttr> description;
      if (auto maybeDescription = field.get("description"))
        description = cast<StringAttr>(maybeDescription);
      auto eltAttr = parseAugmentedType(
          state, tpe, root, companion, name, defName, std::nullopt, description,
          clazz, companionAttr, path + "_" + name.getValue());
      if (!name || !tpe || !eltAttr)
        return std::nullopt;

      // Collect information necessary to build a module with this view later.
      // This includes the optional description and name.
      NamedAttrList attrs;
      if (auto maybeDescription = field.get("description"))
        attrs.append("description", cast<StringAttr>(maybeDescription));
      attrs.append("name", name);
      attrs.append("tpe", tpe.getAs<StringAttr>("class"));
      elements.push_back(*eltAttr);
    }
    // Add an annotation that stores information necessary to construct the
    // module for the view.  This needs the name of the module (defName) and the
    // names of the components inside it.
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    attrs.append("defName", defName);
    if (description)
      attrs.append("description", *description);
    attrs.append("elements", ArrayAttr::get(context, elements));
    if (id)
      attrs.append("id", *id);
    attrs.append("name", name);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // An AugmentedGroundType looks like:
  //   "ref": ReferenceTarget
  //   "tpe": GroundType
  // The ReferenceTarget is not serialized to a string.  The GroundType will
  // either be an actual FIRRTL ground type or a GrandCentral uninferred type.
  // This can be ignored for us.
  if (classBase == "GroundType") {
    auto maybeTarget = refToTarget(augmentedType.getAs<DictionaryAttr>("ref"));
    if (!maybeTarget) {
      mlir::emitError(loc, "Failed to parse ReferenceTarget").attachNote()
          << "See the full Annotation here: " << root;
      return std::nullopt;
    }

    auto id = state.newID();

    auto target = *maybeTarget;

    NamedAttrList elementIface, elementScattered;

    // Populate the annotation for the interface element.
    elementIface.append("class", classAttr);
    if (description)
      elementIface.append("description", *description);
    elementIface.append("id", id);
    elementIface.append("name", name);
    // Populate an annotation that will be scattered onto the element.
    elementScattered.append("class", classAttr);
    elementScattered.append("id", id);
    // If there are sub-targets, then add these.
    auto targetAttr = StringAttr::get(context, target);
    auto xmrSrcTarget = resolvePath(targetAttr.getValue(), state.circuit,
                                    state.symTbl, state.targetCaches);
    if (!xmrSrcTarget) {
      mlir::emitError(loc, "Failed to resolve target ") << targetAttr;
      return std::nullopt;
    }

    // Determine the source for this Wiring Problem.  The source is the value
    // that will be eventually by read from, via cross-module reference, to
    // drive this element of the SystemVerilog Interface.
    auto sourceRef = xmrSrcTarget->ref;
    ImplicitLocOpBuilder builder(sourceRef.getOp()->getLoc(), context);
    std::optional<Value> source =
        TypeSwitch<Operation *, std::optional<Value>>(sourceRef.getOp())
            // The target is an external module port.  The source is the
            // instance port of this singly-instantiated external module.
            .Case<FExtModuleOp>([&](FExtModuleOp extMod)
                                    -> std::optional<Value> {
              auto portNo = sourceRef.getImpl().getPortNo();
              if (xmrSrcTarget->instances.empty()) {
                auto paths = state.instancePathCache.getAbsolutePaths(extMod);
                if (paths.size() > 1) {
                  extMod.emitError(
                      "cannot resolve a unique instance path from the "
                      "external module '")
                      << targetAttr << "'";
                  return std::nullopt;
                }
                auto *it = xmrSrcTarget->instances.begin();
                for (auto inst : paths.back()) {
                  xmrSrcTarget->instances.insert(it, cast<InstanceOp>(inst));
                  ++it;
                }
              }
              auto lastInst = xmrSrcTarget->instances.pop_back_val();
              builder.setInsertionPointAfter(lastInst);
              return getValueByFieldID(builder, lastInst.getResult(portNo),
                                       xmrSrcTarget->fieldIdx);
            })
            // The target is a module port.  The source is the port _inside_
            // that module.
            .Case<FModuleOp>([&](FModuleOp module) -> std::optional<Value> {
              builder.setInsertionPointToEnd(module.getBodyBlock());
              auto portNum = sourceRef.getImpl().getPortNo();
              return getValueByFieldID(builder, module.getArgument(portNum),
                                       xmrSrcTarget->fieldIdx);
            })
            // The target is something else.
            .Default([&](Operation *op) -> std::optional<Value> {
              auto module = cast<FModuleOp>(sourceRef.getModule());
              builder.setInsertionPointToEnd(module.getBodyBlock());
              auto is = dyn_cast<hw::InnerSymbolOpInterface>(op);
              // Resolve InnerSymbol references to their target result.
              if (is && is.getTargetResult())
                return getValueByFieldID(builder, is.getTargetResult(),
                                         xmrSrcTarget->fieldIdx);
              if (sourceRef.getOp()->getNumResults() != 1) {
                op->emitOpError()
                    << "cannot be used as a target of the Grand Central View \""
                    << defName.getValue()
                    << "\" because it does not have exactly one result";
                return std::nullopt;
              }
              return getValueByFieldID(builder, sourceRef.getOp()->getResult(0),
                                       xmrSrcTarget->fieldIdx);
            });

    // Exit if there was an error in the source.
    if (!source)
      return std::nullopt;

    // Compute the sink of this Wiring Problem.  The final sink will eventually
    // be a SystemVerilog Interface.  However, this cannot exist until the
    // GrandCentral pass runs.  Create an undriven WireOp and use that as the
    // sink.  The WireOp will be driven later when the Wiring Problem is
    // resolved. Apply the scattered element annotation to this directly to save
    // having to reprocess this in LowerAnnotations.
    auto companionMod =
        cast<FModuleOp>(resolvePath(companionAttr.getValue(), state.circuit,
                                    state.symTbl, state.targetCaches)
                            ->ref.getOp());
    builder.setInsertionPointToEnd(companionMod.getBodyBlock());
    auto sink = builder.create<WireOp>(source->getType(), name);
    state.targetCaches.insertOp(sink);
    AnnotationSet annotations(context);
    annotations.addAnnotations(
        {DictionaryAttr::getWithSorted(context, elementScattered)});
    annotations.applyToOperation(sink);

    // Append this new Wiring Problem to the ApplyState.  The Wiring Problem
    // will be resolved to bore RefType ports before LowerAnnotations finishes.
    state.wiringProblems.push_back({*source, sink.getResult(),
                                    (path + "__bore").str(),
                                    WiringProblem::RefTypeUsage::Prefer});

    return DictionaryAttr::getWithSorted(context, elementIface);
  }

  // An AugmentedVectorType looks like:
  //   "elements": Seq[AugmentedType]
  if (classBase == "VectorType") {
    auto elementsAttr =
        tryGetAs<ArrayAttr>(augmentedType, root, "elements", loc, clazz, path);
    if (!elementsAttr)
      return std::nullopt;
    SmallVector<Attribute> elements;
    for (auto [i, elt] : llvm::enumerate(elementsAttr)) {
      auto eltAttr = parseAugmentedType(
          state, cast<DictionaryAttr>(elt), root, companion, name,
          StringAttr::get(context, ""), id, std::nullopt, clazz, companionAttr,
          path + "_" + Twine(i));
      if (!eltAttr)
        return std::nullopt;
      elements.push_back(*eltAttr);
    }
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    if (description)
      attrs.append("description", *description);
    attrs.append("elements", ArrayAttr::get(context, elements));
    attrs.append("name", name);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // Any of the following are known and expected, but are legacy AugmentedTypes
  // do not have a target:
  //   - AugmentedStringType
  //   - AugmentedBooleanType
  //   - AugmentedIntegerType
  //   - AugmentedDoubleType
  bool isIgnorable =
      llvm::StringSwitch<bool>(classBase)
          .Cases("StringType", "BooleanType", "IntegerType", "DoubleType", true)
          .Default(false);
  if (isIgnorable) {
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    attrs.append("name", name);
    auto value =
        tryGetAs<Attribute>(augmentedType, root, "value", loc, clazz, path);
    if (!value)
      return std::nullopt;
    attrs.append("value", value);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // Anything else is unexpected or a user error if they manually wrote
  // annotations.  Print an error and error out.
  mlir::emitError(loc, "found unknown AugmentedType '" + classAttr.getValue() +
                           "' (Did you misspell it?)")
          .attachNote()
      << "see annotation: " << augmentedType;
  return std::nullopt;
}

LogicalResult circt::firrtl::applyGCTView(const AnnoPathValue &target,
                                          DictionaryAttr anno,
                                          ApplyState &state) {

  auto id = state.newID();
  auto *context = state.circuit.getContext();
  auto loc = state.circuit.getLoc();
  NamedAttrList companionAttrs;
  companionAttrs.append("class", StringAttr::get(context, companionAnnoClass));
  companionAttrs.append("id", id);
  auto viewAttr =
      tryGetAs<DictionaryAttr>(anno, anno, "view", loc, viewAnnoClass);
  if (!viewAttr)
    return failure();
  auto name = tryGetAs<StringAttr>(anno, anno, "name", loc, viewAnnoClass);
  if (!name)
    return failure();
  companionAttrs.append("name", name);
  auto companionAttr =
      tryGetAs<StringAttr>(anno, anno, "companion", loc, viewAnnoClass);
  if (!companionAttr)
    return failure();
  companionAttrs.append("target", companionAttr);
  state.addToWorklistFn(DictionaryAttr::get(context, companionAttrs));

  auto prunedAttr =
      parseAugmentedType(state, viewAttr, anno, companionAttr.getValue(), name,
                         {}, id, {}, viewAnnoClass, companionAttr, Twine(name));
  if (!prunedAttr)
    return failure();

  AnnotationSet annotations(state.circuit);
  annotations.addAnnotations({*prunedAttr});
  annotations.applyToOperation(state.circuit);

  return success();
}

//===----------------------------------------------------------------------===//
// GrandCentralPass Implementation
//===----------------------------------------------------------------------===//

std::optional<Attribute> GrandCentralPass::fromAttr(Attribute attr) {
  auto dict = dyn_cast<DictionaryAttr>(attr);
  if (!dict) {
    emitCircuitError() << "attribute is not a dictionary: " << attr << "\n";
    return std::nullopt;
  }

  auto clazz = dict.getAs<StringAttr>("class");
  if (!clazz) {
    emitCircuitError() << "missing 'class' key in " << dict << "\n";
    return std::nullopt;
  }

  auto classBase = clazz.getValue();
  classBase.consume_front("sifive.enterprise.grandcentral.Augmented");

  if (classBase == "BundleType") {
    if (dict.getAs<StringAttr>("defName") && dict.getAs<ArrayAttr>("elements"))
      return AugmentedBundleTypeAttr::get(&getContext(), dict);
    emitCircuitError() << "has an invalid AugmentedBundleType that does not "
                          "contain 'defName' and 'elements' fields: "
                       << dict;
  } else if (classBase == "VectorType") {
    if (dict.getAs<StringAttr>("name") && dict.getAs<ArrayAttr>("elements"))
      return AugmentedVectorTypeAttr::get(&getContext(), dict);
    emitCircuitError() << "has an invalid AugmentedVectorType that does not "
                          "contain 'name' and 'elements' fields: "
                       << dict;
  } else if (classBase == "GroundType") {
    auto id = dict.getAs<IntegerAttr>("id");
    auto name = dict.getAs<StringAttr>("name");
    if (id && leafMap.count(id) && name)
      return AugmentedGroundTypeAttr::get(&getContext(), dict);
    if (!id || !name)
      emitCircuitError() << "has an invalid AugmentedGroundType that does not "
                            "contain 'id' and 'name' fields:  "
                         << dict;
    if (id && !leafMap.count(id))
      emitCircuitError() << "has an AugmentedGroundType with 'id == "
                         << id.getValue().getZExtValue()
                         << "' that does not have a scattered leaf to connect "
                            "to in the circuit "
                            "(was the leaf deleted or constant prop'd away?)";
  } else if (classBase == "StringType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedStringTypeAttr::get(&getContext(), dict);
  } else if (classBase == "BooleanType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedBooleanTypeAttr::get(&getContext(), dict);
  } else if (classBase == "IntegerType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedIntegerTypeAttr::get(&getContext(), dict);
  } else if (classBase == "DoubleType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedDoubleTypeAttr::get(&getContext(), dict);
  } else if (classBase == "LiteralType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedLiteralTypeAttr::get(&getContext(), dict);
  } else if (classBase == "DeletedType") {
    if (auto name = dict.getAs<StringAttr>("name"))
      return AugmentedDeletedTypeAttr::get(&getContext(), dict);
  } else {
    emitCircuitError() << "has an invalid AugmentedType";
  }
  return std::nullopt;
}

bool GrandCentralPass::traverseField(
    Attribute field, IntegerAttr id, VerbatimBuilder &path,
    SmallVector<VerbatimXMRbuilder> &xmrElems,
    SmallVector<InterfaceElemsBuilder> &interfaceBuilder) {
  return TypeSwitch<Attribute, bool>(field)
      .Case<AugmentedGroundTypeAttr>([&](AugmentedGroundTypeAttr ground) {
        auto [fieldRef, sym] = leafMap.lookup(ground.getID());
        hw::HierPathOp nla;
        if (sym)
          nla = nlaTable->getNLA(sym.getAttr());
        Value leafValue = fieldRef.getValue();
        assert(leafValue && "leafValue not found");

        auto companionModule = companionIDMap.lookup(id).companion;
        HWModuleLike enclosing = getEnclosingModule(leafValue, sym);

        auto tpe = type_cast<FIRRTLBaseType>(leafValue.getType());

        // If the type is zero-width then do not emit an XMR.
        if (!tpe.getBitWidthOrSentinel())
          return true;

        // The leafValue is assumed to conform to a very specific pattern:
        //
        //   1) The leaf value is in the companion.
        //   2) The leaf value is a NodeOp
        //
        // Anything else means that there is an error or the IR is somehow using
        // "old-style" Annotations to encode a Grand Central View.  This
        // _really_ should be impossible to hit given that LowerAnnotations must
        // generate code that conforms to the check here.
        auto *nodeOp = leafValue.getDefiningOp();
        if (companionModule != enclosing) {
          auto diag = companionModule->emitError()
                      << "Grand Central View \""
                      << companionIDMap.lookup(id).name
                      << "\" is invalid because a leaf is not inside the "
                         "companion module";
          diag.attachNote(leafValue.getLoc())
              << "the leaf value is declared here";
          if (nodeOp) {
            auto leafModule = nodeOp->getParentOfType<FModuleOp>();
            diag.attachNote(leafModule.getLoc())
                << "the leaf value is inside this module";
          }
          return false;
        }

        if (!isa<NodeOp>(nodeOp)) {
          emitError(leafValue.getLoc())
              << "Grand Central View \"" << companionIDMap.lookup(id).name
              << "\" has an invalid leaf value (this must be a node)";
          return false;
        }

        /// Increment all the indices inside `{{`, `}}` by one. This is to
        /// indicate that a value is added to the `substitutions` of the
        /// verbatim op, other than the symbols.
        auto getStrAndIncrementIds = [&](StringRef base) -> StringAttr {
          SmallString<128> replStr;
          StringRef begin = "{{";
          StringRef end = "}}";
          // The replacement string.
          size_t from = 0;
          while (from < base.size()) {
            // Search for the first `{{` and `}}`.
            size_t beginAt = base.find(begin, from);
            size_t endAt = base.find(end, from);
            // If not found, then done.
            if (beginAt == StringRef::npos || endAt == StringRef::npos ||
                (beginAt > endAt)) {
              replStr.append(base.substr(from));
              break;
            }
            // Copy the string as is, until the `{{`.
            replStr.append(base.substr(from, beginAt - from));
            // Advance `from` to the character after the `}}`.
            from = endAt + 2;
            auto idChar = base.substr(beginAt + 2, endAt - beginAt - 2);
            int idNum;
            bool failed = idChar.getAsInteger(10, idNum);
            (void)failed;
            assert(!failed && "failed to parse integer from verbatim string");
            // Now increment the id and append.
            replStr.append("{{");
            Twine(idNum + 1).toVector(replStr);
            replStr.append("}}");
          }
          return StringAttr::get(&getContext(), "assign " + replStr + ";");
        };

        // This is the new style of XMRs using RefTypes.  The value subsitution
        // index is set to -1, as it will be incremented when generating the
        // string.
        // Generate the path from the LCA to the module that contains the leaf.
        path += " = {{-1}}";
        AnnotationSet::removeDontTouch(nodeOp);
        // Assemble the verbatim op.
        xmrElems.emplace_back(
            nodeOp->getOperand(0), getStrAndIncrementIds(path.getString()),
            ArrayAttr::get(&getContext(), path.getSymbols()), companionModule);
        return true;
      })
      .Case<AugmentedVectorTypeAttr>([&](auto vector) {
        bool notFailed = true;
        auto elements = vector.getElements();
        for (size_t i = 0, e = elements.size(); i != e; ++i) {
          auto field = fromAttr(elements[i]);
          if (!field)
            return false;
          notFailed &= traverseField(
              *field, id, path.snapshot().append("[" + Twine(i) + "]"),
              xmrElems, interfaceBuilder);
        }
        return notFailed;
      })
      .Case<AugmentedBundleTypeAttr>([&](AugmentedBundleTypeAttr bundle) {
        bool anyFailed = true;
        for (auto element : bundle.getElements()) {
          auto field = fromAttr(element);
          if (!field)
            return false;
          auto name = cast<DictionaryAttr>(element).getAs<StringAttr>("name");
          if (!name)
            name = cast<DictionaryAttr>(element).getAs<StringAttr>("defName");
          anyFailed &= traverseField(
              *field, id, path.snapshot().append("." + name.getValue()),
              xmrElems, interfaceBuilder);
        }

        return anyFailed;
      })
      .Case<AugmentedStringTypeAttr>([&](auto a) { return false; })
      .Case<AugmentedBooleanTypeAttr>([&](auto a) { return false; })
      .Case<AugmentedIntegerTypeAttr>([&](auto a) { return false; })
      .Case<AugmentedDoubleTypeAttr>([&](auto a) { return false; })
      .Case<AugmentedLiteralTypeAttr>([&](auto a) { return false; })
      .Case<AugmentedDeletedTypeAttr>([&](auto a) { return false; })
      .Default([](auto a) { return true; });
}

std::optional<TypeSum> GrandCentralPass::computeField(
    Attribute field, IntegerAttr id, StringAttr prefix, VerbatimBuilder &path,
    SmallVector<VerbatimXMRbuilder> &xmrElems,
    SmallVector<InterfaceElemsBuilder> &interfaceBuilder) {

  auto unsupported = [&](StringRef name, StringRef kind) {
    return VerbatimType({("// <unsupported " + kind + " type>").str(), false});
  };

  return TypeSwitch<Attribute, std::optional<TypeSum>>(field)
      .Case<AugmentedGroundTypeAttr>(
          [&](AugmentedGroundTypeAttr ground) -> std::optional<TypeSum> {
            // Traverse to generate mappings.
            if (!traverseField(field, id, path, xmrElems, interfaceBuilder))
              return std::nullopt;
            FieldRef fieldRef = leafMap.lookup(ground.getID()).field;
            auto value = fieldRef.getValue();
            auto fieldID = fieldRef.getFieldID();
            auto tpe = firrtl::type_cast<FIRRTLBaseType>(
                value.getType()
                    .cast<circt::hw::FieldIDTypeInterface>()
                    .getFinalTypeByFieldID(fieldID));
            if (!tpe.isGround()) {
              value.getDefiningOp()->emitOpError()
                  << "cannot be added to interface with id '"
                  << id.getValue().getZExtValue()
                  << "' because it is not a ground type";
              return std::nullopt;
            }
            return TypeSum(IntegerType::get(getOperation().getContext(),
                                            tpe.getBitWidthOrSentinel()));
          })
      .Case<AugmentedVectorTypeAttr>(
          [&](AugmentedVectorTypeAttr vector) -> std::optional<TypeSum> {
            auto elements = vector.getElements();
            auto firstElement = fromAttr(elements[0]);
            auto elementType =
                computeField(*firstElement, id, prefix,
                             path.snapshot().append("[" + Twine(0) + "]"),
                             xmrElems, interfaceBuilder);
            if (!elementType)
              return std::nullopt;

            for (size_t i = 1, e = elements.size(); i != e; ++i) {
              auto subField = fromAttr(elements[i]);
              if (!subField)
                return std::nullopt;
              (void)traverseField(*subField, id,
                                  path.snapshot().append("[" + Twine(i) + "]"),
                                  xmrElems, interfaceBuilder);
            }

            if (auto *tpe = std::get_if<Type>(&*elementType))
              return TypeSum(
                  hw::UnpackedArrayType::get(*tpe, elements.getValue().size()));
            auto str = std::get<VerbatimType>(*elementType);
            str.dimensions.push_back(elements.getValue().size());
            return TypeSum(str);
          })
      .Case<AugmentedBundleTypeAttr>(
          [&](AugmentedBundleTypeAttr bundle) -> TypeSum {
            auto ifaceName = traverseBundle(bundle, id, prefix, path, xmrElems,
                                            interfaceBuilder);
            assert(ifaceName && *ifaceName);
            return VerbatimType({ifaceName->str(), true});
          })
      .Case<AugmentedStringTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "string");
      })
      .Case<AugmentedBooleanTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "boolean");
      })
      .Case<AugmentedIntegerTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "integer");
      })
      .Case<AugmentedDoubleTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "double");
      })
      .Case<AugmentedLiteralTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "literal");
      })
      .Case<AugmentedDeletedTypeAttr>([&](auto field) -> TypeSum {
        return unsupported(field.getName().getValue(), "deleted");
      });
}

/// Traverse an Annotation that is an AugmentedBundleType.  During traversal,
/// construct any discovered SystemVerilog interfaces.  If this is the root
/// interface, instantiate that interface in the companion. Recurse into fields
/// of the AugmentedBundleType to construct nested interfaces and generate
/// stringy-typed SystemVerilog hierarchical references to drive the
/// interface. Returns false on any failure and true on success.
std::optional<StringAttr> GrandCentralPass::traverseBundle(
    AugmentedBundleTypeAttr bundle, IntegerAttr id, StringAttr prefix,
    VerbatimBuilder &path, SmallVector<VerbatimXMRbuilder> &xmrElems,
    SmallVector<InterfaceElemsBuilder> &interfaceBuilder) {

  unsigned lastIndex = interfaceBuilder.size();
  auto iFaceName = StringAttr::get(
      &getContext(), getNamespace().newName(getInterfaceName(prefix, bundle)));
  interfaceBuilder.emplace_back(iFaceName, id);

  for (auto element : bundle.getElements()) {
    auto field = fromAttr(element);
    if (!field)
      return std::nullopt;

    auto name = cast<DictionaryAttr>(element).getAs<StringAttr>("name");
    // auto signalSym = hw::InnerRefAttr::get(iface.sym_nameAttr(), name);
    // TODO: The `append(name.getValue())` in the following should actually be
    // `append(signalSym)`, but this requires that `computeField` and the
    // functions it calls always return a type for which we can construct an
    // `InterfaceSignalOp`. Since nested interface instances are currently
    // busted (due to the interface being a symbol table), this doesn't work at
    // the moment. Passing a `name` works most of the time, but can be brittle
    // if the interface field requires renaming in the output (e.g. due to
    // naming conflicts).
    auto elementType = computeField(
        *field, id, prefix, path.snapshot().append(".").append(name.getValue()),
        xmrElems, interfaceBuilder);
    if (!elementType)
      return std::nullopt;
    StringAttr description =
        cast<DictionaryAttr>(element).getAs<StringAttr>("description");
    interfaceBuilder[lastIndex].elementsList.emplace_back(description, name,
                                                          *elementType);
  }
  return iFaceName;
}

/// Return the module that is associated with this value.  Use the cached/lazily
/// constructed symbol table to make this fast.
HWModuleLike GrandCentralPass::getEnclosingModule(Value value,
                                                  FlatSymbolRefAttr sym) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return cast<HWModuleLike>(blockArg.getOwner()->getParentOp());

  auto *op = value.getDefiningOp();
  if (InstanceOp instance = dyn_cast<InstanceOp>(op))
    return getSymbolTable().lookup<HWModuleLike>(
        instance.getModuleNameAttr().getValue());

  return op->getParentOfType<HWModuleLike>();
}

/// This method contains the business logic of this pass.
void GrandCentralPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===- Running Grand Central Views/Interface Pass "
                             "-----------------------------===\n");

  CircuitOp circuitOp = getOperation();

  // Look at the circuit annotaitons to do two things:
  //
  // 1. Determine extraction information (directory and filename).
  // 2. Populate a worklist of all annotations that encode interfaces.
  //
  // Remove annotations encoding interfaces, but leave extraction information as
  // this may be needed by later passes.
  SmallVector<Annotation> worklist;
  bool removalError = false;
  AnnotationSet::removeAnnotations(circuitOp, [&](Annotation anno) {
    if (anno.isClass(augmentedBundleTypeClass)) {
      // If we are in "instantiateCompanionOnly" mode, then we don't need to
      // create the interface, so we can skip adding it to the worklist.  This
      // is a janky hack for situations where you want to synthesize assertion
      // logic included in the companion, but don't want to have a dead
      // interface hanging around (or have problems with tools understanding
      // interfaces).
      if (!instantiateCompanionOnly)
        worklist.push_back(anno);
      ++numAnnosRemoved;
      return true;
    }
    if (anno.isClass(extractGrandCentralClass)) {
      if (maybeExtractInfo) {
        emitCircuitError("more than one 'ExtractGrandCentralAnnotation' was "
                         "found, but exactly one must be provided");
        removalError = true;
        return false;
      }

      auto directory = anno.getMember<StringAttr>("directory");
      auto filename = anno.getMember<StringAttr>("filename");
      if (!directory || !filename) {
        emitCircuitError()
            << "contained an invalid 'ExtractGrandCentralAnnotation' that does "
               "not contain 'directory' and 'filename' fields: "
            << anno.getDict();
        removalError = true;
        return false;
      }
      if (directory.getValue().empty())
        directory = StringAttr::get(circuitOp.getContext(), ".");

      maybeExtractInfo = {directory, filename};
      // Do not delete this annotation.  Extraction info may be needed later.
      return false;
    }
    if (anno.isClass(grandCentralHierarchyFileAnnoClass)) {
      if (maybeHierarchyFileYAML) {
        emitCircuitError("more than one 'GrandCentralHierarchyFileAnnotation' "
                         "was found, but zero or one may be provided");
        removalError = true;
        return false;
      }

      auto filename = anno.getMember<StringAttr>("filename");
      if (!filename) {
        emitCircuitError()
            << "contained an invalid 'GrandCentralHierarchyFileAnnotation' "
               "that does not contain 'directory' and 'filename' fields: "
            << anno.getDict();
        removalError = true;
        return false;
      }

      maybeHierarchyFileYAML = filename;
      ++numAnnosRemoved;
      return true;
    }
    if (anno.isClass(prefixInterfacesAnnoClass)) {
      if (!interfacePrefix.empty()) {
        emitCircuitError("more than one 'PrefixInterfacesAnnotation' was "
                         "found, but zero or one may be provided");
        removalError = true;
        return false;
      }

      auto prefix = anno.getMember<StringAttr>("prefix");
      if (!prefix) {
        emitCircuitError()
            << "contained an invalid 'PrefixInterfacesAnnotation' that does "
               "not contain a 'prefix' field: "
            << anno.getDict();
        removalError = true;
        return false;
      }

      interfacePrefix = prefix.getValue();
      ++numAnnosRemoved;
      return true;
    }
    if (anno.isClass(testBenchDirAnnoClass)) {
      testbenchDir = anno.getMember<StringAttr>("dirname");
      return false;
    }
    return false;
  });

  // Find the DUT if it exists.  This needs to be known before the circuit is
  // walked.
  for (auto mod : circuitOp.getOps<FModuleOp>()) {
    if (failed(extractDUT(mod, dut)))
      removalError = true;
  }

  if (removalError)
    return signalPassFailure();

  LLVM_DEBUG({
    llvm::dbgs() << "Extraction Info:\n";
    if (maybeExtractInfo)
      llvm::dbgs() << "  directory: " << maybeExtractInfo->directory << "\n"
                   << "  filename: " << maybeExtractInfo->bindFilename << "\n";
    else
      llvm::dbgs() << "  <none>\n";
    llvm::dbgs() << "DUT: ";
    if (dut)
      llvm::dbgs() << dut.getModuleName() << "\n";
    else
      llvm::dbgs() << "<none>\n";
    llvm::dbgs()
        << "Prefix Info (from PrefixInterfacesAnnotation):\n"
        << "  prefix: " << interfacePrefix << "\n"
        << "Hierarchy File Info (from GrandCentralHierarchyFileAnnotation):\n"
        << "  filename: ";
    if (maybeHierarchyFileYAML)
      llvm::dbgs() << *maybeHierarchyFileYAML;
    else
      llvm::dbgs() << "<none>";
    llvm::dbgs() << "\n";
  });

  // Exit immediately if no annotations indicative of interfaces that need to be
  // built exist.  However, still generate the YAML file if the annotation for
  // this was passed in because some flows expect this.
  if (worklist.empty()) {
    if (!maybeHierarchyFileYAML)
      return markAllAnalysesPreserved();
    std::string yamlString;
    llvm::raw_string_ostream stream(yamlString);
    ::yaml::Context yamlContext({interfaceMap});
    llvm::yaml::Output yout(stream);
    OpBuilder builder(circuitOp);
    SmallVector<sv::InterfaceOp, 0> interfaceVec;
    yamlize(yout, interfaceVec, true, yamlContext);
    builder.setInsertionPointToStart(circuitOp.getBodyBlock());
    builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), yamlString)
        ->setAttr("output_file",
                  hw::OutputFileAttr::getFromFilename(
                      &getContext(), maybeHierarchyFileYAML->getValue(),
                      /*excludFromFileList=*/true));
    LLVM_DEBUG({ llvm::dbgs() << "Generated YAML:" << yamlString << "\n"; });
    return;
  }

  // Setup the builder to create ops _inside the FIRRTL circuit_.  This is
  // necessary because interfaces and interface instances are created.
  // Instances link to their definitions via symbols and we don't want to
  // break this.
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBodyBlock());

  // Maybe get an "id" from an Annotation.  Generate error messages on the op if
  // no "id" exists.
  auto getID = [&](Operation *op,
                   Annotation annotation) -> std::optional<IntegerAttr> {
    auto id = annotation.getMember<IntegerAttr>("id");
    if (!id) {
      op->emitOpError()
          << "contained a malformed "
             "'sifive.enterprise.grandcentral.AugmentedGroundType' annotation "
             "that did not contain an 'id' field";
      removalError = true;
      return std::nullopt;
    }
    return id;
  };

  /// TODO: Handle this differently to allow construction of an optionsl
  auto instancePathCache = InstancePathCache(getAnalysis<InstanceGraph>());
  instancePaths = &instancePathCache;

  /// Contains the set of modules which are instantiated by the DUT, but not a
  /// companion or instantiated by a companion.  If no DUT exists, treat the top
  /// module as if it were the DUT.  This works by doing a depth-first walk of
  /// the instance graph, starting from the "effective" DUT and stopping the
  /// search at any modules which are known companions.
  DenseSet<hw::HWModuleLike> dutModules;
  FModuleOp effectiveDUT = dut;
  if (!effectiveDUT)
    effectiveDUT = cast<FModuleOp>(
        *instancePaths->instanceGraph.getTopLevelNode()->getModule());
  auto dfRange =
      llvm::depth_first(instancePaths->instanceGraph.lookup(effectiveDUT));
  for (auto i = dfRange.begin(), e = dfRange.end(); i != e;) {
    auto module = cast<FModuleLike>(*i->getModule());
    if (AnnotationSet(module).hasAnnotation(companionAnnoClass)) {
      i.skipChildren();
      continue;
    }
    dutModules.insert(i->getModule());
    // Manually increment the iterator to avoid walking off the end from
    // skipChildren.
    ++i;
  }

  // Maybe return the lone instance of a module.  Generate errors on the op if
  // the module is not instantiated or is multiply instantiated.
  auto exactlyOneInstance = [&](FModuleOp op,
                                StringRef msg) -> std::optional<InstanceOp> {
    auto *node = instancePaths->instanceGraph[op];

    switch (node->getNumUses()) {
    case 0:
      op->emitOpError() << "is marked as a GrandCentral '" << msg
                        << "', but is never instantiated";
      return std::nullopt;
    case 1:
      return cast<InstanceOp>(*(*node->uses().begin())->getInstance());
    default:
      auto diag = op->emitOpError()
                  << "is marked as a GrandCentral '" << msg
                  << "', but it is instantiated more than once";
      for (auto *instance : node->uses())
        diag.attachNote(instance->getInstance()->getLoc())
            << "it is instantiated here";
      return std::nullopt;
    }
  };

  nlaTable = &getAnalysis<NLATable>();

  /// Walk the circuit and extract all information related to scattered Grand
  /// Central annotations.  This is used to populate: (1) the companionIDMap and
  /// (2) the leafMap.  Annotations are removed as they are discovered and if
  /// they are not malformed.
  removalError = false;
  circuitOp.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<RegOp, RegResetOp, WireOp, NodeOp>([&](auto op) {
          AnnotationSet::removeAnnotations(op, [&](Annotation annotation) {
            if (!annotation.isClass(augmentedGroundTypeClass))
              return false;
            auto maybeID = getID(op, annotation);
            if (!maybeID)
              return false;
            auto sym =
                annotation.getMember<FlatSymbolRefAttr>("circt.nonlocal");
            leafMap[*maybeID] = {{op.getResult(), annotation.getFieldID()},
                                 sym};
            ++numAnnosRemoved;
            return true;
          });
        })
        // TODO: Figure out what to do with this.
        .Case<InstanceOp>([&](auto op) {
          AnnotationSet::removePortAnnotations(op, [&](unsigned i,
                                                       Annotation annotation) {
            if (!annotation.isClass(augmentedGroundTypeClass))
              return false;
            op.emitOpError()
                << "is marked as an interface element, but this should be "
                   "impossible due to how the Chisel Grand Central API works";
            removalError = true;
            return false;
          });
        })
        .Case<MemOp>([&](auto op) {
          AnnotationSet::removeAnnotations(op, [&](Annotation annotation) {
            if (!annotation.isClass(augmentedGroundTypeClass))
              return false;
            op.emitOpError()
                << "is marked as an interface element, but this does not make "
                   "sense (is there a scattering bug or do you have a "
                   "malformed hand-crafted MLIR circuit?)";
            removalError = true;
            return false;
          });
          AnnotationSet::removePortAnnotations(
              op, [&](unsigned i, Annotation annotation) {
                if (!annotation.isClass(augmentedGroundTypeClass))
                  return false;
                op.emitOpError()
                    << "has port '" << i
                    << "' marked as an interface element, but this does not "
                       "make sense (is there a scattering bug or do you have a "
                       "malformed hand-crafted MLIR circuit?)";
                removalError = true;
                return false;
              });
        })
        .Case<FModuleOp>([&](FModuleOp op) {
          // Handle annotations on the ports.
          AnnotationSet::removePortAnnotations(op, [&](unsigned i,
                                                       Annotation annotation) {
            if (!annotation.isClass(augmentedGroundTypeClass))
              return false;
            auto maybeID = getID(op, annotation);
            if (!maybeID)
              return false;
            auto sym =
                annotation.getMember<FlatSymbolRefAttr>("circt.nonlocal");
            leafMap[*maybeID] = {{op.getArgument(i), annotation.getFieldID()},
                                 sym};
            ++numAnnosRemoved;
            return true;
          });

          // Handle annotations on the module.
          AnnotationSet::removeAnnotations(op, [&](Annotation annotation) {
            if (!annotation.getClass().startswith(viewAnnoClass))
              return false;
            auto isNonlocal = annotation.getMember<FlatSymbolRefAttr>(
                                  "circt.nonlocal") != nullptr;
            auto name = annotation.getMember<StringAttr>("name");
            auto id = annotation.getMember<IntegerAttr>("id");
            if (!id) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that did "
                     "not contain an 'id' field with an 'IntegerAttr' value";
              goto FModuleOp_error;
            }
            if (!name) {
              op.emitOpError()
                  << "has a malformed "
                     "'sifive.enterprise.grandcentral.ViewAnnotation' that did "
                     "not contain a 'name' field with a 'StringAttr' value";
              goto FModuleOp_error;
            }

            // If this is a companion, then:
            //   1. Insert it into the companion map
            //   2. Create a new mapping module.
            //   3. Instatiate the mapping module in the companion.
            //   4. Check that the companion is instantated exactly once.
            //   5. Set attributes on that lone instance so it will become a
            //      bind if extraction information was provided.  If a DUT is
            //      known, then anything in the test harness will not be
            //      extracted.
            if (annotation.getClass() == companionAnnoClass) {
              builder.setInsertionPointToEnd(circuitOp.getBodyBlock());

              companionIDMap[id] = {name.getValue(), op, isNonlocal};

              // Assert that the companion is instantiated once and only once.
              auto instance = exactlyOneInstance(op, "companion");
              if (!instance)
                goto FModuleOp_error;

              // If no extraction info was provided, exit.  Otherwise, setup the
              // lone instance of the companion to be lowered as a bind.
              if (!maybeExtractInfo) {
                ++numAnnosRemoved;
                return true;
              }

              // If the companion is instantiated above the DUT, then don't
              // extract it.
              if (dut && !instancePaths->instanceGraph.isAncestor(op, dut)) {
                ++numAnnosRemoved;
                return true;
              }

              // Lower the companion to a bind unless the user told us
              // explicitly not to.
              if (!instantiateCompanionOnly)
                (*instance)->setAttr("lowerToBind", builder.getUnitAttr());

              (*instance)->setAttr(
                  "output_file",
                  hw::OutputFileAttr::getFromFilename(
                      &getContext(), maybeExtractInfo->bindFilename.getValue(),
                      /*excludeFromFileList=*/true));

              // Look for any modules/extmodules _only_ instantiated by the
              // companion.  If these have no output file attribute, then mark
              // them as being extracted into the Grand Central directory.
              InstanceGraphNode *companionNode =
                  instancePaths->instanceGraph.lookup(op);

              LLVM_DEBUG({
                llvm::dbgs()
                    << "Found companion module: "
                    << companionNode->getModule().getModuleName() << "\n"
                    << "  submodules exclusively instantiated "
                       "(including companion):\n";
              });

              for (auto &node : llvm::depth_first(companionNode)) {
                auto mod = node->getModule();

                // Check to see if we should change the output directory of a
                // module.  Only update in the following conditions:
                //   1) The module is the companion.
                //   2) The module is NOT instantiated by the effective DUT.
                auto *modNode = instancePaths->instanceGraph.lookup(mod);
                SmallVector<InstanceRecord *> instances(modNode->uses());
                if (modNode != companionNode &&
                    dutModules.count(modNode->getModule()))
                  continue;

                LLVM_DEBUG({
                  llvm::dbgs()
                      << "    - module: " << mod.getModuleName() << "\n";
                });

                if (auto extmodule = dyn_cast<FExtModuleOp>(*mod)) {
                  for (auto anno : AnnotationSet(extmodule)) {
                    if (!anno.isClass(blackBoxInlineAnnoClass) &&
                        !anno.isClass(blackBoxPathAnnoClass))
                      continue;
                    if (extmodule->hasAttr("output_file"))
                      break;
                    extmodule->setAttr(
                        "output_file",
                        hw::OutputFileAttr::getAsDirectory(
                            &getContext(),
                            maybeExtractInfo->directory.getValue()));
                    break;
                  }
                  continue;
                }

                // Move this module under the Grand Central output directory if
                // no pre-existing output file information is present.
                if (!mod->hasAttr("output_file")) {
                  mod->setAttr("output_file",
                               hw::OutputFileAttr::getAsDirectory(
                                   &getContext(),
                                   maybeExtractInfo->directory.getValue(),
                                   /*excludeFromFileList=*/true,
                                   /*includeReplicatedOps=*/true));
                  mod->setAttr("comment", builder.getStringAttr(
                                              "VCS coverage exclude_file"));
                }
              }

              ++numAnnosRemoved;
              return true;
            }

            op.emitOpError()
                << "unknown annotation class: " << annotation.getDict();

          FModuleOp_error:
            removalError = true;
            return false;
          });
        });
  });

  if (removalError)
    return signalPassFailure();

  LLVM_DEBUG({
    // Print out the companion map and all leaf values that were discovered.
    // Sort these by their keys before printing to make this easier to read.
    SmallVector<IntegerAttr> ids;
    auto sort = [&ids]() {
      llvm::sort(ids, [](IntegerAttr a, IntegerAttr b) {
        return a.getValue().getZExtValue() < b.getValue().getZExtValue();
      });
    };
    for (auto tuple : companionIDMap)
      ids.push_back(cast<IntegerAttr>(tuple.first));
    sort();
    llvm::dbgs() << "companionIDMap:\n";
    for (auto id : ids) {
      auto value = companionIDMap.lookup(id);
      llvm::dbgs() << "  - " << id.getValue() << ": "
                   << value.companion.getName() << " -> " << value.name << "\n";
    }
    ids.clear();
    for (auto tuple : leafMap)
      ids.push_back(cast<IntegerAttr>(tuple.first));
    sort();
    llvm::dbgs() << "leafMap:\n";
    for (auto id : ids) {
      auto fieldRef = leafMap.lookup(id).field;
      auto value = fieldRef.getValue();
      auto fieldID = fieldRef.getFieldID();
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        FModuleOp module = cast<FModuleOp>(blockArg.getOwner()->getParentOp());
        llvm::dbgs() << "  - " << id.getValue() << ": "
                     << module.getName() + ">" +
                            module.getPortName(blockArg.getArgNumber());
        if (fieldID)
          llvm::dbgs() << ", fieldID=" << fieldID;
        llvm::dbgs() << "\n";
      } else {
        llvm::dbgs() << "  - " << id.getValue() << ": "
                     << value.getDefiningOp()
                            ->getAttr("name")
                            .cast<StringAttr>()
                            .getValue();
        if (fieldID)
          llvm::dbgs() << ", fieldID=" << fieldID;
        llvm::dbgs() << "\n";
      }
    }
  });

  // Now, iterate over the worklist of interface-encoding annotations to create
  // the interface and all its sub-interfaces (interfaces that it instantiates),
  // instantiate the top-level interface, and generate a "mappings file" that
  // will use XMRs to drive the interface.  If extraction info is available,
  // then the top-level instantiate interface will be marked for extraction via
  // a SystemVerilog bind.
  SmallVector<sv::InterfaceOp, 2> interfaceVec;
  SmallDenseMap<FModuleLike, SmallVector<InterfaceElemsBuilder>>
      companionToInterfaceMap;
  auto compareInterfaceSignal = [&](InterfaceElemsBuilder &lhs,
                                    InterfaceElemsBuilder &rhs) {
    auto compareProps = [&](InterfaceElemsBuilder::Properties &lhs,
                            InterfaceElemsBuilder::Properties &rhs) {
      // If it's a verbatim op, no need to check the string, because the
      // interface names might not match. As long as the signal types match that
      // is sufficient.
      if (lhs.elemType.index() == 0 && rhs.elemType.index() == 0)
        return true;
      if (std::get<Type>(lhs.elemType) == std::get<Type>(rhs.elemType))
        return true;
      return false;
    };
    return std::equal(lhs.elementsList.begin(), lhs.elementsList.end(),
                      rhs.elementsList.begin(), compareProps);
  };
  for (auto anno : worklist) {
    auto bundle = AugmentedBundleTypeAttr::get(&getContext(), anno.getDict());

    // The top-level AugmentedBundleType must have a global ID field so that
    // this can be linked to the companion.
    if (!bundle.isRoot()) {
      emitCircuitError() << "missing 'id' in root-level BundleType: "
                         << anno.getDict() << "\n";
      removalError = true;
      continue;
    }

    if (companionIDMap.count(bundle.getID()) == 0) {
      emitCircuitError() << "no companion found with 'id' value '"
                         << bundle.getID().getValue().getZExtValue() << "'\n";
      removalError = true;
      continue;
    }

    // Decide on a symbol name to use for the interface instance. This is needed
    // in `traverseBundle` as a placeholder for the connect operations.
    auto companionIter = companionIDMap.lookup(bundle.getID());
    auto companionModule = companionIter.companion;
    auto symbolName = getNamespace().newName(
        "__" + companionIDMap.lookup(bundle.getID()).name + "_" +
        getInterfaceName(bundle.getPrefix(), bundle) + "__");

    // Recursively walk the AugmentedBundleType to generate interfaces and XMRs.
    // Error out if this returns None (indicating that the annotation is
    // malformed in some way).  A good error message is generated inside
    // `traverseBundle` or the functions it calls.
    auto instanceSymbol =
        hw::InnerRefAttr::get(SymbolTable::getSymbolName(companionModule),
                              StringAttr::get(&getContext(), symbolName));
    VerbatimBuilder::Base verbatimData;
    VerbatimBuilder verbatim(verbatimData);
    verbatim += instanceSymbol;
    // List of interface elements.

    SmallVector<VerbatimXMRbuilder> xmrElems;
    SmallVector<InterfaceElemsBuilder> interfaceBuilder;

    auto ifaceName = traverseBundle(bundle, bundle.getID(), bundle.getPrefix(),
                                    verbatim, xmrElems, interfaceBuilder);
    if (!ifaceName) {
      removalError = true;
      continue;
    }

    if (companionIter.isNonlocal) {
      // If the companion module has two exactly same ViewAnnotation.companion
      // annotations, then add the interface for only one of them. This happens
      // when the companion is deduped.
      auto viewMapIter = companionToInterfaceMap.find(companionModule);
      if (viewMapIter != companionToInterfaceMap.end())
        if (std::equal(interfaceBuilder.begin(), interfaceBuilder.end(),
                       viewMapIter->getSecond().begin(),
                       compareInterfaceSignal)) {
          continue;
        }

      companionToInterfaceMap[companionModule] = interfaceBuilder;
    }

    if (interfaceBuilder.empty())
      continue;
    auto companionBuilder =
        OpBuilder::atBlockEnd(companionModule.getBodyBlock());

    // Generate gathered XMR's.
    for (auto xmrElem : xmrElems) {
      auto uloc = companionBuilder.getUnknownLoc();
      companionBuilder.create<sv::VerbatimOp>(uloc, xmrElem.str, xmrElem.val,
                                              xmrElem.syms);
    }
    numXMRs += xmrElems.size();

    sv::InterfaceOp topIface;
    for (const auto &ifaceBuilder : interfaceBuilder) {
      auto builder = OpBuilder::atBlockEnd(getOperation().getBodyBlock());
      auto loc = getOperation().getLoc();
      sv::InterfaceOp iface =
          builder.create<sv::InterfaceOp>(loc, ifaceBuilder.iFaceName);
      if (!topIface)
        topIface = iface;
      ++numInterfaces;
      if (dut &&
          !instancePaths->instanceGraph.isAncestor(
              companionIDMap[ifaceBuilder.id].companion, dut) &&
          testbenchDir)
        iface->setAttr("output_file",
                       hw::OutputFileAttr::getAsDirectory(
                           &getContext(), testbenchDir.getValue(),
                           /*excludeFromFileList=*/true));
      else if (maybeExtractInfo)
        iface->setAttr("output_file",
                       hw::OutputFileAttr::getAsDirectory(
                           &getContext(), getOutputDirectory().getValue(),
                           /*excludeFromFileList=*/true));
      iface.setCommentAttr(builder.getStringAttr("VCS coverage exclude_file"));
      builder.setInsertionPointToEnd(
          cast<sv::InterfaceOp>(iface).getBodyBlock());
      interfaceMap[FlatSymbolRefAttr::get(builder.getContext(),
                                          ifaceBuilder.iFaceName)] = iface;
      for (auto elem : ifaceBuilder.elementsList) {

        auto uloc = builder.getUnknownLoc();

        auto description = elem.description;

        if (description) {
          auto descriptionOp = builder.create<sv::VerbatimOp>(
              uloc, ("// " + cleanupDescription(description.getValue())));

          // If we need to generate a YAML representation of this interface,
          // then add an attribute indicating that this `sv::VerbatimOp` is
          // actually a description.
          if (maybeHierarchyFileYAML)
            descriptionOp->setAttr("firrtl.grandcentral.yaml.type",
                                   builder.getStringAttr("description"));
        }
        if (auto *str = std::get_if<VerbatimType>(&elem.elemType)) {
          auto instanceOp = builder.create<sv::VerbatimOp>(
              uloc, str->toStr(elem.elemName.getValue()));

          // If we need to generate a YAML representation of the interface, then
          // add attirbutes that describe what this `sv::VerbatimOp` is.
          if (maybeHierarchyFileYAML) {
            if (str->instantiation)
              instanceOp->setAttr("firrtl.grandcentral.yaml.type",
                                  builder.getStringAttr("instance"));
            else
              instanceOp->setAttr("firrtl.grandcentral.yaml.type",
                                  builder.getStringAttr("unsupported"));
            instanceOp->setAttr("firrtl.grandcentral.yaml.name", elem.elemName);
            instanceOp->setAttr("firrtl.grandcentral.yaml.dimensions",
                                builder.getI32ArrayAttr(str->dimensions));
            instanceOp->setAttr(
                "firrtl.grandcentral.yaml.symbol",
                FlatSymbolRefAttr::get(builder.getContext(), str->str));
          }
          continue;
        }

        auto tpe = std::get<Type>(elem.elemType);
        builder.create<sv::InterfaceSignalOp>(uloc, elem.elemName.getValue(),
                                              tpe);
      }
    }

    ++numViews;

    interfaceVec.push_back(topIface);

    // Instantiate the interface inside the companion.
    builder.setInsertionPointToStart(companionModule.getBodyBlock());
    builder.create<sv::InterfaceInstanceOp>(
        getOperation().getLoc(), topIface.getInterfaceType(),
        companionIDMap.lookup(bundle.getID()).name,
        hw::InnerSymAttr::get(builder.getStringAttr(symbolName)));

    // If no extraction information was present, then just leave the interface
    // instantiated in the companion.  Otherwise, make it a bind.
    if (!maybeExtractInfo)
      continue;

    // If the interface is associated with a companion that is instantiated
    // above the DUT (e.g.., in the test harness), then don't extract it.
    if (dut && !instancePaths->instanceGraph.isAncestor(
                   companionIDMap[bundle.getID()].companion, dut))
      continue;
  }

  // If a `GrandCentralHierarchyFileAnnotation` was passed in, generate a YAML
  // representation of the interfaces that we produced with the filename that
  // that annotation provided.
  if (maybeHierarchyFileYAML) {
    std::string yamlString;
    llvm::raw_string_ostream stream(yamlString);
    ::yaml::Context yamlContext({interfaceMap});
    llvm::yaml::Output yout(stream);
    yamlize(yout, interfaceVec, true, yamlContext);

    builder.setInsertionPointToStart(circuitOp.getBodyBlock());
    builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), yamlString)
        ->setAttr("output_file",
                  hw::OutputFileAttr::getFromFilename(
                      &getContext(), maybeHierarchyFileYAML->getValue(),
                      /*excludFromFileList=*/true));
    LLVM_DEBUG({ llvm::dbgs() << "Generated YAML:" << yamlString << "\n"; });
  }

  // Signal pass failure if any errors were found while examining circuit
  // annotations.
  if (removalError)
    return signalPassFailure();
  markAnalysesPreserved<NLATable>();
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
circt::firrtl::createGrandCentralPass(bool instantiateCompanionOnly) {
  return std::make_unique<GrandCentralPass>(instantiateCompanionOnly);
}
