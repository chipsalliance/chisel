//===- CalyxEmitter.cpp - Calyx dialect to .futil emitter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements an emitter for the native Calyx language, which uses
// .futil as an alias.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxEmitter.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace circt;
using namespace calyx;
using namespace mlir;

namespace {

static constexpr std::string_view LSquare() { return "["; }
static constexpr std::string_view RSquare() { return "]"; }
static constexpr std::string_view LAngleBracket() { return "<"; }
static constexpr std::string_view RAngleBracket() { return ">"; }
static constexpr std::string_view LParen() { return "("; }
static constexpr std::string_view RParen() { return ")"; }
static constexpr std::string_view colon() { return ": "; }
static constexpr std::string_view space() { return " "; }
static constexpr std::string_view period() { return "."; }
static constexpr std::string_view questionMark() { return " ? "; }
static constexpr std::string_view exclamationMark() { return "!"; }
static constexpr std::string_view equals() { return "="; }
static constexpr std::string_view comma() { return ", "; }
static constexpr std::string_view arrow() { return " -> "; }
static constexpr std::string_view delimiter() { return "\""; }
static constexpr std::string_view apostrophe() { return "'"; }
static constexpr std::string_view LBraceEndL() { return "{\n"; }
static constexpr std::string_view RBraceEndL() { return "}\n"; }
static constexpr std::string_view semicolonEndL() { return ";\n"; }
static constexpr std::string_view addressSymbol() { return "@"; }
static constexpr std::string_view endl() { return "\n"; }
static constexpr std::string_view metadataLBrace() { return "#{\n"; }
static constexpr std::string_view metadataRBrace() { return "}#\n"; }

/// A list of integer attributes supported by the native Calyx compiler.
constexpr std::array<StringRef, 7> integerAttributes{
    "external",       "static",        "share", "bound",
    "write_together", "read_together", "pos",
};

/// A list of boolean attributes supported by the native Calyx compiler.
constexpr std::array<StringRef, 12> booleanAttributes{
    "clk",      "done",   "go",          "reset",  "generated",   "precious",
    "toplevel", "stable", "nointerface", "inline", "state_share", "data",
};

static std::optional<StringRef> getCalyxAttrIdentifier(NamedAttribute attr) {
  StringRef identifier = attr.getName().strref();
  if (identifier.contains(".")) {
    Dialect *dialect = attr.getNameDialect();
    if (dialect != nullptr && isa<CalyxDialect>(*dialect)) {
      return std::get<1>(identifier.split("."));
    }
    return std::nullopt;
  }

  return identifier;
}

/// Determines whether the given identifier is a valid Calyx attribute.
static bool isValidCalyxAttribute(StringRef identifier) {

  return llvm::find(integerAttributes, identifier) != integerAttributes.end() ||
         llvm::find(booleanAttributes, identifier) != booleanAttributes.end();
}

/// Additional information about an unsupported operation.
static std::optional<StringRef> unsupportedOpInfo(Operation *op) {
  return llvm::TypeSwitch<Operation *, std::optional<StringRef>>(op)
      .Case<ExtSILibOp>([](auto) -> std::optional<StringRef> {
        static constexpr std::string_view info =
            "calyx.std_extsi is currently not available in the native Rust "
            "compiler (see github.com/cucapra/calyx/issues/1009)";
        return {info};
      })
      .Default([](auto) { return std::nullopt; });
}

/// A tracker to determine which libraries should be imported for a given
/// program.
struct ImportTracker {
public:
  /// Returns the list of library names used for in this program.
  /// E.g. if `primitives/core.futil` is used, returns { "core" }.
  FailureOr<llvm::SmallSet<StringRef, 4>> getLibraryNames(ModuleOp module) {
    auto walkRes = module.walk([&](ComponentOp component) {
      for (auto &op : *component.getBodyBlock()) {
        if (!isa<CellInterface>(op) || isa<InstanceOp, PrimitiveOp>(op))
          // It is not a primitive.
          continue;
        auto libraryName = getLibraryFor(&op);
        if (failed(libraryName))
          return WalkResult::interrupt();
        usedLibraries.insert(*libraryName);
      }
      return WalkResult::advance();
    });
    if (walkRes.wasInterrupted())
      return failure();
    return usedLibraries;
  }

private:
  /// Returns the library name for a given Operation Type.
  FailureOr<StringRef> getLibraryFor(Operation *op) {
    return TypeSwitch<Operation *, FailureOr<StringRef>>(op)
        .Case<MemoryOp, RegisterOp, NotLibOp, AndLibOp, OrLibOp, XorLibOp,
              AddLibOp, SubLibOp, GtLibOp, LtLibOp, EqLibOp, NeqLibOp, GeLibOp,
              LeLibOp, LshLibOp, RshLibOp, SliceLibOp, PadLibOp, WireLibOp>(
            [&](auto op) -> FailureOr<StringRef> {
              static constexpr std::string_view sCore = "core";
              return {sCore};
            })
        .Case<SgtLibOp, SltLibOp, SeqLibOp, SneqLibOp, SgeLibOp, SleLibOp,
              SrshLibOp, MultPipeLibOp, RemUPipeLibOp, RemSPipeLibOp,
              DivUPipeLibOp, DivSPipeLibOp>(
            [&](auto op) -> FailureOr<StringRef> {
              static constexpr std::string_view sBinaryOperators =
                  "binary_operators";
              return {sBinaryOperators};
            })
        .Case<SeqMemoryOp>([&](auto op) -> FailureOr<StringRef> {
          static constexpr std::string_view sMemories = "memories";
          return {sMemories};
        })
        /*.Case<>([&](auto op) { library = "math"; })*/
        .Default([&](auto op) {
          auto diag = op->emitOpError() << "not supported for emission";
          auto note = unsupportedOpInfo(op);
          if (note)
            diag.attachNote() << *note;
          return diag;
        });
  }
  /// Maintains a unique list of libraries used throughout the lifetime of the
  /// tracker.
  llvm::SmallSet<StringRef, 4> usedLibraries;
};

//===----------------------------------------------------------------------===//
// Emitter
//===----------------------------------------------------------------------===//

/// An emitter for Calyx dialect operations to .futil output.
struct Emitter {
  Emitter(llvm::raw_ostream &os) : os(os) {}
  LogicalResult finalize();

  // Indentation
  raw_ostream &indent() { return os.indent(currentIndent); }
  void addIndent() { currentIndent += 2; }
  void reduceIndent() {
    assert(currentIndent >= 2 && "Unintended indentation wrap");
    currentIndent -= 2;
  }

  // Module emission
  void emitModule(ModuleOp op);

  // Metadata emission for the Cider debugger.
  void emitCiderMetadata(mlir::ModuleOp op) {
    auto metadata = op->getAttrOfType<ArrayAttr>("calyx.metadata");
    if (!metadata)
      return;

    constexpr std::string_view metadataIdentifier = "metadata";
    os << endl() << metadataIdentifier << space() << metadataLBrace();

    for (auto sourceLoc : llvm::enumerate(metadata)) {
      // <index>: <source-location>\n
      os << std::to_string(sourceLoc.index()) << colon();
      os << sourceLoc.value().cast<StringAttr>().getValue() << endl();
    }

    os << metadataRBrace();
  }

  /// Import emission.
  LogicalResult emitImports(ModuleOp op) {
    auto emitImport = [&](StringRef library) {
      // Libraries share a common relative path:
      //   primitives/<library-name>.futil
      os << "import " << delimiter() << "primitives/" << library << period()
         << "futil" << delimiter() << semicolonEndL();
    };

    auto libraryNames = importTracker.getLibraryNames(op);
    if (failed(libraryNames))
      return failure();

    for (StringRef library : *libraryNames)
      emitImport(library);

    return success();
  }

  // Component emission
  void emitComponent(ComponentInterface op);
  void emitComponentPorts(ComponentInterface op);

  // HWModuleExtern emission
  void emitPrimitiveExtern(hw::HWModuleExternOp op);
  void emitPrimitivePorts(hw::HWModuleExternOp op);

  // Instance emission
  void emitInstance(InstanceOp op);

  // Primitive emission
  void emitPrimitive(PrimitiveOp op);

  // Wires emission
  void emitWires(WiresOp op);

  // Group emission
  void emitGroup(GroupInterface group);

  // Control emission
  void emitControl(ControlOp control);

  // Assignment emission
  void emitAssignment(AssignOp op);

  // Enable emission
  void emitEnable(EnableOp enable);

  // Register emission
  void emitRegister(RegisterOp reg);

  // Memory emission
  void emitMemory(MemoryOp memory);

  // Seq Memory emission
  void emitSeqMemory(SeqMemoryOp memory);

  // Invoke emission
  void emitInvoke(InvokeOp invoke);

  // Emits a library primitive with template parameters based on all in- and
  // output ports.
  // e.g.:
  //   $f.in0, $f.in1, $f.in2, $f.out : calyx.std_foo "f" : i1, i2, i3, i4
  // emits:
  //   f = std_foo(1, 2, 3, 4);
  void emitLibraryPrimTypedByAllPorts(Operation *op);

  // Emits a library primitive with a single template parameter based on the
  // first input port.
  // e.g.:
  //   $f.in0, $f.in1, $f.out : calyx.std_foo "f" : i32, i32, i1
  // emits:
  //   f = std_foo(32);
  void emitLibraryPrimTypedByFirstInputPort(Operation *op);

  // Emits a library primitive with a single template parameter based on the
  // first output port.
  // e.g.:
  //   $f.in0, $f.in1, $f.out : calyx.std_foo "f" : i32, i32, i1
  // emits:
  //   f = std_foo(1);
  void emitLibraryPrimTypedByFirstOutputPort(
      Operation *op, std::optional<StringRef> calyxLibName = {});

private:
  /// Used to track which imports are required for this program.
  ImportTracker importTracker;

  /// Emit an error and remark that emission failed.
  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitError(message);
  }

  /// Emit an error and remark that emission failed.
  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    encounteredError = true;
    return op->emitOpError(message);
  }

  /// Calyx attributes are emitted in one of the two following formats:
  /// (1)  @<attribute-name>(<attribute-value>), e.g. `@go`, `@bound(5)`.
  /// (2)  <"<attribute-name>"=<attribute-value>>, e.g. `<"static"=1>`.
  ///
  /// Since ports are structural in nature and not operations, an
  /// extra boolean value is added to determine whether this is a port of the
  /// given operation.
  std::string getAttribute(Operation *op, NamedAttribute attr, bool isPort) {

    std::optional<StringRef> identifierOpt = getCalyxAttrIdentifier(attr);
    // Verify this is a Calyx attribute
    if (!identifierOpt.has_value())
      return "";

    StringRef identifier = *identifierOpt;
    // Verify this attribute is supported for emission.
    if (!isValidCalyxAttribute(identifier))
      return "";

    // Determines whether the attribute should follow format (2).
    bool isGroupOrComponentAttr = isa<GroupOp, ComponentOp>(op) && !isPort;

    std::string output;
    llvm::raw_string_ostream buffer(output);
    buffer.reserveExtraSpace(16);

    bool isBooleanAttribute =
        llvm::find(booleanAttributes, identifier) != booleanAttributes.end();
    if (attr.getValue().isa<UnitAttr>()) {
      assert(isBooleanAttribute &&
             "Non-boolean attributes must provide an integer value.");
      if (isGroupOrComponentAttr) {
        buffer << LAngleBracket() << delimiter() << identifier << delimiter()
               << equals() << "1" << RAngleBracket();
      } else {
        buffer << addressSymbol() << identifier << space();
      }
    } else if (auto intAttr = attr.getValue().dyn_cast<IntegerAttr>()) {
      APInt value = intAttr.getValue();
      if (isGroupOrComponentAttr) {
        buffer << LAngleBracket() << delimiter() << identifier << delimiter()
               << equals() << value << RAngleBracket();
      } else {
        buffer << addressSymbol() << identifier;
        // The only time we may omit the value is when it is a Boolean attribute
        // with value 1.
        if (!isBooleanAttribute || intAttr.getValue() != 1) {
          // Retrieve the unsigned representation of the value.
          SmallVector<char, 4> s;
          value.toStringUnsigned(s, /*Radix=*/10);
          buffer << LParen() << s << RParen();
        }
        buffer << space();
      }
    }
    return buffer.str();
  }

  /// Emits the attributes of a dictionary. If the `attributes` dictionary is
  /// not nullptr, we assume this is for a port.
  std::string getAttributes(Operation *op,
                            DictionaryAttr attributes = nullptr) {
    bool isPort = attributes != nullptr;
    if (!isPort)
      attributes = op->getAttrDictionary();

    SmallString<16> calyxAttributes;
    for (auto &attr : attributes)
      calyxAttributes.append(getAttribute(op, attr, isPort));

    return calyxAttributes.c_str();
  }

  /// Helper function for emitting a Calyx section. It emits the body in the
  /// following format:
  /// {
  ///   <body>
  /// }
  template <typename Func>
  void emitCalyxBody(Func emitBody) {
    os << space() << LBraceEndL();
    addIndent();
    emitBody();
    reduceIndent();
    indent() << RBraceEndL();
  }

  /// Emits a Calyx section.
  template <typename Func>
  void emitCalyxSection(StringRef sectionName, Func emitBody,
                        StringRef symbolName = "") {
    indent() << sectionName;
    if (!symbolName.empty())
      os << space() << symbolName;
    emitCalyxBody(emitBody);
  }

  /// Helper function for emitting combinational operations.
  template <typename CombinationalOp>
  void emitCombinationalValue(CombinationalOp op, StringRef logicalSymbol) {
    auto inputs = op.getInputs();
    os << LParen();
    for (size_t i = 0, e = inputs.size(); i != e; ++i) {
      emitValue(inputs[i], /*isIndented=*/false);
      if (i + 1 == e)
        continue;
      os << space() << logicalSymbol << space();
    }
    os << RParen();
  }

  void emitCycleValue(CycleOp op) {
    os << "%";
    if (op.getEnd().has_value()) {
      os << LSquare();
      os << op.getStart() << ":" << op.getEnd();
      os << RSquare();
    } else {
      os << op.getStart();
    }
  }

  /// Emits the value of a guard or assignment.
  void emitValue(Value value, bool isIndented) {
    if (auto blockArg = value.dyn_cast<BlockArgument>()) {
      // Emit component block argument.
      StringAttr portName = getPortInfo(blockArg).name;
      (isIndented ? indent() : os) << portName.getValue();
      return;
    }

    auto definingOp = value.getDefiningOp();
    assert(definingOp && "Value does not have a defining operation.");

    TypeSwitch<Operation *>(definingOp)
        .Case<CellInterface>([&](auto cell) {
          // A cell port should be defined as <instance-name>.<port-name>
          (isIndented ? indent() : os)
              << cell.instanceName() << period() << cell.portName(value);
        })
        .Case<hw::ConstantOp>([&](auto op) {
          // A constant is defined as <bit-width>'<base><value>, where the base
          // is `b` (binary), `o` (octal), `h` hexadecimal, or `d` (decimal).
          APInt value = op.getValue();

          (isIndented ? indent() : os)
              << std::to_string(value.getBitWidth()) << apostrophe() << "d";
          // We currently default to the decimal representation.
          value.print(os, /*isSigned=*/false);
        })
        .Case<comb::AndOp>([&](auto op) { emitCombinationalValue(op, "&"); })
        .Case<comb::OrOp>([&](auto op) { emitCombinationalValue(op, "|"); })
        .Case<comb::XorOp>([&](auto op) {
          // The XorOp is a bit different, since the Combinational dialect
          // uses it to represent binary not.
          if (!op.isBinaryNot()) {
            emitOpError(op, "Only supporting Binary Not for XOR.");
            return;
          }
          // The LHS is the value to be negated, and the RHS is a constant with
          // all ones (guaranteed by isBinaryNot).
          os << exclamationMark();
          emitValue(op.getInputs()[0], /*isIndented=*/false);
        })
        .Case<CycleOp>([&](auto op) { emitCycleValue(op); })
        .Default(
            [&](auto op) { emitOpError(op, "not supported for emission"); });
  }

  /// Emits a port for a Group.
  template <typename OpTy>
  void emitGroupPort(GroupInterface group, OpTy op, StringRef portHole) {
    assert((isa<GroupGoOp>(op) || isa<GroupDoneOp>(op)) &&
           "Required to be a group port.");
    indent() << group.symName().getValue() << LSquare() << portHole << RSquare()
             << space() << equals() << space();
    if (op.getGuard()) {
      emitValue(op.getGuard(), /*isIndented=*/false);
      os << questionMark();
    }
    emitValue(op.getSrc(), /*isIndented=*/false);
    os << semicolonEndL();
  }

  /// Recursively emits the Calyx control.
  void emitCalyxControl(Block *body) {
    Operation *parent = body->getParentOp();
    assert((isa<ControlOp>(parent) || parent->hasTrait<ControlLike>()) &&
           "This should only be used to emit Calyx Control structures.");

    // Check to see if this is a stand-alone EnableOp, i.e.
    // calyx.control { calyx.enable @G }
    if (auto enable = dyn_cast<EnableOp>(parent)) {
      emitEnable(enable);
      // Early return since an EnableOp has no body.
      return;
    }
    // Attribute dictionary is always prepended for a control operation.
    auto prependAttributes = [&](Operation *op, StringRef sym) {
      return (getAttributes(op) + sym).str();
    };

    for (auto &&op : *body) {

      TypeSwitch<Operation *>(&op)
          .Case<SeqOp>([&](auto op) {
            emitCalyxSection(prependAttributes(op, "seq"),
                             [&]() { emitCalyxControl(op.getBodyBlock()); });
          })
          .Case<StaticSeqOp>([&](auto op) {
            emitCalyxSection(prependAttributes(op, "static seq"),
                             [&]() { emitCalyxControl(op.getBodyBlock()); });
          })
          .Case<ParOp>([&](auto op) {
            emitCalyxSection(prependAttributes(op, "par"),
                             [&]() { emitCalyxControl(op.getBodyBlock()); });
          })
          .Case<WhileOp>([&](auto op) {
            indent() << prependAttributes(op, "while ");
            emitValue(op.getCond(), /*isIndented=*/false);

            if (auto groupName = op.getGroupName())
              os << " with " << *groupName;

            emitCalyxBody([&]() { emitCalyxControl(op.getBodyBlock()); });
          })
          .Case<IfOp>([&](auto op) {
            indent() << prependAttributes(op, "if ");
            emitValue(op.getCond(), /*isIndented=*/false);

            if (auto groupName = op.getGroupName())
              os << " with " << *groupName;

            emitCalyxBody([&]() { emitCalyxControl(op.getThenBody()); });
            if (op.elseBodyExists())
              emitCalyxSection("else",
                               [&]() { emitCalyxControl(op.getElseBody()); });
          })
          .Case<StaticIfOp>([&](auto op) {
            indent() << prependAttributes(op, "static if ");
            emitValue(op.getCond(), /*isIndented=*/false);

            emitCalyxBody([&]() { emitCalyxControl(op.getThenBody()); });
            if (op.elseBodyExists())
              emitCalyxSection("else",
                               [&]() { emitCalyxControl(op.getElseBody()); });
          })
          .Case<RepeatOp>([&](auto op) {
            indent() << prependAttributes(op, "repeat ");
            os << op.getCount();

            emitCalyxBody([&]() { emitCalyxControl(op.getBodyBlock()); });
          })
          .Case<StaticRepeatOp>([&](auto op) {
            indent() << prependAttributes(op, "static repeat ");
            os << op.getCount();

            emitCalyxBody([&]() { emitCalyxControl(op.getBodyBlock()); });
          })
          .Case<StaticParOp>([&](auto op) {
            emitCalyxSection(prependAttributes(op, "static par"),
                             [&]() { emitCalyxControl(op.getBodyBlock()); });
          })
          .Case<EnableOp>([&](auto op) { emitEnable(op); })
          .Case<InvokeOp>([&](auto op) { emitInvoke(op); })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside control.");
          });
    }
  }

  /// The stream we are emitting into.
  llvm::raw_ostream &os;

  /// Whether we have encountered any errors during emission.
  bool encounteredError = false;

  /// Current level of indentation. See `indent()` and
  /// `addIndent()`/`reduceIndent()`.
  unsigned currentIndent = 0;
};

} // end anonymous namespace

LogicalResult Emitter::finalize() { return failure(encounteredError); }

/// Emit an entire program.
void Emitter::emitModule(ModuleOp op) {
  for (auto &bodyOp : *op.getBody()) {
    if (auto componentOp = dyn_cast<ComponentInterface>(bodyOp))
      emitComponent(componentOp);
    else if (auto hwModuleExternOp = dyn_cast<hw::HWModuleExternOp>(bodyOp))
      emitPrimitiveExtern(hwModuleExternOp);
    else
      emitOpError(&bodyOp, "Unexpected op");
  }
}

/// Emit a component.
void Emitter::emitComponent(ComponentInterface op) {
  std::string combinationalPrefix = op.isComb() ? "comb " : "";

  indent() << combinationalPrefix << "component " << op.getName()
           << getAttributes(op);
  // Emit the ports.
  emitComponentPorts(op);
  os << space() << LBraceEndL();
  addIndent();
  WiresOp wires;
  ControlOp control;

  // Emit cells.
  emitCalyxSection("cells", [&]() {
    for (auto &&bodyOp : *op.getBodyBlock()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<WiresOp>([&](auto op) { wires = op; })
          .Case<ControlOp>([&](auto op) { control = op; })
          .Case<InstanceOp>([&](auto op) { emitInstance(op); })
          .Case<PrimitiveOp>([&](auto op) { emitPrimitive(op); })
          .Case<RegisterOp>([&](auto op) { emitRegister(op); })
          .Case<MemoryOp>([&](auto op) { emitMemory(op); })
          .Case<SeqMemoryOp>([&](auto op) { emitSeqMemory(op); })
          .Case<hw::ConstantOp>([&](auto op) { /*Do nothing*/ })
          .Case<SliceLibOp, PadLibOp>(
              [&](auto op) { emitLibraryPrimTypedByAllPorts(op); })
          .Case<LtLibOp, GtLibOp, EqLibOp, NeqLibOp, GeLibOp, LeLibOp, SltLibOp,
                SgtLibOp, SeqLibOp, SneqLibOp, SgeLibOp, SleLibOp, AddLibOp,
                SubLibOp, ShruLibOp, RshLibOp, SrshLibOp, LshLibOp, AndLibOp,
                NotLibOp, OrLibOp, XorLibOp, WireLibOp>(
              [&](auto op) { emitLibraryPrimTypedByFirstInputPort(op); })
          .Case<MultPipeLibOp>(
              [&](auto op) { emitLibraryPrimTypedByFirstOutputPort(op); })
          .Case<RemUPipeLibOp, DivUPipeLibOp>([&](auto op) {
            emitLibraryPrimTypedByFirstOutputPort(
                op, /*calyxLibName=*/{"std_div_pipe"});
          })
          .Case<RemSPipeLibOp, DivSPipeLibOp>([&](auto op) {
            emitLibraryPrimTypedByFirstOutputPort(
                op, /*calyxLibName=*/{"std_sdiv_pipe"});
          })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside component");
          });
    }
  });

  emitWires(wires);
  emitControl(control);
  reduceIndent();
  os << RBraceEndL();
}

/// Emit the ports of a component.
void Emitter::emitComponentPorts(ComponentInterface op) {
  auto emitPorts = [&](auto ports) {
    os << LParen();
    for (size_t i = 0, e = ports.size(); i < e; ++i) {
      const PortInfo &port = ports[i];

      // We only care about the bit width in the emitted .futil file.
      unsigned int bitWidth = port.type.getIntOrFloatBitWidth();
      os << getAttributes(op, port.attributes) << port.name.getValue()
         << colon() << bitWidth;

      if (i + 1 < e)
        os << comma();
    }
    os << RParen();
  };
  emitPorts(op.getInputPortInfo());
  os << arrow();
  emitPorts(op.getOutputPortInfo());
}

/// Emit a primitive extern
void Emitter::emitPrimitiveExtern(hw::HWModuleExternOp op) {
  Attribute filename = op->getAttrDictionary().get("filename");
  indent() << "extern " << filename << space() << LBraceEndL();
  addIndent();
  indent() << "primitive " << op.getName();

  if (!op.getParameters().empty()) {
    os << LSquare();
    llvm::interleaveComma(op.getParameters(), os, [&](Attribute param) {
      auto paramAttr = param.cast<hw::ParamDeclAttr>();
      os << paramAttr.getName().str();
    });
    os << RSquare();
  }
  os << getAttributes(op);
  // Emit the ports.
  emitPrimitivePorts(op);
  os << semicolonEndL();
  reduceIndent();
  os << RBraceEndL();
}

/// Emit the ports of a component.
void Emitter::emitPrimitivePorts(hw::HWModuleExternOp op) {
  auto emitPorts = [&](auto ports, bool isInput) {
    auto e = static_cast<size_t>(std::distance(ports.begin(), ports.end()));
    os << LParen();
    for (auto [i, port] : llvm::enumerate(ports)) {
      DictionaryAttr portAttr =
          isInput ? op.getArgAttrDict(i) : op.getResultAttrDict(i);

      os << getAttributes(op, portAttr) << port.name.getValue() << colon();
      // We only care about the bit width in the emitted .futil file.
      // Emit parameterized or non-parameterized bit width.
      if (hw::isParametricType(port.type)) {
        hw::ParamDeclRefAttr bitWidth =
            port.type.template cast<hw::IntType>()
                .getWidth()
                .template dyn_cast<hw::ParamDeclRefAttr>();
        os << bitWidth.getName().str();
      } else {
        unsigned int bitWidth = port.type.getIntOrFloatBitWidth();
        os << bitWidth;
      }

      if (i < e - 1)
        os << comma();
    }
    os << RParen();
  };
  auto ports = op.getPortList();
  emitPorts(ports.getInputs(), true);
  os << arrow();
  emitPorts(ports.getOutputs(), false);
}

void Emitter::emitInstance(InstanceOp op) {
  indent() << getAttributes(op) << op.instanceName() << space() << equals()
           << space() << op.getComponentName() << LParen() << RParen()
           << semicolonEndL();
}

void Emitter::emitPrimitive(PrimitiveOp op) {
  indent() << getAttributes(op) << op.instanceName() << space() << equals()
           << space() << op.getPrimitiveName() << LParen();

  if (op.getParameters().has_value()) {
    llvm::interleaveComma(*op.getParameters(), os, [&](Attribute param) {
      auto paramAttr = param.cast<hw::ParamDeclAttr>();
      auto value = paramAttr.getValue();
      if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
        os << intAttr.getInt();
      } else if (auto fpAttr = value.dyn_cast<FloatAttr>()) {
        os << fpAttr.getValue().convertToFloat();
      } else {
        llvm_unreachable("Primitive parameter type not supported");
      }
    });
  }

  os << RParen() << semicolonEndL();
}

void Emitter::emitRegister(RegisterOp reg) {
  size_t bitWidth = reg.getIn().getType().getIntOrFloatBitWidth();
  indent() << getAttributes(reg) << reg.instanceName() << space() << equals()
           << space() << "std_reg" << LParen() << std::to_string(bitWidth)
           << RParen() << semicolonEndL();
}

void Emitter::emitMemory(MemoryOp memory) {
  size_t dimension = memory.getSizes().size();
  if (dimension < 1 || dimension > 4) {
    emitOpError(memory, "Only memories with dimensionality in range [1, 4] are "
                        "supported by the native Calyx compiler.");
    return;
  }
  indent() << getAttributes(memory) << memory.instanceName() << space()
           << equals() << space() << "std_mem_d" << std::to_string(dimension)
           << LParen() << memory.getWidth() << comma();
  for (Attribute size : memory.getSizes()) {
    APInt memSize = size.cast<IntegerAttr>().getValue();
    memSize.print(os, /*isSigned=*/false);
    os << comma();
  }

  ArrayAttr addrSizes = memory.getAddrSizes();
  for (size_t i = 0, e = addrSizes.size(); i != e; ++i) {
    APInt addrSize = addrSizes[i].cast<IntegerAttr>().getValue();
    addrSize.print(os, /*isSigned=*/false);
    if (i + 1 == e)
      continue;
    os << comma();
  }
  os << RParen() << semicolonEndL();
}

void Emitter::emitSeqMemory(SeqMemoryOp memory) {
  size_t dimension = memory.getSizes().size();
  if (dimension < 1 || dimension > 4) {
    emitOpError(memory, "Only memories with dimensionality in range [1, 4] are "
                        "supported by the native Calyx compiler.");
    return;
  }
  indent() << getAttributes(memory) << memory.instanceName() << space()
           << equals() << space() << "seq_mem_d" << std::to_string(dimension)
           << LParen() << memory.getWidth() << comma();
  for (Attribute size : memory.getSizes()) {
    APInt memSize = size.cast<IntegerAttr>().getValue();
    memSize.print(os, /*isSigned=*/false);
    os << comma();
  }

  ArrayAttr addrSizes = memory.getAddrSizes();
  for (size_t i = 0, e = addrSizes.size(); i != e; ++i) {
    APInt addrSize = addrSizes[i].cast<IntegerAttr>().getValue();
    addrSize.print(os, /*isSigned=*/false);
    if (i + 1 == e)
      continue;
    os << comma();
  }
  os << RParen() << semicolonEndL();
}

void Emitter::emitInvoke(InvokeOp invoke) {
  StringRef callee = invoke.getCallee();
  indent() << "invoke " << callee;
  ArrayAttr portNames = invoke.getPortNames();
  ArrayAttr inputNames = invoke.getInputNames();
  /// Because the ports of all components of calyx.invoke are inside a (),
  /// here the input and output ports are divided, inputs and outputs store
  /// the connections for a subset of input and output ports of the instance.
  llvm::StringMap<std::string> inputsMap;
  llvm::StringMap<std::string> outputsMap;
  for (auto [portNameAttr, inputNameAttr, input] :
       llvm::zip(portNames, inputNames, invoke.getInputs())) {
    StringRef portName = cast<StringAttr>(portNameAttr).getValue();
    StringRef inputName = cast<StringAttr>(inputNameAttr).getValue();
    /// Classify the connection of ports,here's an example. calyx.invoke
    /// @r(%r.in = %id.out, %out = %r.out) -> (i32, i32) %r.in = %id.out will be
    /// stored in inputs, because %.r.in is the input port of the component, and
    /// %out = %r.out will be stored in outputs, because %r.out is the output
    /// port of the component, which is a bit different from calyx's native
    /// compiler. Later on, the classified connection relations are outputted
    /// uniformly and converted to calyx's native compiler format.
    StringRef inputMapKey = portName.drop_front(2 + callee.size());
    if (portName.substr(1, callee.size()) == callee) {
      // If the input to the port is a number.
      if (isa_and_nonnull<hw::ConstantOp>(input.getDefiningOp())) {
        hw::ConstantOp constant = cast<hw::ConstantOp>(input.getDefiningOp());
        APInt value = constant.getValue();
        std::string mapValue = std::to_string(value.getBitWidth()) +
                               apostrophe().data() + "d" +
                               std::to_string(value.getZExtValue());
        inputsMap[inputMapKey] = mapValue;
        continue;
      }
      inputsMap[inputMapKey] = inputName.drop_front(1).str();
    } else if (inputName.substr(1, callee.size()) == callee)
      outputsMap[inputName.drop_front(2 + callee.size())] =
          portName.drop_front(1).str();
  }
  /// Emit inputs
  os << LParen();
  llvm::interleaveComma(inputsMap, os, [&](const auto &iter) {
    os << iter.getKey() << " = " << iter.getValue();
  });
  os << RParen();
  /// Emit outputs
  os << LParen();
  llvm::interleaveComma(outputsMap, os, [&](const auto &iter) {
    os << iter.getKey() << " = " << iter.getValue();
  });
  os << RParen() << semicolonEndL();
}

/// Calling getName() on a calyx operation will return "calyx.${opname}". This
/// function returns whatever is left after the first '.' in the string,
/// removing the 'calyx' prefix.
static StringRef removeCalyxPrefix(StringRef s) { return s.split(".").second; }

void Emitter::emitLibraryPrimTypedByAllPorts(Operation *op) {
  auto cell = cast<CellInterface>(op);
  indent() << getAttributes(op) << cell.instanceName() << space() << equals()
           << space() << removeCalyxPrefix(op->getName().getStringRef())
           << LParen();
  llvm::interleaveComma(op->getResults(), os, [&](auto res) {
    os << std::to_string(res.getType().getIntOrFloatBitWidth());
  });
  os << RParen() << semicolonEndL();
}

void Emitter::emitLibraryPrimTypedByFirstInputPort(Operation *op) {
  auto cell = cast<CellInterface>(op);
  unsigned bitWidth = cell.getInputPorts()[0].getType().getIntOrFloatBitWidth();
  StringRef opName = op->getName().getStringRef();
  indent() << getAttributes(op) << cell.instanceName() << space() << equals()
           << space() << removeCalyxPrefix(opName) << LParen() << bitWidth
           << RParen() << semicolonEndL();
}

void Emitter::emitLibraryPrimTypedByFirstOutputPort(
    Operation *op, std::optional<StringRef> calyxLibName) {
  auto cell = cast<CellInterface>(op);
  unsigned bitWidth =
      cell.getOutputPorts()[0].getType().getIntOrFloatBitWidth();
  StringRef opName = op->getName().getStringRef();
  indent() << getAttributes(op) << cell.instanceName() << space() << equals()
           << space()
           << (calyxLibName ? *calyxLibName : removeCalyxPrefix(opName))
           << LParen() << bitWidth << RParen() << semicolonEndL();
}

void Emitter::emitAssignment(AssignOp op) {

  emitValue(op.getDest(), /*isIndented=*/true);
  os << space() << equals() << space();
  if (op.getGuard()) {
    emitValue(op.getGuard(), /*isIndented=*/false);
    os << questionMark();
  }
  emitValue(op.getSrc(), /*isIndented=*/false);
  os << semicolonEndL();
}

void Emitter::emitWires(WiresOp op) {
  emitCalyxSection("wires", [&]() {
    for (auto &&bodyOp : *op.getBodyBlock()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<GroupInterface>([&](auto op) { emitGroup(op); })
          .Case<AssignOp>([&](auto op) { emitAssignment(op); })
          .Case<hw::ConstantOp, comb::AndOp, comb::OrOp, comb::XorOp, CycleOp>(
              [&](auto op) { /* Do nothing. */ })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside wires section");
          });
    }
  });
}

void Emitter::emitGroup(GroupInterface group) {
  auto emitGroupBody = [&]() {
    for (auto &&bodyOp : *group.getBody()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<AssignOp>([&](auto op) { emitAssignment(op); })
          .Case<GroupDoneOp>([&](auto op) { emitGroupPort(group, op, "done"); })
          .Case<GroupGoOp>([&](auto op) { emitGroupPort(group, op, "go"); })
          .Case<hw::ConstantOp, comb::AndOp, comb::OrOp, comb::XorOp, CycleOp>(
              [&](auto op) { /* Do nothing. */ })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside group.");
          });
    }
  };
  std::string prefix;
  if (isa<StaticGroupOp>(group)) {
    auto staticGroup = cast<StaticGroupOp>(group);
    prefix = llvm::formatv("static<{0}> group", staticGroup.getLatency());
  } else {
    prefix = isa<CombGroupOp>(group) ? "comb group" : "group";
  }
  auto groupHeader = (group.symName().getValue() + getAttributes(group)).str();
  emitCalyxSection(prefix, emitGroupBody, groupHeader);
}

void Emitter::emitEnable(EnableOp enable) {
  indent() << getAttributes(enable) << enable.getGroupName() << semicolonEndL();
}

void Emitter::emitControl(ControlOp control) {
  // A valid Calyx program does not necessarily need a control section.
  if (control == nullptr)
    return;
  emitCalyxSection("control",
                   [&]() { emitCalyxControl(control.getBodyBlock()); });
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Emit the specified Calyx circuit into the given output stream.
mlir::LogicalResult circt::calyx::exportCalyx(mlir::ModuleOp module,
                                              llvm::raw_ostream &os) {
  Emitter emitter(os);
  if (failed(emitter.emitImports(module)))
    return failure();
  emitter.emitModule(module);
  emitter.emitCiderMetadata(module);
  return emitter.finalize();
}

void circt::calyx::registerToCalyxTranslation() {
  static mlir::TranslateFromMLIRRegistration toCalyx(
      "export-calyx", "export Calyx",
      [](ModuleOp module, llvm::raw_ostream &os) {
        return exportCalyx(module, os);
      },
      [](mlir::DialectRegistry &registry) {
        registry
            .insert<calyx::CalyxDialect, comb::CombDialect, hw::HWDialect>();
      });
}
