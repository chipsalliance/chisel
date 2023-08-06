//===- FIREmitter.cpp - FIRRTL dialect to .fir emitter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a .fir file emitter.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIREmitter.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/PrettyPrinterHelpers.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "export-firrtl"

using namespace circt;
using namespace firrtl;
using namespace chirrtl;
using namespace pretty;

//===----------------------------------------------------------------------===//
// Emitter
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(misc-no-recursion)
namespace {

constexpr size_t defaultTargetLineLength = 80;

/// An emitter for FIRRTL dialect operations to .fir output.
struct Emitter {
  Emitter(llvm::raw_ostream &os, FIRVersion version,
          size_t targetLineLength = defaultTargetLineLength)
      : pp(os, targetLineLength), ps(pp, saver), version(version) {
    pp.setListener(&saver);
  }
  LogicalResult finalize();

  // Circuit/module emission
  void emitCircuit(CircuitOp op);
  void emitModule(FModuleOp op);
  void emitModule(FExtModuleOp op);
  void emitModule(FIntModuleOp op);
  void emitModulePorts(ArrayRef<PortInfo> ports,
                       Block::BlockArgListType arguments = {});
  void emitModuleParameters(Operation *op, ArrayAttr parameters);
  void emitDeclaration(GroupDeclOp op);

  // Statement emission
  void emitStatementsInBlock(Block &block);
  void emitStatement(WhenOp op);
  void emitStatement(WireOp op);
  void emitStatement(RegOp op);
  void emitStatement(RegResetOp op);
  void emitStatement(NodeOp op);
  void emitStatement(StopOp op);
  void emitStatement(SkipOp op);
  void emitStatement(PrintFOp op);
  void emitStatement(ConnectOp op);
  void emitStatement(StrictConnectOp op);
  void emitStatement(PropAssignOp op);
  void emitStatement(InstanceOp op);
  void emitStatement(AttachOp op);
  void emitStatement(MemOp op);
  void emitStatement(InvalidValueOp op);
  void emitStatement(CombMemOp op);
  void emitStatement(SeqMemOp op);
  void emitStatement(MemoryPortOp op);
  void emitStatement(MemoryDebugPortOp op);
  void emitStatement(MemoryPortAccessOp op);
  void emitStatement(RefDefineOp op);
  void emitStatement(RefForceOp op);
  void emitStatement(RefForceInitialOp op);
  void emitStatement(RefReleaseOp op);
  void emitStatement(RefReleaseInitialOp op);
  void emitStatement(GroupOp op);

  template <class T>
  void emitVerifStatement(T op, StringRef mnemonic);
  void emitStatement(AssertOp op) { emitVerifStatement(op, "assert"); }
  void emitStatement(AssumeOp op) { emitVerifStatement(op, "assume"); }
  void emitStatement(CoverOp op) { emitVerifStatement(op, "cover"); }

  // Exprsesion emission
  void emitExpression(Value value);
  void emitExpression(ConstantOp op);
  void emitExpression(SpecialConstantOp op);
  void emitExpression(SubfieldOp op);
  void emitExpression(SubindexOp op);
  void emitExpression(SubaccessOp op);
  void emitExpression(OpenSubfieldOp op);
  void emitExpression(OpenSubindexOp op);
  void emitExpression(RefResolveOp op);
  void emitExpression(RefSendOp op);
  void emitExpression(RefSubOp op);
  void emitExpression(UninferredResetCastOp op);
  void emitExpression(ConstCastOp op);
  void emitExpression(StringConstantOp op);
  void emitExpression(FIntegerConstantOp op);

  void emitPrimExpr(StringRef mnemonic, Operation *op,
                    ArrayRef<uint32_t> attrs = {});

  void emitExpression(BitsPrimOp op) {
    emitPrimExpr("bits", op, {op.getHi(), op.getLo()});
  }
  void emitExpression(HeadPrimOp op) {
    emitPrimExpr("head", op, op.getAmount());
  }
  void emitExpression(TailPrimOp op) {
    emitPrimExpr("tail", op, op.getAmount());
  }
  void emitExpression(PadPrimOp op) { emitPrimExpr("pad", op, op.getAmount()); }
  void emitExpression(ShlPrimOp op) { emitPrimExpr("shl", op, op.getAmount()); }
  void emitExpression(ShrPrimOp op) { emitPrimExpr("shr", op, op.getAmount()); }

  // Funnel all ops without attrs into `emitPrimExpr`.
#define HANDLE(OPTYPE, MNEMONIC)                                               \
  void emitExpression(OPTYPE op) { emitPrimExpr(MNEMONIC, op); }
  HANDLE(AddPrimOp, "add");
  HANDLE(SubPrimOp, "sub");
  HANDLE(MulPrimOp, "mul");
  HANDLE(DivPrimOp, "div");
  HANDLE(RemPrimOp, "rem");
  HANDLE(AndPrimOp, "and");
  HANDLE(OrPrimOp, "or");
  HANDLE(XorPrimOp, "xor");
  HANDLE(LEQPrimOp, "leq");
  HANDLE(LTPrimOp, "lt");
  HANDLE(GEQPrimOp, "geq");
  HANDLE(GTPrimOp, "gt");
  HANDLE(EQPrimOp, "eq");
  HANDLE(NEQPrimOp, "neq");
  HANDLE(CatPrimOp, "cat");
  HANDLE(DShlPrimOp, "dshl");
  HANDLE(DShlwPrimOp, "dshlw");
  HANDLE(DShrPrimOp, "dshr");
  HANDLE(MuxPrimOp, "mux");
  HANDLE(AsSIntPrimOp, "asSInt");
  HANDLE(AsUIntPrimOp, "asUInt");
  HANDLE(AsAsyncResetPrimOp, "asAsyncReset");
  HANDLE(AsClockPrimOp, "asClock");
  HANDLE(CvtPrimOp, "cvt");
  HANDLE(NegPrimOp, "neg");
  HANDLE(NotPrimOp, "not");
  HANDLE(AndRPrimOp, "andr");
  HANDLE(OrRPrimOp, "orr");
  HANDLE(XorRPrimOp, "xorr");
#undef HANDLE

  // Attributes
  void emitAttribute(MemDirAttr attr);
  void emitAttribute(RUWAttr attr);

  // Types
  void emitType(Type type, bool includeConst = true);
  void emitTypeWithColon(Type type) {
    ps << PP::space << ":" << PP::nbsp;
    emitType(type);
  }

  // Locations
  void emitLocation(Location loc);
  void emitLocation(Operation *op) { emitLocation(op->getLoc()); }
  template <typename... Args>
  void emitLocationAndNewLine(Args... args) {
    // Break so previous content is not impacted by following,
    // but use a 'neverbreak' so it always fits.
    ps << PP::neverbreak;
    emitLocation(args...);
    setPendingNewline();
  }

  void emitAssignLike(llvm::function_ref<void()> emitLHS,
                      llvm::function_ref<void()> emitRHS,
                      PPExtString syntax = PPExtString("="),
                      std::optional<PPExtString> wordBeforeLHS = std::nullopt) {
    // If wraps, indent.
    ps.scopedBox(PP::ibox2, [&]() {
      if (wordBeforeLHS) {
        ps << *wordBeforeLHS << PP::space;
      }
      emitLHS();
      // Allow breaking before 'syntax' (e.g., '=') if long assignment.
      ps << PP::space << syntax << PP::nbsp; /* PP::space; */
      // RHS is boxed to right of the syntax.
      ps.scopedBox(PP::ibox0, [&]() { emitRHS(); });
    });
  }

  /// Emit the specified value as a subexpression, wrapping in an ibox2.
  void emitSubExprIBox2(Value v) {
    ps.scopedBox(PP::ibox2, [&]() { emitExpression(v); });
  }

  /// Emit a range of values separated by commas and a breakable space.
  /// Each value is emitted by invoking `eachFn`.
  template <typename Container, typename EachFn>
  void interleaveComma(const Container &c, EachFn eachFn) {
    llvm::interleave(c, eachFn, [&]() { ps << "," << PP::space; });
  }

  /// Emit a range of values separated by commas and a breakable space.
  /// Each value is emitted in an ibox2.
  void interleaveComma(ValueRange ops) {
    return interleaveComma(ops, [&](Value v) { emitSubExprIBox2(v); });
  }

  void emitStatementFunctionOp(PPExtString name, Operation *op) {
    startStatement();
    ps << name << "(";
    ps.scopedBox(PP::ibox0, [&]() {
      interleaveComma(op->getOperands());
      ps << ")";
    });
    emitLocationAndNewLine(op);
  }

private:
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

  /// Return the name used during emission of a `Value`, or none if the value
  /// has not yet been emitted or it was emitted inline.
  std::optional<StringRef> lookupEmittedName(Value value) {
    auto it = valueNames.find(value);
    if (it != valueNames.end())
      return {it->second};
    return {};
  }

  /// If previous emission requires a newline, emit it now.
  /// This gives us opportunity to open/close boxes before linebreak.
  void emitPendingNewlineIfNeeded() {
    if (pendingNewline) {
      pendingNewline = false;
      ps << PP::newline;
    }
  }
  void setPendingNewline() {
    assert(!pendingNewline);
    pendingNewline = true;
  }

  void startStatement() { emitPendingNewlineIfNeeded(); }

private:
  /// String storage backing Tokens built from temporary strings.
  /// PrettyPrinter will clear this as appropriate.
  TokenStringSaver saver;

  /// Pretty printer.
  PrettyPrinter pp;

  /// Stream helper (pp, saver).
  TokenStream<> ps;

  /// Whether a newline is expected, emitted late to provide opportunity to
  /// open/close boxes we don't know we need at level of individual statement.
  /// Every statement should set this instead of directly emitting (last)
  /// newline. Most statements end with emitLocationInfoAndNewLine which handles
  /// this.
  bool pendingNewline = false;

  /// Whether we have encountered any errors during emission.
  bool encounteredError = false;

  /// The names used to emit values already encountered. Anything that gets a
  /// name in the output FIR is listed here, such that future expressions can
  /// reference it.
  DenseMap<Value, StringRef> valueNames;
  StringSet<> valueNamesStorage;

  /// Legalize names for emission.  Convert names which begin with a number to
  /// be escaped using backticks.
  StringAttr legalize(StringAttr attr) {
    StringRef str = attr.getValue();
    if (str.empty() || !isdigit(str.front()))
      return attr;
    return StringAttr::get(attr.getContext(), "`" + Twine(attr) + "`");
  }

  void addValueName(Value value, StringAttr attr) {
    valueNames.insert({value, attr.getValue()});
  }
  void addValueName(Value value, StringRef str) {
    auto it = valueNamesStorage.insert(str);
    valueNames.insert({value, it.first->getKey()});
  }

  /// The current circuit namespace valid within the call to `emitCircuit`.
  CircuitNamespace circuitNamespace;

  /// The version of the FIRRTL spec that should be emitted.
  FIRVersion version;
};
} // namespace

LogicalResult Emitter::finalize() { return failure(encounteredError); }

/// Emit an entire circuit.
void Emitter::emitCircuit(CircuitOp op) {
  circuitNamespace.add(op);
  startStatement();
  ps << "FIRRTL version ";
  ps.addAsString(version.major);
  ps << ".";
  ps.addAsString(version.minor);
  ps << ".";
  ps.addAsString(version.patch);
  ps << PP::newline;
  ps << "circuit " << PPExtString(legalize(op.getNameAttr())) << " :";
  setPendingNewline();
  ps.scopedBox(PP::bbox2, [&]() {
    for (auto &bodyOp : *op.getBodyBlock()) {
      if (encounteredError)
        break;
      TypeSwitch<Operation *>(&bodyOp)
          .Case<FModuleOp, FExtModuleOp, FIntModuleOp>([&](auto op) {
            emitModule(op);
            ps << PP::newline;
          })
          .Case<GroupDeclOp>([&](auto op) { emitDeclaration(op); })
          .Default([&](auto op) {
            emitOpError(op, "not supported for emission inside circuit");
          });
    }
  });
  circuitNamespace.clear();
}

/// Emit an entire module.
void Emitter::emitModule(FModuleOp op) {
  startStatement();
  ps << "module " << PPExtString(legalize(op.getNameAttr())) << " :";
  emitLocation(op);
  ps.scopedBox(PP::bbox2, [&]() {
    setPendingNewline();

    // Emit the ports.
    auto ports = op.getPorts();
    emitModulePorts(ports, op.getArguments());
    if (!ports.empty() && !op.getBodyBlock()->empty())
      ps << PP::newline;

    // Emit the module body.
    emitStatementsInBlock(*op.getBodyBlock());
  });
  valueNames.clear();
  valueNamesStorage.clear();
}

/// Emit an external module.
void Emitter::emitModule(FExtModuleOp op) {
  startStatement();
  ps << "extmodule " << PPExtString(legalize(op.getNameAttr())) << " :";
  emitLocation(op);
  ps.scopedBox(PP::bbox2, [&]() {
    setPendingNewline();

    // Emit the ports.
    auto ports = op.getPorts();
    emitModulePorts(ports);

    // Emit the optional `defname`.
    if (op.getDefname() && !op.getDefname()->empty()) {
      startStatement();
      ps << "defname = " << PPExtString(*op.getDefname());
      setPendingNewline();
    }

    // Emit the parameters.
    emitModuleParameters(op, op.getParameters());
  });
}

/// Emit an intrinsic module
void Emitter::emitModule(FIntModuleOp op) {
  startStatement();
  ps << "intmodule " << PPExtString(legalize(op.getNameAttr())) << " :";
  emitLocation(op);
  ps.scopedBox(PP::bbox2, [&]() {
    setPendingNewline();

    // Emit the ports.
    auto ports = op.getPorts();
    emitModulePorts(ports);

    // Emit the optional intrinsic.
    //
    // TODO: This really shouldn't be optional, but it is currently encoded like
    // this.
    if (op.getIntrinsic().has_value()) {
      auto intrinsic = *op.getIntrinsic();
      if (!intrinsic.empty()) {
        startStatement();
        ps << "intrinsic = " << PPExtString(*op.getIntrinsic());
        setPendingNewline();
      }
    }

    // Emit the parameters.
    emitModuleParameters(op, op.getParameters());
  });
}

/// Emit the ports of a module or extmodule. If the `arguments` array is
/// non-empty, it is used to populate `emittedNames` with the port names for use
/// during expression emission.
void Emitter::emitModulePorts(ArrayRef<PortInfo> ports,
                              Block::BlockArgListType arguments) {
  for (unsigned i = 0, e = ports.size(); i < e; ++i) {
    startStatement();
    const auto &port = ports[i];
    ps << (port.direction == Direction::In ? "input " : "output ");
    auto legalName = legalize(port.name);
    if (!arguments.empty())
      addValueName(arguments[i], legalName);
    ps << PPExtString(legalName) << " : ";
    emitType(port.type);
    emitLocation(ports[i].loc);
    setPendingNewline();
  }
}

void Emitter::emitModuleParameters(Operation *op, ArrayAttr parameters) {
  for (auto param : llvm::map_range(parameters, [](Attribute attr) {
         return cast<ParamDeclAttr>(attr);
       })) {
    startStatement();
    // TODO: AssignLike ?
    ps << "parameter " << PPExtString(param.getName().getValue()) << " = ";
    TypeSwitch<Attribute>(param.getValue())
        .Case<IntegerAttr>([&](auto attr) { ps.addAsString(attr.getValue()); })
        .Case<FloatAttr>([&](auto attr) {
          SmallString<16> str;
          attr.getValue().toString(str);
          ps << str;
        })
        .Case<StringAttr>(
            [&](auto attr) { ps.writeQuotedEscaped(attr.getValue()); })
        .Default([&](auto attr) {
          emitOpError(op, "with unsupported parameter attribute: ") << attr;
          ps << "<unsupported-attr ";
          ps.addAsString(attr);
          ps << ">";
        });
    setPendingNewline();
  }
}

/// Emit an optional group declaration.
void Emitter::emitDeclaration(GroupDeclOp op) {
  startStatement();
  ps << "declgroup " << PPExtString(op.getSymName()) << ", "
     << PPExtString(stringifyGroupConvention(op.getConvention())) << " : ";
  emitLocationAndNewLine(op);
  ps.scopedBox(PP::bbox2, [&]() {
    for (auto &bodyOp : op.getBody().getOps()) {
      TypeSwitch<Operation *>(&bodyOp)
          .Case<GroupDeclOp>([&](auto op) { emitDeclaration(op); })
          .Default([&](auto op) {
            emitOpError(op,
                        "not supported for emission inside group declaration");
          });
    }
  });
}

/// Check if an operation is inlined into the emission of their users. For
/// example, subfields are always inlined.
static bool isEmittedInline(Operation *op) {
  return isExpression(op) && !isa<InvalidValueOp>(op);
}

void Emitter::emitStatementsInBlock(Block &block) {
  for (auto &bodyOp : block) {
    if (encounteredError)
      return;
    if (isEmittedInline(&bodyOp))
      continue;
    TypeSwitch<Operation *>(&bodyOp)
        .Case<WhenOp, WireOp, RegOp, RegResetOp, NodeOp, StopOp, SkipOp,
              PrintFOp, AssertOp, AssumeOp, CoverOp, ConnectOp, StrictConnectOp,
              PropAssignOp, InstanceOp, AttachOp, MemOp, InvalidValueOp,
              SeqMemOp, CombMemOp, MemoryPortOp, MemoryDebugPortOp,
              MemoryPortAccessOp, RefDefineOp, RefForceOp, RefForceInitialOp,
              RefReleaseOp, RefReleaseInitialOp, GroupOp>(
            [&](auto op) { emitStatement(op); })
        .Default([&](auto op) {
          startStatement();
          ps << "// operation " << PPExtString(op->getName().getStringRef());
          setPendingNewline();
          emitOpError(op, "not supported as statement");
        });
  }
}

void Emitter::emitStatement(WhenOp op) {
  startStatement();
  ps << "when ";
  emitExpression(op.getCondition());
  ps << " :";
  emitLocationAndNewLine(op);
  ps.scopedBox(PP::bbox2, [&]() { emitStatementsInBlock(op.getThenBlock()); });
  // emitStatementsInBlock(op.getThenBlock());
  if (!op.hasElseRegion())
    return;

  startStatement();
  ps << "else ";
  // Sugar to `else when ...` if there's only a single when statement in the
  // else block.
  auto &elseBlock = op.getElseBlock();
  if (!elseBlock.empty() && &elseBlock.front() == &elseBlock.back()) {
    if (auto whenOp = dyn_cast<WhenOp>(&elseBlock.front())) {
      emitStatement(whenOp);
      return;
    }
  }
  // Otherwise print the block as `else :`.
  ps << ":";
  setPendingNewline();
  ps.scopedBox(PP::bbox2, [&]() { emitStatementsInBlock(elseBlock); });
}

void Emitter::emitStatement(WireOp op) {
  auto legalName = legalize(op.getNameAttr());
  addValueName(op.getResult(), legalName);
  startStatement();
  ps.scopedBox(PP::ibox2, [&]() {
    ps << "wire " << PPExtString(legalName);
    emitTypeWithColon(op.getResult().getType());
  });
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(RegOp op) {
  auto legalName = legalize(op.getNameAttr());
  addValueName(op.getResult(), legalName);
  startStatement();
  ps.scopedBox(PP::ibox2, [&]() {
    ps << "reg " << PPExtString(legalName);
    emitTypeWithColon(op.getResult().getType());
    ps << "," << PP::space;
    emitExpression(op.getClockVal());
  });
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(RegResetOp op) {
  auto legalName = legalize(op.getNameAttr());
  addValueName(op.getResult(), legalName);
  startStatement();
  if (FIRVersion::compare(version, {3, 0, 0}) >= 0) {
    ps.scopedBox(PP::ibox2, [&]() {
      ps << "regreset " << legalName;
      emitTypeWithColon(op.getResult().getType());
      ps << "," << PP::space;
      emitExpression(op.getClockVal());
      ps << "," << PP::space;
      emitExpression(op.getResetSignal());
      ps << "," << PP::space;
      emitExpression(op.getResetValue());
    });
  } else {
    ps.scopedBox(PP::ibox2, [&]() {
      ps << "reg " << legalName;
      emitTypeWithColon(op.getResult().getType());
      ps << "," << PP::space;
      emitExpression(op.getClockVal());
      ps << PP::space << "with :";
      // Don't break this because of the newline.
      ps << PP::neverbreak;
      // No-paren version must be newline + indent.
      ps << PP::newline; // ibox2 will indent.
      ps << "reset => (" << PP::ibox0;
      emitExpression(op.getResetSignal());
      ps << "," << PP::space;
      emitExpression(op.getResetValue());
      ps << ")" << PP::end;
    });
  }
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(NodeOp op) {
  auto legalName = legalize(op.getNameAttr());
  addValueName(op.getResult(), legalName);
  startStatement();
  emitAssignLike([&]() { ps << "node " << PPExtString(legalName); },
                 [&]() { emitExpression(op.getInput()); });
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(StopOp op) {
  startStatement();
  ps.scopedBox(PP::ibox2, [&]() {
    ps << "stop(" << PP::ibox0;
    emitExpression(op.getClock());
    ps << "," << PP::space;
    emitExpression(op.getCond());
    ps << "," << PP::space;
    ps.addAsString(op.getExitCode());
    ps << ")" << PP::end;
    if (!op.getName().empty()) {
      ps << PP::space << ": " << PPExtString(legalize(op.getNameAttr()));
    }
  });
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(SkipOp op) {
  startStatement();
  ps << "skip";
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(PrintFOp op) {
  startStatement();
  ps.scopedBox(PP::ibox2, [&]() {
    ps << "printf(" << PP::ibox0;
    emitExpression(op.getClock());
    ps << "," << PP::space;
    emitExpression(op.getCond());
    ps << "," << PP::space;
    ps.writeQuotedEscaped(op.getFormatString());
    for (auto operand : op.getSubstitutions()) {
      ps << "," << PP::space;
      emitExpression(operand);
    }
    ps << ")" << PP::end;
    if (!op.getName().empty()) {
      ps << PP::space << ": " << PPExtString(legalize(op.getNameAttr()));
    }
  });
  emitLocationAndNewLine(op);
}

template <class T>
void Emitter::emitVerifStatement(T op, StringRef mnemonic) {
  startStatement();
  ps.scopedBox(PP::ibox2, [&]() {
    ps << mnemonic << "(" << PP::ibox0;
    emitExpression(op.getClock());
    ps << "," << PP::space;
    emitExpression(op.getPredicate());
    ps << "," << PP::space;
    emitExpression(op.getEnable());
    ps << "," << PP::space;
    ps.writeQuotedEscaped(op.getMessage());
    ps << ")" << PP::end;
    if (!op.getName().empty()) {
      ps << PP::space << ": " << PPExtString(legalize(op.getNameAttr()));
    }
  });
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(ConnectOp op) {
  startStatement();
  if (FIRVersion::compare(version, {3, 0, 0}) >= 0) {
    ps.scopedBox(PP::ibox2, [&]() {
      if (op.getSrc().getDefiningOp<InvalidValueOp>()) {
        ps << "invalidate" << PP::space;
        emitExpression(op.getDest());
      } else {
        ps << "connect" << PP::space;
        emitExpression(op.getDest());
        ps << "," << PP::space;
        emitExpression(op.getSrc());
      }
    });
  } else {
    auto emitLHS = [&]() { emitExpression(op.getDest()); };
    if (op.getSrc().getDefiningOp<InvalidValueOp>()) {
      emitAssignLike(
          emitLHS, [&]() { ps << "invalid"; }, PPExtString("is"));
    } else {
      emitAssignLike(
          emitLHS, [&]() { emitExpression(op.getSrc()); }, PPExtString("<="));
    }
  }
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(StrictConnectOp op) {
  startStatement();
  if (FIRVersion::compare(version, {3, 0, 0}) >= 0) {
    ps.scopedBox(PP::ibox2, [&]() {
      if (op.getSrc().getDefiningOp<InvalidValueOp>()) {
        ps << "invalidate" << PP::space;
        emitExpression(op.getDest());
      } else {
        ps << "connect" << PP::space;
        emitExpression(op.getDest());
        ps << "," << PP::space;
        emitExpression(op.getSrc());
      }
    });
  } else {
    auto emitLHS = [&]() { emitExpression(op.getDest()); };
    if (op.getSrc().getDefiningOp<InvalidValueOp>()) {
      emitAssignLike(
          emitLHS, [&]() { ps << "invalid"; }, PPExtString("is"));
    } else {
      emitAssignLike(
          emitLHS, [&]() { emitExpression(op.getSrc()); }, PPExtString("<="));
    }
  }
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(PropAssignOp op) {
  startStatement();
  ps.scopedBox(PP::ibox2, [&]() {
    ps << "propassign" << PP::space;
    interleaveComma(op.getOperands());
  });
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(InstanceOp op) {
  startStatement();
  auto legalName = legalize(op.getNameAttr());
  ps << "inst " << PPExtString(legalName) << " of "
     << PPExtString(legalize(op.getModuleNameAttr().getAttr()));
  emitLocationAndNewLine(op);

  // Make sure we have a name like `<inst>.<port>` for each of the instance
  // result values.
  SmallString<16> portName(legalName);
  portName.push_back('.');
  unsigned baseLen = portName.size();
  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    portName.append(legalize(op.getPortName(i)));
    addValueName(op.getResult(i), portName);
    portName.resize(baseLen);
  }
}

void Emitter::emitStatement(AttachOp op) {
  emitStatementFunctionOp(PPExtString("attach"), op);
}

void Emitter::emitStatement(MemOp op) {
  auto legalName = legalize(op.getNameAttr());
  SmallString<16> portName(legalName);
  portName.push_back('.');
  auto portNameBaseLen = portName.size();
  for (auto result : llvm::zip(op.getResults(), op.getPortNames())) {
    portName.resize(portNameBaseLen);
    portName.append(legalize(cast<StringAttr>(std::get<1>(result))));
    addValueName(std::get<0>(result), portName);
  }

  startStatement();
  ps << "mem " << PPExtString(legalName) << " :";
  emitLocationAndNewLine(op);
  ps.scopedBox(PP::bbox2, [&]() {
    startStatement();
    ps << "data-type => ";
    emitType(op.getDataType());
    ps << PP::newline;
    ps << "depth => ";
    ps.addAsString(op.getDepth());
    ps << PP::newline;
    ps << "read-latency => ";
    ps.addAsString(op.getReadLatency());
    ps << PP::newline;
    ps << "write-latency => ";
    ps.addAsString(op.getWriteLatency());
    ps << PP::newline;

    SmallString<16> reader, writer, readwriter;
    for (std::pair<StringAttr, MemOp::PortKind> port : op.getPorts()) {
      auto add = [&](SmallString<16> &to, StringAttr name) {
        if (!to.empty())
          to.push_back(' ');
        to.append(name.getValue());
      };
      switch (port.second) {
      case MemOp::PortKind::Read:
        add(reader, legalize(port.first));
        break;
      case MemOp::PortKind::Write:
        add(writer, legalize(port.first));
        break;
      case MemOp::PortKind::ReadWrite:
        add(readwriter, legalize(port.first));
        break;
      case MemOp::PortKind::Debug:
        emitOpError(op, "has unsupported 'debug' port");
        return;
      }
    }
    if (!reader.empty())
      ps << "reader => " << reader << PP::newline;
    if (!writer.empty())
      ps << "writer => " << writer << PP::newline;
    if (!readwriter.empty())
      ps << "readwriter => " << readwriter << PP::newline;

    ps << "read-under-write => ";
    emitAttribute(op.getRuw());
    setPendingNewline();
  });
}

void Emitter::emitStatement(SeqMemOp op) {
  startStatement();
  ps.scopedBox(PP::ibox2, [&]() {
    ps << "smem " << PPExtString(legalize(op.getNameAttr()));
    emitTypeWithColon(op.getType());
    ps << PP::space;
    emitAttribute(op.getRuw());
  });
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(CombMemOp op) {
  startStatement();
  ps.scopedBox(PP::ibox2, [&]() {
    ps << "cmem " << PPExtString(legalize(op.getNameAttr()));
    emitTypeWithColon(op.getType());
  });
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(MemoryPortOp op) {
  // Nothing to output for this operation.
  addValueName(op.getData(), legalize(op.getNameAttr()));
}

void Emitter::emitStatement(MemoryDebugPortOp op) {
  // Nothing to output for this operation.
  addValueName(op.getData(), legalize(op.getNameAttr()));
}

void Emitter::emitStatement(MemoryPortAccessOp op) {
  startStatement();

  // Print the port direction and name.
  auto port = cast<MemoryPortOp>(op.getPort().getDefiningOp());
  emitAttribute(port.getDirection());
  // TODO: emitAssignLike
  ps << " mport " << PPExtString(legalize(port.getNameAttr())) << " = ";

  // Print the memory name.
  auto *mem = port.getMemory().getDefiningOp();
  if (auto seqMem = dyn_cast<SeqMemOp>(mem))
    ps << legalize(seqMem.getNameAttr());
  else
    ps << legalize(cast<CombMemOp>(mem).getNameAttr());

  // Print the address.
  ps << "[";
  emitExpression(op.getIndex());
  ps << "], ";

  // Print the clock.
  emitExpression(op.getClock());

  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(RefDefineOp op) {
  startStatement();
  emitAssignLike(
      [&]() {
        ps << "define ";
        emitExpression(op.getDest());
      },
      [&]() {
        auto src = op.getSrc();
        if (auto forceable = src.getDefiningOp<Forceable>();
            forceable && forceable.isForceable() &&
            forceable.getDataRef() == src) {
          ps << "rwprobe(";
          emitExpression(forceable.getData());
          ps << ")";
        } else
          emitExpression(src);
      });
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(RefForceOp op) {
  emitStatementFunctionOp(PPExtString("force"), op);
}

void Emitter::emitStatement(RefForceInitialOp op) {
  startStatement();
  auto constantPredicate =
      dyn_cast_or_null<ConstantOp>(op.getPredicate().getDefiningOp());
  bool hasEnable = !constantPredicate || constantPredicate.getValue() == 0;
  if (hasEnable) {
    ps << "when ";
    emitExpression(op.getPredicate());
    ps << ":" << PP::bbox2 << PP::neverbreak << PP::newline;
  }
  ps << "force_initial(";
  ps.scopedBox(PP::ibox0, [&]() {
    interleaveComma({op.getDest(), op.getSrc()});
    ps << ")";
  });
  if (hasEnable)
    ps << PP::end;
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(RefReleaseOp op) {
  emitStatementFunctionOp(PPExtString("release"), op);
}

void Emitter::emitStatement(RefReleaseInitialOp op) {
  startStatement();
  auto constantPredicate =
      dyn_cast_or_null<ConstantOp>(op.getPredicate().getDefiningOp());
  bool hasEnable = !constantPredicate || constantPredicate.getValue() == 0;
  if (hasEnable) {
    ps << "when ";
    emitExpression(op.getPredicate());
    ps << ":" << PP::bbox2 << PP::neverbreak << PP::newline;
  }
  ps << "release_initial(";
  emitExpression(op.getDest());
  ps << ")";
  if (hasEnable)
    ps << PP::end;
  emitLocationAndNewLine(op);
}

void Emitter::emitStatement(GroupOp op) {
  startStatement();
  ps << "group " << op.getGroupName().getLeafReference() << " :";
  emitLocationAndNewLine(op);
  auto *body = op.getBody();
  ps.scopedBox(PP::bbox2, [&]() { emitStatementsInBlock(*body); });
}

void Emitter::emitStatement(InvalidValueOp op) {
  // Only emit this invalid value if it is used somewhere else than the RHS of
  // a connect.
  if (llvm::all_of(op->getUses(), [&](OpOperand &use) {
        return use.getOperandNumber() == 1 &&
               isa<ConnectOp, StrictConnectOp>(use.getOwner());
      }))
    return;

  // TODO: emitAssignLike ?
  startStatement();
  auto name = circuitNamespace.newName("_invalid");
  addValueName(op, name);
  ps << "wire " << PPExtString(name) << " : ";
  emitType(op.getType());
  emitLocationAndNewLine(op);
  startStatement();
  if (FIRVersion::compare(version, {3, 0, 0}) >= 0)
    ps << "invalidate " << PPExtString(name);
  else
    ps << PPExtString(name) << " is invalid";
  emitLocationAndNewLine(op);
}

void Emitter::emitExpression(Value value) {
  // Handle the trivial case where we already have a name for this value which
  // we can use.
  if (auto name = lookupEmittedName(value)) {
    // Don't use PPExtString here, can't trust valueNames storage, cleared.
    ps << *name;
    return;
  }

  auto op = value.getDefiningOp();
  assert(op && "value must either be a block arg or the result of an op");
  TypeSwitch<Operation *>(op)
      .Case<
          // Basic expressions
          ConstantOp, SpecialConstantOp, SubfieldOp, SubindexOp, SubaccessOp,
          OpenSubfieldOp, OpenSubindexOp,
          // Binary
          AddPrimOp, SubPrimOp, MulPrimOp, DivPrimOp, RemPrimOp, AndPrimOp,
          OrPrimOp, XorPrimOp, LEQPrimOp, LTPrimOp, GEQPrimOp, GTPrimOp,
          EQPrimOp, NEQPrimOp, CatPrimOp, DShlPrimOp, DShlwPrimOp, DShrPrimOp,
          // Unary
          AsSIntPrimOp, AsUIntPrimOp, AsAsyncResetPrimOp, AsClockPrimOp,
          CvtPrimOp, NegPrimOp, NotPrimOp, AndRPrimOp, OrRPrimOp, XorRPrimOp,
          // Miscellaneous
          BitsPrimOp, HeadPrimOp, TailPrimOp, PadPrimOp, MuxPrimOp, ShlPrimOp,
          ShrPrimOp, UninferredResetCastOp, ConstCastOp, StringConstantOp,
          FIntegerConstantOp,
          // Reference expressions
          RefSendOp, RefResolveOp, RefSubOp>([&](auto op) {
        ps.scopedBox(PP::ibox0, [&]() { emitExpression(op); });
      })
      .Default([&](auto op) {
        emitOpError(op, "not supported as expression");
        ps << "<unsupported-expr-" << PPExtString(op->getName().stripDialect())
           << ">";
      });
}

void Emitter::emitExpression(ConstantOp op) {
  // Don't include 'const' on the type in a literal expression
  emitType(op.getType(), false);
  // TODO: Add option to control base-2/8/10/16 output here.
  ps << "(";
  ps.addAsString(op.getValue());
  ps << ")";
}

void Emitter::emitExpression(SpecialConstantOp op) {
  auto emitInner = [&]() {
    ps << "UInt<1>(";
    ps.addAsString(op.getValue());
    ps << ")";
  };
  // TODO: Emit type decl for type alias.
  FIRRTLTypeSwitch<FIRRTLType>(type_cast<FIRRTLType>(op.getType()))
      .Case<ClockType>([&](auto type) {
        ps << "asClock(";
        emitInner();
        ps << ")";
      })
      .Case<ResetType>([&](auto type) { emitInner(); })
      .Case<AsyncResetType>([&](auto type) {
        ps << "asAsyncReset(";
        emitInner();
        ps << ")";
      });
}

// NOLINTNEXTLINE(misc-no-recursion)
void Emitter::emitExpression(SubfieldOp op) {
  BundleType type = op.getInput().getType();
  emitExpression(op.getInput());
  ps << "." << legalize(type.getElementNameAttr(op.getFieldIndex()));
}

// NOLINTNEXTLINE(misc-no-recursion)
void Emitter::emitExpression(SubindexOp op) {
  emitExpression(op.getInput());
  ps << "[";
  ps.addAsString(op.getIndex());
  ps << "]";
}

// NOLINTNEXTLINE(misc-no-recursion)
void Emitter::emitExpression(SubaccessOp op) {
  emitExpression(op.getInput());
  ps << "[";
  emitExpression(op.getIndex());
  ps << "]";
}

void Emitter::emitExpression(OpenSubfieldOp op) {
  auto type = op.getInput().getType();
  emitExpression(op.getInput());
  ps << "." << legalize(type.getElementNameAttr(op.getFieldIndex()));
}

void Emitter::emitExpression(OpenSubindexOp op) {
  emitExpression(op.getInput());
  ps << "[";
  ps.addAsString(op.getIndex());
  ps << "]";
}

void Emitter::emitExpression(RefSendOp op) {
  ps << "probe(";
  emitExpression(op.getBase());
  ps << ")";
}

void Emitter::emitExpression(RefResolveOp op) {
  ps << "read(";
  emitExpression(op.getRef());
  ps << ")";
}

void Emitter::emitExpression(RefSubOp op) {
  emitExpression(op.getInput());
  FIRRTLTypeSwitch<FIRRTLBaseType, void>(op.getInput().getType().getType())
      .Case<FVectorType>([&](auto type) {
        ps << "[";
        ps.addAsString(op.getIndex());
        ps << "]";
      })
      .Case<BundleType>(
          [&](auto type) { ps << "." << type.getElementName(op.getIndex()); });
}

void Emitter::emitExpression(UninferredResetCastOp op) {
  emitExpression(op.getInput());
}

void Emitter::emitExpression(FIntegerConstantOp op) {
  ps << "Integer(";
  ps.addAsString(op.getValue());
  ps << ")";
}

void Emitter::emitExpression(StringConstantOp op) {
  ps << "String(";
  ps.writeQuotedEscaped(op.getValue());
  ps << ")";
}

void Emitter::emitExpression(ConstCastOp op) { emitExpression(op.getInput()); }

void Emitter::emitPrimExpr(StringRef mnemonic, Operation *op,
                           ArrayRef<uint32_t> attrs) {
  ps << mnemonic << "(" << PP::ibox0;
  interleaveComma(op->getOperands());
  if (!op->getOperands().empty() && !attrs.empty())
    ps << "," << PP::space;
  interleaveComma(attrs, [&](auto attr) { ps.addAsString(attr); });
  ps << ")" << PP::end;
}

void Emitter::emitAttribute(MemDirAttr attr) {
  switch (attr) {
  case MemDirAttr::Infer:
    ps << "infer";
    break;
  case MemDirAttr::Read:
    ps << "read";
    break;
  case MemDirAttr::Write:
    ps << "write";
    break;
  case MemDirAttr::ReadWrite:
    ps << "rdwr";
    break;
  }
}

void Emitter::emitAttribute(RUWAttr attr) {
  switch (attr) {
  case RUWAttr::Undefined:
    ps << "undefined";
    break;
  case RUWAttr::Old:
    ps << "old";
    break;
  case RUWAttr::New:
    ps << "new";
    break;
  }
}

/// Emit a FIRRTL type into the output.
void Emitter::emitType(Type type, bool includeConst) {
  if (includeConst && isConst(type))
    ps << "const ";
  auto emitWidth = [&](std::optional<int32_t> width) {
    if (width) {
      ps << "<";
      ps.addAsString(*width);
      ps << ">";
    }
  };
  // TODO: Emit type decl for type alias.
  FIRRTLTypeSwitch<Type>(type)
      .Case<ClockType>([&](auto) { ps << "Clock"; })
      .Case<ResetType>([&](auto) { ps << "Reset"; })
      .Case<AsyncResetType>([&](auto) { ps << "AsyncReset"; })
      .Case<UIntType>([&](auto type) {
        ps << "UInt";
        emitWidth(type.getWidth());
      })
      .Case<SIntType>([&](auto type) {
        ps << "SInt";
        emitWidth(type.getWidth());
      })
      .Case<AnalogType>([&](auto type) {
        ps << "Analog";
        emitWidth(type.getWidth());
      })
      .Case<OpenBundleType, BundleType>([&](auto type) {
        ps << "{";
        if (!type.getElements().empty())
          ps << PP::nbsp;
        bool anyEmitted = false;
        ps.scopedBox(PP::cbox0, [&]() {
          for (auto &element : type.getElements()) {
            if (anyEmitted)
              ps << "," << PP::space;
            ps.scopedBox(PP::ibox2, [&]() {
              if (element.isFlip)
                ps << "flip ";
              ps << legalize(element.name);
              emitTypeWithColon(element.type);
              anyEmitted = true;
            });
          }
          if (anyEmitted)
            ps << PP::nbsp;
          ps << "}";
        });
      })
      .Case<OpenVectorType, FVectorType, CMemoryType>([&](auto type) {
        emitType(type.getElementType());
        ps << "[";
        ps.addAsString(type.getNumElements());
        ps << "]";
      })
      .Case<RefType>([&](RefType type) {
        if (type.getForceable())
          ps << "RW";
        ps << "Probe<";
        emitType(type.getType());
        ps << ">";
      })
      .Case<StringType>([&](StringType type) { ps << "String"; })
      .Case<FIntegerType>([&](FIntegerType type) { ps << "Integer"; })
      .Case<PathType>([&](PathType type) { ps << "Path"; })
      .Default([&](auto type) {
        llvm_unreachable("all types should be implemented");
      });
}

/// Emit a location as `@[<filename> <line>:<column>]` annotation, including a
/// leading space.
void Emitter::emitLocation(Location loc) {
  // TODO: Handle FusedLoc and uniquify locations, avoid repeated file names.
  ps << PP::neverbreak;
  if (auto fileLoc = loc->dyn_cast_or_null<FileLineColLoc>()) {
    ps << " @[" << fileLoc.getFilename().getValue();
    if (auto line = fileLoc.getLine()) {
      ps << " ";
      ps.addAsString(line);
      if (auto col = fileLoc.getColumn()) {
        ps << ":";
        ps.addAsString(col);
      }
    }
    ps << "]";
  }
}
// NOLINTEND(misc-no-recursion)

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Emit the specified FIRRTL circuit into the given output stream.
mlir::LogicalResult
circt::firrtl::exportFIRFile(mlir::ModuleOp module, llvm::raw_ostream &os,
                             std::optional<size_t> targetLineLength,
                             FIRVersion version) {
  Emitter emitter(os, version,
                  targetLineLength.value_or(defaultTargetLineLength));
  for (auto &op : *module.getBody()) {
    if (auto circuitOp = dyn_cast<CircuitOp>(op))
      emitter.emitCircuit(circuitOp);
  }
  return emitter.finalize();
}

void circt::firrtl::registerToFIRFileTranslation() {
  static llvm::cl::opt<size_t> targetLineLength(
      "target-line-length",
      llvm::cl::desc("Target line length for emitted .fir"),
      llvm::cl::value_desc("number of chars"),
      llvm::cl::init(defaultTargetLineLength));
  static mlir::TranslateFromMLIRRegistration toFIR(
      "export-firrtl", "emit FIRRTL dialect operations to .fir output",
      [](ModuleOp module, llvm::raw_ostream &os) {
        return exportFIRFile(module, os, targetLineLength, {3, 1, 0});
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<chirrtl::CHIRRTLDialect>();
        registry.insert<firrtl::FIRRTLDialect>();
      });
}
