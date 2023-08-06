//===- SystemCEmissionPatterns.cpp - SystemC Dialect Emission Patterns ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission patterns for the systemc dialect.
//
//===----------------------------------------------------------------------===//

#include "SystemCEmissionPatterns.h"
#include "../EmissionPrinter.h"
#include "circt/Dialect/SystemC/SystemCOps.h"

using namespace circt;
using namespace circt::systemc;
using namespace circt::ExportSystemC;

//===----------------------------------------------------------------------===//
// Operation emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit a SystemC module using the SC_MODULE macro and emit all ports as fields
/// of the module. Users of the ports request an expression to be inlined and we
/// simply return the name of the port.
struct SCModuleEmitter : OpEmissionPattern<SCModuleOp> {
  using OpEmissionPattern::OpEmissionPattern;
  MatchResult matchInlinable(Value value) override {
    if (value.isa<BlockArgument>() &&
        value.getParentRegion()->getParentOfType<SCModuleOp>())
      return Precedence::VAR;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto module = value.getParentRegion()->getParentOfType<SCModuleOp>();
    for (size_t i = 0, e = module.getNumArguments(); i < e; ++i) {
      if (module.getArgument(i) == value) {
        p << module.getPortNames()[i].cast<StringAttr>().getValue();
        return;
      }
    }
  }

  void emitStatement(SCModuleOp module, EmissionPrinter &p) override {
    // Emit a newline at the start to ensure an empty line before the module for
    // better readability.
    p << "\nSC_MODULE(" << module.getModuleName() << ") ";
    auto scope = p.getOstream().scope("{\n", "};\n");
    for (size_t i = 0, e = module.getNumArguments(); i < e; ++i) {
      p.emitType(module.getArgument(i).getType());

      auto portName = module.getPortNames()[i].cast<StringAttr>().getValue();
      p << " " << portName << ";\n";
    }

    p.emitRegion(module.getRegion(), scope);
  }
};

/// Emit a systemc.signal.write operation by using the explicit 'write' member
/// function of the signal and port classes.
struct SignalWriteEmitter : OpEmissionPattern<SignalWriteOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(SignalWriteOp op, EmissionPrinter &p) override {
    p.getInlinable(op.getDest()).emit();
    p << ".write(";
    p.getInlinable(op.getSrc()).emit();
    p << ");\n";
  }
};

/// Emit a systemc.signal.read operation by using the explicit 'read' member
/// function of the signal and port classes.
struct SignalReadEmitter : OpEmissionPattern<SignalReadOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<SignalReadOp>())
      return Precedence::FUNCTION_CALL;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p.getInlinable(value.getDefiningOp<SignalReadOp>().getInput()).emit();
    p << ".read()";
  }
};

/// Emit a systemc.cpp.new operation.
struct NewEmitter : OpEmissionPattern<NewOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<NewOp>())
      return Precedence::NEW;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p << "new ";
    p.emitType(value.getType().cast<mlir::emitc::PointerType>().getPointee());

    auto newOp = value.getDefiningOp<NewOp>();
    if (newOp.getArgs().empty())
      return;

    p << "(";
    llvm::interleaveComma(newOp.getArgs(), p, [&](Value arg) {
      p.getInlinable(arg).emitWithParensOnLowerPrecedence(Precedence::COMMA);
    });
    p << ")";
  }
};

/// Emit a systemc.ctor operation by using the SC_CTOR macro.
struct CtorEmitter : OpEmissionPattern<CtorOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(CtorOp op, EmissionPrinter &p) override {
    // Emit a new line before the SC_CTOR to ensure an empty line for better
    // readability.
    p << "\nSC_CTOR(" << op->getParentOfType<SCModuleOp>().getModuleName()
      << ") ";
    p.emitRegion(op.getBody());
  }
};

/// Emit a systemc.cpp.destructor operation.
struct DestructorEmitter : OpEmissionPattern<DestructorOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(DestructorOp op, EmissionPrinter &p) override {
    // Emit a new line before the destructor to ensure an empty line for better
    // readability.
    // TODO: the 'override' keyword is hardcoded here because the destructor can
    // only be inside a class inheriting from 'sc_module', if we ever support
    // custom classes the override should probably be a unitAttr on the
    // destructor operation.
    p << "\n~" << op->getParentOfType<SCModuleOp>().getModuleName()
      << "() override ";
    p.emitRegion(op.getBody());
  }
};

/// Emit a systemc.func operation.
struct SCFuncEmitter : OpEmissionPattern<SCFuncOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<SCFuncOp>())
      return Precedence::VAR;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p << value.getDefiningOp<SCFuncOp>().getName();
  }

  void emitStatement(SCFuncOp op, EmissionPrinter &p) override {
    // Emit a new line before the member function to ensure an empty line for
    // better readability.
    p << "\nvoid " << op.getName() << "() ";
    p.emitRegion(op.getBody());
  }
};

/// Emit a systemc.method operation by using the SC_METHOD macro.
struct MethodEmitter : OpEmissionPattern<MethodOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(MethodOp op, EmissionPrinter &p) override {
    p << "SC_METHOD(";
    p.getInlinable(op.getFuncHandle()).emit();
    p << ");\n";
  }
};

/// Emit a systemc.sensitive operation by using the 'sensitive' data member;
struct SensitiveEmitter : OpEmissionPattern<SensitiveOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(SensitiveOp op, EmissionPrinter &p) override {
    if (op.getSensitivities().empty())
      return;

    p << "sensitive << ";
    llvm::interleave(
        op.getSensitivities(), p,
        [&](Value sensitive) {
          p.getInlinable(sensitive).emitWithParensOnLowerPrecedence(
              Precedence::SHL);
        },
        " << ");
    p << ";\n";
  }
};

/// Emit a systemc.thread operation by using the SC_THREAD macro.
struct ThreadEmitter : OpEmissionPattern<ThreadOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(ThreadOp op, EmissionPrinter &p) override {
    p << "SC_THREAD(";
    p.getInlinable(op.getFuncHandle()).emit();
    p << ");\n";
  }
};

/// Emit a systemc.cpp.delete operation.
struct DeleteEmitter : OpEmissionPattern<DeleteOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(DeleteOp op, EmissionPrinter &p) override {
    p << "delete ";
    p.getInlinable(op.getPointer())
        .emitWithParensOnLowerPrecedence(Precedence::DELETE);
    p << ";\n";
  }
};

/// Emit a systemc.signal operation.
struct SignalEmitter : OpEmissionPattern<SignalOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (llvm::isa_and_nonnull<SignalOp>(value.getDefiningOp()))
      return Precedence::VAR;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p << value.getDefiningOp<SignalOp>().getName();
  }

  void emitStatement(SignalOp op, EmissionPrinter &p) override {
    p.emitType(op.getSignal().getType());
    p << " ";

    if (op.getNamed()) {
      // This style of emitting SC_NAMED requires the printed code to be
      // compiled with at least C++11.
      p << "SC_NAMED(" << op.getName() << ");\n";
      return;
    }

    p << op.getName() << ";\n";
  }
};

/// Emit a systemc.instance.decl operation.
struct InstanceDeclEmitter : OpEmissionPattern<InstanceDeclOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<InstanceDeclOp>())
      return Precedence::VAR;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p << value.getDefiningOp<InstanceDeclOp>().getName();
  }

  void emitStatement(InstanceDeclOp op, EmissionPrinter &p) override {
    p.emitType(op.getInstanceType());
    p << " " << op.getName() << ";\n";
  }
};
} // namespace

/// Emit a systemc.instance.bind_port operation using the operator() rather than
/// .bind() variant.
struct BindPortEmitter : OpEmissionPattern<BindPortOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(BindPortOp op, EmissionPrinter &p) override {
    p.getInlinable(op.getInstance())
        .emitWithParensOnLowerPrecedence(Precedence::MEMBER_ACCESS);

    p << "." << op.getPortName() << "(";
    p.getInlinable(op.getChannel()).emit();
    p << ");\n";
  }
};

/// Emit a systemc.cpp.member_access operation.
struct MemberAccessEmitter : OpEmissionPattern<MemberAccessOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<MemberAccessOp>())
      return Precedence::MEMBER_ACCESS;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto op = value.getDefiningOp<MemberAccessOp>();
    p.getInlinable(op.getObject())
        .emitWithParensOnLowerPrecedence(Precedence::MEMBER_ACCESS);

    if (op.getAccessKind() == MemberAccessKind::Arrow)
      p << "->";
    else if (op.getAccessKind() == MemberAccessKind::Dot)
      p << ".";
    else
      p.emitError(op, "member access kind not implemented");

    p << op.getMemberName();
  }
};

/// Emit a systemc.cpp.assign operation.
struct AssignEmitter : OpEmissionPattern<AssignOp> {
  using OpEmissionPattern::OpEmissionPattern;

  void emitStatement(AssignOp op, EmissionPrinter &p) override {
    p.getInlinable(op.getDest())
        .emitWithParensOnLowerPrecedence(Precedence::ASSIGN);
    p << " = ";
    p.getInlinable(op.getSource())
        .emitWithParensOnLowerPrecedence(Precedence::ASSIGN);
    p << ";\n";
  }
};

/// Emit a systemc.cpp.variable operation.
struct VariableEmitter : OpEmissionPattern<VariableOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<VariableOp>())
      return Precedence::VAR;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    p << value.getDefiningOp<VariableOp>().getName();
  }

  void emitStatement(VariableOp op, EmissionPrinter &p) override {
    p.emitType(op.getVariable().getType());
    p << " " << op.getName();

    if (op.getInit()) {
      p << " = ";
      p.getInlinable(op.getInit())
          .emitWithParensOnLowerPrecedence(Precedence::ASSIGN);
    }

    p << ";\n";
  }
};

/// Emit a systemc.cpp.func function. Users of the function arguments request an
/// expression to be inlined and we simply return the name of the argument. This
/// name has to be passed to this emission pattern via an array of strings
/// attribute called 'argNames' because the emitter cannot do any name uniquing
/// as it just emits the IR statement by statement. However, relying on an
/// attribute for the argument names also has the advantage that the names
/// can be preserved 1-1 during a lowering pipeline and upstream passes have
/// more control on how the arguments should be named (e.g. when they create a
/// function and have some context to assign better names).
struct FuncEmitter : OpEmissionPattern<FuncOp> {
  using OpEmissionPattern::OpEmissionPattern;
  MatchResult matchInlinable(Value value) override {
    // Note that the verifier of systemc::FuncOp guarantees that whenever the
    // function has a body, argNames is present and of the correct length
    if (value.isa<BlockArgument>() &&
        value.getParentRegion()->getParentOfType<FuncOp>())
      return Precedence::VAR;

    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    auto func = value.getParentRegion()->getParentOfType<FuncOp>();
    for (auto [arg, name] :
         llvm::zip(func.getArguments(), func.getArgNames())) {
      if (arg == value) {
        p << name.cast<StringAttr>().getValue();
        return;
      }
    }
  }

  void emitStatement(FuncOp func, EmissionPrinter &p) override {
    // Emit a newline at the start to ensure an empty line before the function
    // for better readability.
    p << "\n";

    if (func.getExternC())
      p << "extern \"C\" ";

    // Emit return type.
    if (func.getFunctionType().getNumResults() == 0)
      p << "void";
    else
      p.emitType(func.getFunctionType().getResult(0));

    p << " " << func.getSymName() << "(";

    // Emit the argument list. When the function is a declaration, it is not
    // required to have argument names, in that case just print the types.
    if (func.isDeclaration() && func.getArgNames().empty())
      llvm::interleaveComma(func.getFunctionType().getInputs(), p,
                            [&](Type ty) { p.emitType(ty); });
    else
      llvm::interleaveComma(
          llvm::zip(func.getFunctionType().getInputs(), func.getArgNames()), p,
          [&](std::tuple<Type, Attribute> arg) {
            p.emitType(std::get<0>(arg));
            p << " " << std::get<1>(arg).cast<StringAttr>().getValue();
          });

    p << ")";

    // Emit body when present.
    if (func.isDeclaration()) {
      p << ";\n";
    } else {
      p << " ";
      p.emitRegion(func.getRegion());
    }
  }
};

/// Emit a systemc.cpp.call operation. If it has no result, it is treated as a
/// statement, otherwise as an expression that will always be inlined. That
/// means, an emission preparation pass has to insert a VariableOp to bind the
/// call result to such that reordering of the call cannot lead to incorrectness
/// due to interference of side-effects.
class CallEmitter : public OpEmissionPattern<CallOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<CallOp>())
      return Precedence::FUNCTION_CALL;
    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    printCall(value.getDefiningOp<CallOp>(), p);
  }

  void emitStatement(CallOp op, EmissionPrinter &p) override {
    // If the call returns values, then it is treated as an expression rather
    // than a statement.
    if (op.getNumResults() > 0)
      return;

    printCall(op, p);
    p << ";\n";
  }

private:
  void printCall(CallOp op, EmissionPrinter &p) {
    p << op.getCallee() << "(";
    llvm::interleaveComma(op.getOperands(), p, [&](auto arg) {
      p.getInlinable(arg).emitWithParensOnLowerPrecedence(Precedence::COMMA);
    });
    p << ")";
  }
};

/// Emit a systemc.cpp.call_indirect operation. If it has no result, it is
/// treated as a statement, otherwise as an expression that will always be
/// inlined. That means, an emission preparation pass has to insert a VariableOp
/// to bind the call result to such that reordering of the call cannot lead to
/// incorrectness due to interference of side-effects.
class CallIndirectEmitter : public OpEmissionPattern<CallIndirectOp> {
  using OpEmissionPattern::OpEmissionPattern;

  MatchResult matchInlinable(Value value) override {
    if (value.getDefiningOp<CallIndirectOp>())
      return Precedence::FUNCTION_CALL;

    return {};
  }

  void emitInlined(Value value, EmissionPrinter &p) override {
    printCall(value.getDefiningOp<CallIndirectOp>(), p);
  }

  void emitStatement(CallIndirectOp op, EmissionPrinter &p) override {
    // If the call returns values, then it is treated as an expression rather
    // than a statement.
    if (op.getNumResults() > 0)
      return;

    printCall(op, p);
    p << ";\n";
  }

private:
  void printCall(CallIndirectOp op, EmissionPrinter &p) {
    p.getInlinable(op.getCallee())
        .emitWithParensOnLowerPrecedence(Precedence::FUNCTION_CALL);
    p << "(";
    llvm::interleaveComma(op.getCalleeOperands(), p, [&](auto arg) {
      p.getInlinable(arg).emitWithParensOnLowerPrecedence(Precedence::COMMA);
    });
    p << ")";
  }
};

/// Emit a systemc.cpp.return operation.
struct ReturnEmitter : OpEmissionPattern<ReturnOp> {
  using OpEmissionPattern::OpEmissionPattern;

  bool matchStatement(Operation *op) override {
    return isa<ReturnOp>(op) && cast<ReturnOp>(op)->getNumOperands() <= 1;
  }

  void emitStatement(ReturnOp op, EmissionPrinter &p) override {
    p << "return";
    if (!op.getReturnValues().empty()) {
      p << " ";
      p.getInlinable(op.getReturnValues()[0]).emit();
    }
    p << ";\n";
  }
};

//===----------------------------------------------------------------------===//
// Type emission patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Emit SystemC signal and port types according to the specification listed in
/// their ODS description.
template <typename Ty, const char Mn[]>
struct SignalTypeEmitter : public TypeEmissionPattern<Ty> {
  void emitType(Ty type, EmissionPrinter &p) override {
    p << Mn << "<";
    p.emitType(type.getBaseType());
    p << ">";
  }
};

/// Emit a systemc::ModuleType by just printing the module name as we are
/// dealing with a nominal type system.
struct ModuleTypeEmitter : public TypeEmissionPattern<ModuleType> {
  void emitType(ModuleType type, EmissionPrinter &p) override {
    p << type.getModuleName().getValue();
  }
};

/// Emit SystemC integer and bit-vector types with known-at-compile-time
/// bit-width according to the specification listed in their class description.
template <typename Ty>
struct IntegerTypeEmitter : public TypeEmissionPattern<Ty> {
  void emitType(Ty type, EmissionPrinter &p) override {
    p << "sc_" << Ty::getMnemonic() << "<" << type.getWidth() << ">";
  }
};

/// Emit SystemC integer and bit-vector types without known bit-width according
/// to the specification listed in their class description.
template <typename Ty>
struct DynIntegerTypeEmitter : public TypeEmissionPattern<Ty> {
  void emitType(Ty type, EmissionPrinter &p) override {
    p << "sc_" << Ty::getMnemonic();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Register Operation and Type emission patterns.
//===----------------------------------------------------------------------===//

void circt::ExportSystemC::populateSystemCOpEmitters(
    OpEmissionPatternSet &patterns, MLIRContext *context) {
  patterns.add<
      SCModuleEmitter, CtorEmitter, SCFuncEmitter, MethodEmitter, ThreadEmitter,
      // Signal and port related emitters
      SignalWriteEmitter, SignalReadEmitter, SignalEmitter, SensitiveEmitter,
      // Instance-related emitters
      InstanceDeclEmitter, BindPortEmitter,
      // CPP-level operation emitters
      AssignEmitter, VariableEmitter, NewEmitter, DestructorEmitter,
      DeleteEmitter, MemberAccessEmitter,
      // Function related emitters
      FuncEmitter, ReturnEmitter, CallEmitter, CallIndirectEmitter>(context);
}

void circt::ExportSystemC::populateSystemCTypeEmitters(
    TypeEmissionPatternSet &patterns) {
  static constexpr const char in[] = "sc_in";
  static constexpr const char inout[] = "sc_inout";
  static constexpr const char out[] = "sc_out";
  static constexpr const char signal[] = "sc_signal";

  patterns.add<
      // Port and signal types
      SignalTypeEmitter<InputType, in>, SignalTypeEmitter<InOutType, inout>,
      SignalTypeEmitter<OutputType, out>, SignalTypeEmitter<SignalType, signal>,
      // SystemC integers with statically known bit-width
      IntegerTypeEmitter<IntType>, IntegerTypeEmitter<UIntType>,
      IntegerTypeEmitter<BigIntType>, IntegerTypeEmitter<BigUIntType>,
      // SystemC integers without statically known bit-width
      DynIntegerTypeEmitter<IntBaseType>, DynIntegerTypeEmitter<UIntBaseType>,
      DynIntegerTypeEmitter<SignedType>, DynIntegerTypeEmitter<UnsignedType>,
      // Vector types with statically known bit-width
      IntegerTypeEmitter<BitVectorType>, IntegerTypeEmitter<LogicVectorType>,
      // Vector types without statically known bit-width
      DynIntegerTypeEmitter<BitVectorBaseType>,
      DynIntegerTypeEmitter<LogicVectorBaseType>,
      // Misc types
      DynIntegerTypeEmitter<LogicType>, ModuleTypeEmitter>();
}
