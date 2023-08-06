//===- SystemCTypes.td - SystemC type definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the SystemC Types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES

include "circt/Dialect/SystemC/SystemCDialect.td"
include "mlir/IR/EnumAttr.td"

class SystemCType<string name> : TypeDef<SystemCDialect, name> { }

class SystemCTypeConstraint<string condition, string descr, string cppClassName>
  : DialectType<SystemCDialect, CPred<condition>, descr, cppClassName> { }

//===----------------------------------------------------------------------===//
// Function constraints
//===----------------------------------------------------------------------===//

// A constraint for a mlir::FunctionType with zero inputs.
def NullaryFunctionConstraint : CPred<
  "::circt::hw::type_isa<mlir::FunctionType>($_self) && " #
  "::circt::hw::type_cast<mlir::FunctionType>($_self).getNumInputs() == 0">;

// A constraint for a mlir::FunctionType with zero results.
def VoidFunctionConstraint : CPred<
  "::circt::hw::type_isa<mlir::FunctionType>($_self) && " #
  "::circt::hw::type_cast<mlir::FunctionType>($_self).getNumResults() == 0">;

// A handle to refer to a mlir::FunctionType with zero inputs and results,
// i.e., a nullary void function.
def NullaryVoidFunctionType : DialectType<SystemCDialect,
    And<[NullaryFunctionConstraint, VoidFunctionConstraint]>,
    "FunctionType with no inputs and results",
    "::circt::hw::TypeAliasOr<mlir::FunctionType>">,
  BuildableType<"mlir::FunctionType::get($_builder.getContext(), {}, {})">;

//===----------------------------------------------------------------------===//
// Logic type
//===----------------------------------------------------------------------===//

// Represents the SystemC sc_logic type. Declares the systemc::LogicType in C++.
def LogicType : SystemCType<"Logic"> {
  let summary = "a SystemC sc_logic type";
  let description = [{
    Represents a single bit with with a value corresponding to one of the four
    logic states '0', '1', 'Z', and 'X' in SystemC as described in
    IEEE 1666-2011 §7.9.2.
    A value of this type can be created using the 'emitc.constant' operation
    with a string attribute containing "0", "1", "X", "Z", "x", or "z" or an
    i1 attribute representing "true" or "false". Any other value will be
    interpreted as 'X' (unknown state).
  }];
  let mnemonic = "logic";
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// Integer types
//===----------------------------------------------------------------------===//

def ValueBaseType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::ValueBaseType>($_self)",
  "a SystemC sc_value_base type",
  "::circt::hw::TypeAliasOr<circt::systemc::ValueBaseType>">;

def UIntBaseType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::UIntBaseType>($_self)",
  "a SystemC sc_uint_base type",
  "::circt::hw::TypeAliasOr<circt::systemc::UIntBaseType>">;

def UIntType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::UIntType>($_self)",
  "a SystemC sc_uint<W> type",
  "::circt::hw::TypeAliasOr<circt::systemc::UIntType>">;

def IntBaseType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::IntBaseType>($_self)",
  "a SystemC sc_int_base type",
  "::circt::hw::TypeAliasOr<circt::systemc::IntBaseType>">;

def IntType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::IntType>($_self)",
  "a SystemC sc_int<W> type",
  "::circt::hw::TypeAliasOr<circt::systemc::IntType>">;

def SignedType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::SignedType>($_self)",
  "a SystemC sc_signed type",
  "::circt::hw::TypeAliasOr<circt::systemc::SignedType>">;

def BigIntType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::BigIntType>($_self)",
  "a SystemC sc_bigint<W> type",
  "::circt::hw::TypeAliasOr<circt::systemc::BigIntType>">;

def UnsignedType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::UnsignedType>($_self)",
  "a SystemC sc_unsigned type",
  "::circt::hw::TypeAliasOr<circt::systemc::UnsignedType>">;

def BigUIntType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::BigUIntType>($_self)",
  "a SystemC sc_biguint<W> type",
  "::circt::hw::TypeAliasOr<circt::systemc::BigUIntType>">;

def AnySystemCInteger : AnyTypeOf<[AnyInteger, ValueBaseType]>;

//===----------------------------------------------------------------------===//
// Bit-vector types
//===----------------------------------------------------------------------===//

def BitVectorBaseType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::BitVectorBaseType>($_self)",
  "a SystemC sc_bv_base type",
  "::circt::hw::TypeAliasOr<circt::systemc::BitVectorBaseType>">;

def BitVectorType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::BitVectorType>($_self)",
  "a SystemC sc_bv<W> type",
  "::circt::hw::TypeAliasOr<circt::systemc::BitVectorType>">;

def LogicVectorBaseType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::LogicVectorBaseType>($_self)",
  "a SystemC sc_lv_base type",
  "::circt::hw::TypeAliasOr<circt::systemc::LogicVectorBaseType>">;

def LogicVectorType : SystemCTypeConstraint<
  "::circt::hw::type_isa<circt::systemc::LogicVectorType>($_self)",
  "a SystemC sc_lv<W> type",
  "::circt::hw::TypeAliasOr<circt::systemc::LogicVectorType>">;

def AnySystemCVector : AnyTypeOf<[AnySignlessInteger,
                                  BitVectorBaseType,
                                  BitVectorType,
                                  LogicVectorBaseType,
                                  LogicVectorType]>;

//===----------------------------------------------------------------------===//
// Signal and port types
//===----------------------------------------------------------------------===//

class SignalTypesBase<string name> : SystemCType<name> {
  let parameters = (ins "::mlir::Type":$baseType);
  let assemblyFormat = "`<` $baseType `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$baseType), [{
      return $_get(baseType.getContext(), baseType);
    }]>
  ];
}

// Represents the SystemC sc_in<T> type.
// Declares the systemc::InputType in C++.
def InputType : SignalTypesBase<"Input"> {
  let summary = "a SystemC sc_in<T> type";
  let description = [{
    Represents the specialized SystemC port class sc_in<T> as described in
    IEEE 1666-2011 §6.8.
  }];
  let mnemonic = "in";
}

// Represents the SystemC sc_inout<T> type.
// Declares the systemc::InOutType in C++.
def InOutType : SignalTypesBase<"InOut"> {
  let summary = "a SystemC sc_inout<T> type";
  let description = [{
    Represents the specialized SystemC port class sc_inout<T> as described in
    IEEE 1666-2011 §6.10.
  }];
  let mnemonic = "inout";
}

// Represents the SystemC sc_out<T> type.
// Declares the systemc::OutputType in C++.
def OutputType : SignalTypesBase<"Output"> {
  let summary = "a SystemC sc_out<T> type";
  let description = [{
    Represents the specialized SystemC port class sc_out<T> as described in
    IEEE 1666-2011 §6.12.
  }];
  let mnemonic = "out";
}

// Represents the SystemC sc_signal<T> type.
// Declares the systemc::SignalType in C++.
def SignalType : SignalTypesBase<"Signal"> {
  let summary = "a SystemC sc_signal<T> type";
  let description = [{
    Represents the predefined primitive channel sc_signal<T> as described in
    IEEE 1666-2011 §6.4.
  }];
  let mnemonic = "signal";
}

//===----------------------------------------------------------------------===//
// Module type
//===----------------------------------------------------------------------===//

// A type to represent systemc::SCModuleOp instances.
// Declares the systemc::ModuleType in C++.
def ModuleType : SystemCType<"Module"> {
  let summary = "a SystemC module type";
  let description = [{
    Represents a SystemC module instantiation. Example:
    `!systemc.module<moduleName(portName1: type1, portName2: type2)>`
  }];
  let mnemonic = "module";

  let hasCustomAssemblyFormat = 1;

  let parameters = (ins
    "mlir::StringAttr":$moduleName,
    ArrayRefParameter<
      "::circt::systemc::ModuleType::PortInfo", "module ports">:$ports
  );

  let extraClassDeclaration = [{
    using PortInfo = ::circt::systemc::detail::PortInfo;
  }];
}


def ChannelType : AnyTypeOf<[InputType, InOutType, OutputType, SignalType]>;

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES
