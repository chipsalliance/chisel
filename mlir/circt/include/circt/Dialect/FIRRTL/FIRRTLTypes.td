//===- FIRRTLTypes.td - FIRRTL Types Definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the FIRRTL types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLTYPES_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLTYPES_TD

include "FIRRTLDialect.td"

//===----------------------------------------------------------------------===//
// FIRRTL Types Definitions
//===----------------------------------------------------------------------===//

class FIRRTLDialectType<Pred pred, string summary, string cpp, string desc = "">
  : DialectType<FIRRTLDialect, pred, summary, cpp> {
  let description = desc;
}

// Helper class to define firrtl types
class FIRRTLDialectTypeHelper<string typeName, string summary, string desc = "">
  : FIRRTLDialectType<CPred<"isa<" # typeName # ">($_self)">, summary,
                      "::circt::firrtl::" # typeName, desc>;

// Helper class to define firrtl types which allows alias types to be used as well.
class FIRRTLDialectAliasTypeHelper<string typeName, string summary, string desc = "">
  : FIRRTLDialectType<CPred<"type_isa<" # typeName # ">($_self)">, summary,
                      "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::" # typeName #">", desc>;

def FIRRTLType : FIRRTLDialectTypeHelper<"FIRRTLType", "FIRRTLType", [{
    Any FIRRTL dialect type, represented by FIRRTLType.
  }]>;

def FIRRTLBaseType : FIRRTLDialectType<
  CPred<"$_self.isa<FIRRTLBaseType>()">,
  "a base type", "::circt::firrtl::FIRRTLBaseType", [{
    A base FIRRTL type, such as a clock, integer, or wire.

    Base types represent circuit elements and constructs,
    examples include ClockType, IntType, or BundleType.
    Nearly all FIRRTL types are base types.

    All base types are FIRRTLType's, and inherit from FIRRTLBaseType.
  }]>;

def ForeignType : FIRRTLDialectType<CPred<"!isa<FIRRTLType>($_self)">,
                                    "foreign type", "::mlir::Type">;

def ClockType : FIRRTLDialectAliasTypeHelper<"ClockType", "clock">;

def NonConstClockType :
  FIRRTLDialectAliasTypeHelper<"ClockType", "clock">,
  BuildableType<"::circt::firrtl::ClockType::get($_builder.getContext())">;

def IntType : FIRRTLDialectAliasTypeHelper<"IntType", "sint or uint type">;

def SIntType : FIRRTLDialectAliasTypeHelper<"SIntType", "sint type">;

def UIntType : FIRRTLDialectAliasTypeHelper<"UIntType", "uint type">;

def AnalogType : FIRRTLDialectAliasTypeHelper<"AnalogType", "analog type">;

def BundleType : FIRRTLDialectAliasTypeHelper<"BundleType", "bundle type">;

def OpenBundleType : FIRRTLDialectTypeHelper<"OpenBundleType", "open bundle type">;

def FVectorType : FIRRTLDialectAliasTypeHelper<"FVectorType", "vector type">;

def OpenVectorType : FIRRTLDialectTypeHelper<"OpenVectorType", "open vector type">;

def FEnumType : FIRRTLDialectAliasTypeHelper<"FEnumType", "enum type">;

def AggregateType : FIRRTLDialectType<
  CPred<"type_isa<FVectorType, BundleType, FEnumType>($_self)">,
  "a aggregate type", "::circt::firrtl::FIRRTLBaseType">;

def PassiveType : FIRRTLDialectType<
  CPred<"type_isa<FIRRTLBaseType>($_self) && cast<FIRRTLBaseType>($_self).isPassive()">,
  "a passive base type (contain no flips)", "::circt::firrtl::FIRRTLBaseType">;

def SizedType : FIRRTLDialectType<CPred<"type_isa<FIRRTLBaseType>($_self) && "
    "!type_cast<FIRRTLBaseType>($_self).hasUninferredWidth()">,
    "a sized type (contains no uninferred widths)", "::circt::firrtl::FIRRTLType">;
def SizedOrForeignType : AnyTypeOf<[SizedType, ForeignType]>;
def SizedPassiveType : FIRRTLDialectType<And<[SizedType.predicate,PassiveType.predicate]>,
    "a sized passive base type (contains no uninferred widths, or flips)", "::circt::firrtl::FIRRTLType">;
def SizedPassiveOrForeignType : AnyTypeOf<[SizedPassiveType, ForeignType]>;

def AsyncResetType : FIRRTLDialectAliasTypeHelper<"AsyncResetType", "async reset type">;

def ResetType : FIRRTLDialectAliasTypeHelper<"ResetType", "reset type">;


def RefType : FIRRTLDialectTypeHelper<"RefType", "reference type">;

def RWProbe : FIRRTLDialectType<
  CPred<"type_isa<RefType>($_self) && type_cast<RefType>($_self).getForceable()">,
   "rwprobe type", "::circt::firrtl::RefType">;

def RWProbeTarget : FIRRTLDialectType<
    CPred<"type_isa<FIRRTLBaseType>($_self) && "
          "!type_cast<FIRRTLBaseType>($_self).hasUninferredReset() &&"
          "!type_cast<FIRRTLBaseType>($_self).hasUninferredWidth()">,
    "RWProbeTarget type (a FIRRTL base type with a known width and not abstract reset)",
    "::circt::firrtl::FIRRTLBaseType">;

def ConnectableType : AnyTypeOf<[FIRRTLBaseType, ForeignType]>;
def StrictConnectableType : AnyTypeOf<[SizedPassiveType, ForeignType]>;

//===----------------------------------------------------------------------===//
// Sized and Unsized Integers
//===----------------------------------------------------------------------===//

def UnsizedUIntType :
  FIRRTLDialectType<
    CPred<"type_isa<UIntType>($_self) && "
          "type_cast<UIntType>($_self).getWidth() == std::nullopt">,
    "uint with uninferred width", "::circt::firrtl::UIntType">;

class SizedUIntType<int width> : FIRRTLDialectType<
    CPred<"type_isa<UIntType>($_self) && "
          "type_cast<UIntType>($_self).getWidth() == " # width>,
    width # "-bit uint", "::circt::firrtl::UIntType">;

class NonConstSizedUIntType<int width> :
  SizedUIntType<width>,
  BuildableType<
    "::circt::firrtl::UIntType::get($_builder.getContext(), " # width # ")">;

def UInt1Type : SizedUIntType<1>;
def UInt2Type : SizedUIntType<2>;
def UInt32Type : SizedUIntType<32>;
def NonConstUInt1Type : NonConstSizedUIntType<1>;

def UInt1OrUnsizedType : AnyTypeOf<[UInt1Type, UnsizedUIntType]>;
def UInt2OrUnsizedType : AnyTypeOf<[UInt2Type, UnsizedUIntType]>;

//===----------------------------------------------------------------------===//
// FIRRTL Types Predicates
//===----------------------------------------------------------------------===//

def OneBitType : FIRRTLDialectType<
 CPred<"(type_isa<IntType>($_self) && type_cast<IntType>($_self).getWidth() == 1) || "
   "(type_isa<AnalogType>($_self) && type_cast<AnalogType>($_self).getWidth() == 1)">,
 "UInt<1>, SInt<1>, or Analog<1>", "::circt::firrtl::FIRRTLBaseType">;


def AnyResetType : FIRRTLDialectType<
    CPred<"type_isa<FIRRTLBaseType>($_self) && type_cast<FIRRTLBaseType>($_self).isResetType()">,
    "Reset", "::circt::firrtl::FIRRTLBaseType">;

def AnyRegisterType : FIRRTLDialectType<
    CPred<"type_isa<FIRRTLBaseType>($_self) && "
          "type_cast<FIRRTLBaseType>($_self).isRegisterType()">,
    "a passive non-'const' base type that does not contain analog",
    "::circt::firrtl::FIRRTLBaseType">;

def UIntSIntClockType : AnyTypeOf<[SIntType, UIntType, ClockType],
                                  "sint, uint, or clock",
                                  "::circt::firrtl::FIRRTLBaseType">;

def 1DVecUIntType : FIRRTLDialectType<
  CPred<"type_isa<FVectorType>($_self) && type_isa<UIntType>(type_cast<FVectorType>($_self).getElementType())">,
  "1d vector with UInt element type", "::circt::firrtl::FIRRTLBaseType">;

def 1DVecIntType : FIRRTLDialectType<
  CPred<"type_isa<FVectorType>($_self) && type_isa<IntType>(type_cast<FVectorType>($_self).getElementType())">,
  "1d vector with Int element type", "::circt::firrtl::FIRRTLBaseType">;

def OneBitCastableType : AnyTypeOf<
  [OneBitType, AnyResetType, AsyncResetType, ClockType],
  "1-bit uint/sint/analog, reset, asyncreset, or clock",
                                  "::circt::firrtl::FIRRTLBaseType">;

//===----------------------------------------------------------------------===//
// Constraints on RefOps
//===----------------------------------------------------------------------===//

class RefTypeConstraint<string ref, string base>
  : TypesMatchWith<"reference base type should match",
                   ref, base,
                   "type_cast<RefType>($_self).getType()">;

class RefResultTypeConstraint<string base, string ref>
  : TypesMatchWith<"reference base type should match",
                   base, ref,
                   "RefType::get(type_cast<FIRRTLBaseType>($_self).getPassiveType())">;

class CompatibleRefTypes<string dst, string src>
  : PredOpTrait<"reference " # dst # " must be compatible with reference " # src #
                ": recursively same or uninferred of same and can only demote rwprobe to probe",
                CPred<"circt::firrtl::areTypesRefCastable($" # dst # ".getType(), $" # src # ".getType())">>;

//===----------------------------------------------------------------------===//
// Property Types
//===----------------------------------------------------------------------===//

def PropertyType : FIRRTLDialectTypeHelper<
  "PropertyType", "property type", [{
    A FIRRTL property type, such as a string.
  }]>;

def ClassType : FIRRTLDialectTypeHelper<"ClassType", "class type">;

def StringType : FIRRTLDialectTypeHelper<"StringType", "string type">,
  BuildableType<"::circt::firrtl::StringType::get($_builder.getContext())">;

def FIntegerType : FIRRTLDialectTypeHelper<"FIntegerType", "integer type">,
  BuildableType<"::circt::firrtl::FIntegerType::get($_builder.getContext())">;

def ListType : FIRRTLDialectTypeHelper<"ListType", "list type">;
def MapType : FIRRTLDialectTypeHelper<"MapType", "map type">;

def PathType : FIRRTLDialectTypeHelper<"PathType", "path type">,
  BuildableType<"::circt::firrtl::PathType::get($_builder.getContext())">;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLTYPES_TD
