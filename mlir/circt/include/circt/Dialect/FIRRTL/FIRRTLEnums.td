//===- FIRRTLEnums.td - FIRRTL Enum Definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of FIRRTL enum attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLENUMS_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLENUMS_TD

include "FIRRTLDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Name Kinds
//===----------------------------------------------------------------------===//

/// An attribute describing a kind of a name which defines name preservation
/// semantics. We preserve names tagged with `InterestingName`.
def DroppableName: I32EnumAttrCase<"DroppableName", 0, "droppable_name">;
def InterestingName: I32EnumAttrCase<"InterestingName", 1, "interesting_name">;

def NameKindEnumImpl: I32EnumAttr<"NameKindEnum", "name kind",
              [DroppableName, InterestingName]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::circt::firrtl";
}

def NameKindAttr: EnumAttr<FIRRTLDialect, NameKindEnumImpl, "name_kind">;

//===----------------------------------------------------------------------===//
// Module Instantiation Conventions
//===----------------------------------------------------------------------===//

def Convention : I32EnumAttr<"Convention", "lowering convention", [
                           I32EnumAttrCase<"Internal", 0, "internal">,
                           I32EnumAttrCase<"Scalarized", 1, "scalarized">]> {
  let genSpecializedAttr = 0;
}

def ConventionAttr : EnumAttr<FIRRTLDialect, Convention, "convention">;

//===----------------------------------------------------------------------===//
// Group Lowering Conventions
//===----------------------------------------------------------------------===//

def GroupConvention : I32EnumAttr<"GroupConvention", "group convention", [
  I32EnumAttrCase<"Bind", 0, "bind">]> {

  let genSpecializedAttr = 0;
}

def GroupConventionAttr : EnumAttr<FIRRTLDialect, GroupConvention, "groupconvention">;

//===----------------------------------------------------------------------===//
// Read Under Write Behaviour
//===----------------------------------------------------------------------===//

def RUW_Undefined: I32EnumAttrCase<"Undefined", 0>;
def RUW_Old: I32EnumAttrCase<"Old", 1>;
def RUW_New: I32EnumAttrCase<"New", 2>;

def RUWAttr: I32EnumAttr<"RUWAttr", "Read Under Write Enum",
                         [RUW_Undefined, RUW_Old, RUW_New]>;

//===----------------------------------------------------------------------===//
// Memory Port Direction
//===----------------------------------------------------------------------===//

def MemDir_Infer    : I32EnumAttrCase<"Infer", 0>;
def MemDir_Read     : I32EnumAttrCase<"Read", 1>;
def MemDir_Write    : I32EnumAttrCase<"Write", 2>;
def MemDir_ReadWrite: I32EnumAttrCase<"ReadWrite", 3>;

def MemDirAttr: I32EnumAttr<"MemDirAttr", "Memory Direction Enum",
                            [MemDir_Infer, MemDir_Read, MemDir_Write,
                              MemDir_ReadWrite]>;

//===----------------------------------------------------------------------===//
// Event Control
//===----------------------------------------------------------------------===//

/// AtPosEdge triggers on a rise from 0 to 1/X/Z, or X/Z to 1.
def AtPosEdge: I32EnumAttrCase<"AtPosEdge", 0, "posedge">;
/// AtNegEdge triggers on a drop from 1 to 0/X/Z, or X/Z to 0.
def AtNegEdge: I32EnumAttrCase<"AtNegEdge", 1, "negedge">;
/// AtEdge(v) is syntactic sugar for "AtPosEdge(v) or AtNegEdge(v)".
def AtEdge   : I32EnumAttrCase<"AtEdge", 2, "edge">;

def EventControlAttr : I32EnumAttr<"EventControl", "edge control trigger",
                                   [AtPosEdge, AtNegEdge, AtEdge]>  {}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLENUMS_TD
