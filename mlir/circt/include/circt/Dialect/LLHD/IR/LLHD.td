//===- LLHD.td - LLHD dialect definition -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_LLHD
#define CIRCT_DIALECT_LLHD_IR_LLHD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// LLHD dialect definition
//===----------------------------------------------------------------------===//

def LLHD_Dialect : Dialect {
  let name = "llhd";
  let cppNamespace = "::circt::llhd";

  let description = [{
    A low-level hardware description dialect in MLIR.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register all LLHD types.
    void registerTypes();
    /// Register all LLHD attributes.
    void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Import HW Types
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWTypes.td"

//===----------------------------------------------------------------------===//
// LLHD type definitions
//===----------------------------------------------------------------------===//

include "LLHDTypesImpl.td"

// LLHD Time Type
def LLHD_TimeType : DialectType<LLHD_Dialect,
    CPred<"$_self.isa<TimeType>()">, "LLHD time type", "TimeType">,
    BuildableType<"TimeType::get($_builder.getContext())">;

// Legal underlying types for signals and pointers.
def LLHD_AnyUnderlyingType :
  AnyTypeOf<[HWIntegerType, ArrayType, StructType]>;

// LLHD sig type
class LLHD_SigType<list<Type> allowedTypes>
  : ContainerType<AnyTypeOf<allowedTypes>, CPred<"$_self.isa<SigType>()">,
    "$_self.cast<SigType>().getUnderlyingType()", "LLHD sig type">;

def LLHD_AnySigType : LLHD_SigType<[LLHD_AnyUnderlyingType]>;

// LLHD ptr type
class LLHD_PtrType<list<Type> allowedTypes>
  : ContainerType<AnyTypeOf<allowedTypes>, CPred<"$_self.isa<PtrType>()">,
    "$_self.cast<PtrType>().getUnderlyingType()", "LLHD pointer type">;

def LLHD_AnyPtrType : LLHD_PtrType<[LLHD_AnyUnderlyingType]>;

// LLHD sig or ptr type
class LLHD_SigOrPtrType<list<Type> allowedTypes>
  : AnyTypeOf<[LLHD_SigType<allowedTypes>, LLHD_PtrType<allowedTypes>]>;

def LLHD_AnySigOrPtrType : LLHD_SigOrPtrType<[LLHD_AnyUnderlyingType]>;

//===----------------------------------------------------------------------===//
// LLHD op definition
//===----------------------------------------------------------------------===//

// Base class for all LLHD ops.
class LLHD_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLHD_Dialect, mnemonic, traits> {

  // For each LLHD op, the following static functions need to be defined in
  // LLHDOps.cpp:
  //
  // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
  //                                               OperationState &state);
  // * static void print<op-c++-class-name>(OpAsmPrinter &p, <op-c++-class-name> op)
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// LLHD trait definitions
//===----------------------------------------------------------------------===//

class SameTypeArbitraryWidth<string desc, string lhs, string rhs>
    : PredOpTrait<desc, CPred<"sameKindArbitraryWidth(" # lhs # ".getType(),"
                                                        # rhs # ".getType())">>;

//===----------------------------------------------------------------------===//
// LLHD Operations
//===----------------------------------------------------------------------===//

include "ValueOps.td"
include "SignalOps.td"
include "ExtractOps.td"
include "StructureOps.td"
include "MemoryOps.td"

#endif // CIRCT_DIALECT_LLHD_IR_LLHD
