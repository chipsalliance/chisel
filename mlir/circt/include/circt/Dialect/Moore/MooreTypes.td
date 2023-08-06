//===- MooreTypes.td - Moore type definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SystemVerilog types for the Moore dialect. Includable in other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOORETYPES
#define CIRCT_DIALECT_MOORE_MOORETYPES

include "circt/Dialect/Moore/MooreDialect.td"

class MooreType<Pred condition, string description, string cppClassName>
  : DialectType<MooreDialect, condition, description, cppClassName>;

def PackedType : MooreType<CPred<"$_self.isa<moore::PackedType>()">,
  "packed type", "moore::PackedType">;

def UnpackedType : MooreType<CPred<"$_self.isa<moore::UnpackedType>()">,
  "unpacked type", "moore::UnpackedType">;

/// A simple bit vector type.
def SimpleBitVectorType : MooreType<CPred<[{
    $_self.isa<moore::UnpackedType>() &&
    $_self.cast<moore::UnpackedType>().isSimpleBitVector()
  }]>, "simple bit vector type", "moore::UnpackedType">;

//===----------------------------------------------------------------------===//
// Integer atom types
//===----------------------------------------------------------------------===//

def MooreIntType : MooreType<CPred<"$_self.isa<moore::IntType>()">,
  "an SystemVerilog int", "moore::IntType">;

//===----------------------------------------------------------------------===//
// LValue / RValue predicates
//===----------------------------------------------------------------------===//

class MooreLValueTypeOf<list<Type> allowedTypes>
  : ContainerType<AnyTypeOf<allowedTypes>,
    CPred<"$_self.isa<moore::LValueType>()">,
    "$_self.cast<moore::LValueType>().getNestedType()", "Moore LValue type">;

def MooreLValueType : MooreLValueTypeOf<[AnyType]>;
def MooreRValueType : AnyTypeOf<[MooreIntType]>;

#endif // CIRCT_DIALECT_MOORE_MOORETYPES
