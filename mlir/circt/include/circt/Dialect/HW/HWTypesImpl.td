//===- HWTypesImpl.td - HW data type definitions -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data type implementations for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWTYPESIMPL_TD
#define CIRCT_DIALECT_HW_HWTYPESIMPL_TD

include "circt/Dialect/HW/HWDialect.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class HWType<string name> : TypeDef<HWDialect, name> { }

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//

// A parameterized integer type. Declares the hw::IntType in C++.
def IntTypeImpl : HWType<"Int"> {
  let summary = "parameterized-width integer";
  let description = [{
    Parameterized integer types are equivalent to the MLIR standard integer
    type: it is signless, and may be any width integer.  This type represents
    the case when the width is a parameter in the HW dialect sense.
  }];

  let mnemonic = "int";
  let parameters = (ins "::mlir::TypedAttr":$width);

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    /// Get an integer type for the specified width.  Note that this may return
    /// a builtin integer type if the width is a known-constant value.
    static Type get(::mlir::TypedAttr width);
  }];
}

// A simple fixed size array. Declares the hw::ArrayType in C++.
def ArrayTypeImpl : HWType<"Array"> {
  let summary = "fixed-sized array";
  let description = [{
    Fixed sized HW arrays are roughly similar to C arrays. On the wire (vs.
    in a memory), arrays are always packed. Memory layout is not defined as
    it does not need to be since in silicon there is not implicit memory
    sharing.
  }];

  let mnemonic = "array";
  let parameters = (ins "::mlir::Type":$elementType, "::mlir::Attribute":$sizeAttr);
  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static ArrayType get(Type elementType, size_t size) {
      auto *ctx = elementType.getContext();
      auto intType = ::mlir::IntegerType::get(ctx, 64);
      auto sizeAttr = ::mlir::IntegerAttr::get(intType, size);
      return ArrayType::get(ctx, elementType, sizeAttr);
    }
    size_t getSize() const;
  }];
}

// An 'unpacked' array of fixed size.
def UnpackedArrayType : HWType<"UnpackedArray"> {
  let summary = "SystemVerilog 'unpacked' fixed-sized array";
  let description = [{
    Unpacked arrays are a more flexible array representation than packed arrays,
    and are typically used to model memories.  See SystemVerilog Spec 7.4.2.
  }];

  let mnemonic = "uarray";
  let parameters = (ins "::mlir::Type":$elementType, "::mlir::Attribute":$sizeAttr);
  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static UnpackedArrayType get(Type elementType, size_t size) {
      auto *ctx = elementType.getContext();
      auto intType = ::mlir::IntegerType::get(ctx, 64);
      auto sizeAttr = ::mlir::IntegerAttr::get(intType, size);
      return UnpackedArrayType::get(ctx, elementType, sizeAttr);
    }
    size_t getSize() const;
  }];
}

def InOutTypeImpl : HWType<"InOut"> {
  let summary = "inout type";
  let description = [{
    InOut type is used for model operations and values that have "connection"
    semantics, instead of typical dataflow behavior.  This is used for wires
    and inout ports in Verilog.
  }];

  let mnemonic = "inout";
  let parameters = (ins "::mlir::Type":$elementType);
  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static InOutType get(Type elementType) {
      return get(elementType.getContext(), elementType);
    }
  }];
}

// A packed struct. Declares the hw::StructType in C++.
def StructTypeImpl : HWType<"Struct"> {
  let summary = "HW struct type";
  let description = [{
    Represents a structure of name, value pairs.
    !hw.struct<fieldName1: Type1, fieldName2: Type2>
  }];
  let mnemonic = "struct";

  let hasCustomAssemblyFormat = 1;

  let parameters = (
    ins ArrayRefParameter<
      "::circt::hw::StructType::FieldInfo", "struct fields">: $elements
  );

  let extraClassDeclaration = [{
    using FieldInfo = ::circt::hw::detail::FieldInfo;
    mlir::Type getFieldType(mlir::StringRef fieldName);
    void getInnerTypes(mlir::SmallVectorImpl<mlir::Type>&);
    std::optional<unsigned> getFieldIndex(mlir::StringRef fieldName);
    std::optional<unsigned> getFieldIndex(mlir::StringAttr fieldName);
  }];
}

// An enum type. Declares the hw::EnumType in C++.
def EnumTypeImpl : HWType<"Enum"> {
  let summary = "HW Enum type";
  let description = [{
    Represents an enumeration of values. Enums are interpreted as integers with
    a synthesis-defined encoding.
    !hw.enum<field1, field2>
  }];
  let mnemonic = "enum";
  let parameters = (
    ins "mlir::ArrayAttr":$fields
  );

  let extraClassDeclaration = [{
    /// Returns true if the requested field is part of this enum
    bool contains(mlir::StringRef field);

    /// Returns the number of bits used by the enum
    size_t getBitWidth();

    /// Returns the index of the requested field, or a nullopt if the field is
    // not part of this enum.
    std::optional<size_t> indexOf(mlir::StringRef field);
  }];

  let hasCustomAssemblyFormat = 1;
}

// An untagged union. Declares the hw::UnionType in C++.
def UnionTypeImpl : HWType<"Union"> {
  let summary = "An untagged union of types";
  let parameters = (
    ins ArrayRefParameter<
      "::circt::hw::UnionType::FieldInfo", "union fields">: $elements
  );
  let mnemonic = "union";

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    using FieldInfo = ::circt::hw::detail::OffsetFieldInfo;

    FieldInfo getFieldInfo(::mlir::StringRef fieldName);
    
    ::mlir::Type getFieldType(::mlir::StringRef fieldName);
  }];
}

def TypeAliasType : HWType<"TypeAlias"> {
  let summary = "An symbolic reference to a type declaration";
  let description = [{
    A TypeAlias is parameterized by a SymbolRefAttr, which points to a
    TypedeclOp. The root reference should refer to a TypeScope within the same
    outer ModuleOp, and the leaf reference should refer to a type within that
    TypeScope. A TypeAlias is further parameterized by the inner type, which is
    needed to be known at the time the type is parsed.

    Upon construction, a TypeAlias stores the symbol reference and type, and
    canonicalizes the type to resolve any nested type aliases. The canonical
    type is also cached to avoid recomputing it when needed.
  }];

  let mnemonic = "typealias";

  let parameters = (ins
    "mlir::SymbolRefAttr":$ref,
    "mlir::Type":$innerType,
    "mlir::Type":$canonicalType
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::SymbolRefAttr":$ref, "mlir::Type":$innerType)>
  ];

  let extraClassDeclaration = [{
    /// Return the Typedecl referenced by this TypeAlias, given the module to
    /// look in.  This returns null when the IR is malformed.
    TypedeclOp getTypeDecl(const HWSymbolCache &cache);
  }];
}

def ModuleTypeImpl : HWType<"Module"> {
  let summary = "Module Type";
  let description = [{
    Module types have ports.
  }];
  let parameters = (ins ArrayRefParameter<"::circt::hw::ModulePort", "port list">:$ports);
  let assemblyFormat = "`(` $ports `)`";
  let genVerifyDecl = 1;
  let mnemonic = "module";

  let extraClassDeclaration = [{
    // Many of these are transitional and will be removed when modules and instances
    // have moved over to this type.
    size_t getNumInputs();
    size_t getNumOutputs();
    SmallVector<Type> getInputTypes();
    SmallVector<Type> getOutputTypes();
    Type getInputType(size_t);
    Type getOutputType(size_t);
    SmallVector<StringAttr> getInputNames();
    SmallVector<StringAttr> getOutputNames();
    StringAttr getNameAttr(size_t);
    StringRef getName(size_t);
    StringAttr getInputNameAttr(size_t);
    StringRef getInputName(size_t);
    StringAttr getOutputNameAttr(size_t);
    StringRef getOutputName(size_t);
    FunctionType getFuncType();
  }];
}

def HWStringTypeImpl : HWType<"String"> {
  let summary = "String type";
  let description = "Defines a string type for the hw-centric dialects";
  let mnemonic = "string";
}

#endif // CIRCT_DIALECT_HW_HWTYPESIMPL_TD
