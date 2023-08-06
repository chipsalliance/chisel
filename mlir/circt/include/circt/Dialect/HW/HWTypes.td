//===- HWTypes.td - HW data type definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data types for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWTYPES
#define CIRCT_DIALECT_HW_HWTYPES

include "circt/Dialect/HW/HWDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWIntegerType : DialectType<HWDialect,
    CPred<"::circt::hw::isHWIntegerType($_self)">,
    "a signless integer bitvector",
    "::circt::hw::TypeVariant<::mlir::IntegerType, ::circt::hw::IntType>">;

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWValueType : DialectType<HWDialect,
    CPred<"::circt::hw::isHWValueType($_self)">, "a known primitive element">;

// Type constraint that indicates that an operand/result may only be a valid
// non-directional type.
def HWNonInOutType : DialectType<HWDialect,
    CPred<"!circt::hw::hasHWInOutType($_self)">, "a type without inout">;

def InOutType : DialectType<HWDialect,
    CPred<"::circt::hw::type_isa<circt::hw::InOutType>($_self)">,
          "InOutType", "InOutType">;

// A handle to refer to circt::hw::ArrayType in ODS.
def ArrayType : DialectType<HWDialect,
    CPred<"::circt::hw::type_isa<circt::hw::ArrayType>($_self)">,
          "an ArrayType", "::circt::hw::TypeAliasOr<circt::hw::ArrayType>">;

// A handle to refer to circt::hw::StructType in ODS.
def StructType : DialectType<HWDialect,
    CPred<"::circt::hw::type_isa<circt::hw::StructType>($_self)">,
          "a StructType", "::circt::hw::TypeAliasOr<circt::hw::StructType>">;

// A handle to refer to circt::hw::UnionType in ODS.
def UnionType : DialectType<HWDialect,
    CPred<"::circt::hw::type_isa<circt::hw::UnionType>($_self)">,
          "a UnionType", "::circt::hw::TypeAliasOr<hw::UnionType>">;

// A handle to refer to circt::hw::EnumType in ODS.
def EnumType : DialectType<HWDialect,
    CPred<"::circt::hw::type_isa<circt::hw::EnumType>($_self)">,
          "a EnumType", "::circt::hw::TypeAliasOr<circt::hw::EnumType>">;

def HWAggregateType : DialectType<HWDialect,
    CPred<[{::circt::hw::type_isa<
      circt::hw::ArrayType,
      circt::hw::UnpackedArrayType,
      circt::hw::StructType>($_self)}]>,
    "an ArrayType or StructType",
    [{::circt::hw::TypeVariant<
      ::circt::hw::ArrayType,
      ::circt::hw::UnpackedArrayType,
      ::circt::hw::StructType>}]>;

// A handle to refer to circt::hw::ModuleType in ODS.
def ModuleType : DialectType<HWDialect,
    CPred<"isa<circt::hw::ModuleType>($_self)">,
          "a module", "::circt::hw::ModuleType">;

// A handle to refer to circt::hw::StringType in ODS.
def HWStringType :
    DialectType<HWDialect,
      CPred<"::circt::hw::type_isa<circt::hw::StringType>($_self)">,
      "a HW string", "::circt::hw::StringType">,
    BuildableType<"::circt::hw::StringType::get($_builder.getContext())">;

/// A flat symbol reference or a reference to a name within a module.
def NameRefAttr : Attr<
  CPred<"$_self.isa<::mlir::FlatSymbolRefAttr, ::circt::hw::InnerRefAttr>()">,
  "name reference attribute">{
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
  let valueType = NoneType;
}

// Like a FlatSymbolRefArrayAttr, but can also refer to names inside modules.
def NameRefArrayAttr : TypedArrayAttrBase<NameRefAttr,
  "name reference array attribute"> {
  let constBuilderCall = "$_builder.getArrayAttr($0)";
}

def InnerSymProperties : AttrDef<HWDialect, "InnerSymProperties"> {
  let mnemonic = "innerSymProps";
  let parameters = (ins
         "mlir::StringAttr":$name,
         DefaultValuedParameter<"uint64_t", "0">:$fieldID,
         DefaultValuedParameter<"::mlir::StringAttr", "public">:$sym_visibility
                     );
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$sym),[{
      return get(sym.getContext(), sym, 0,
                        mlir::StringAttr::get(sym.getContext(), "public") );
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
  // The assembly format is as follows:
  // "`<` `@` $name `,` $fieldID `,` $sym_visibility `>`";
}


def InnerSymAttr : AttrDef<HWDialect, "InnerSym"> {
  let summary = "Inner symbol definition";
  let description = [{
    Defines the properties of an inner_sym attribute. It specifies the symbol
    name and symbol visibility for each field ID. For any ground types,
    there are no subfields and the field ID is 0. For aggregate types, a
    unique field ID is assigned to each field by visiting them in a
    depth-first pre-order. The custom assembly format ensures that for ground
    types, only `@<sym_name>` is printed.
  }];
  let mnemonic = "innerSym";
  let parameters = (ins ArrayRefParameter<"InnerSymPropertiesAttr">:$props);
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$sym),[{
      assert(!sym.getValue().empty() && "symbol name must be non-empty");
      return get(sym.getContext(),
      {InnerSymPropertiesAttr::get(sym.getContext(), sym, 0,
                        mlir::StringAttr::get(sym.getContext(), "public"))});
    }]>,
    // Create an empty array, represents an invalid InnerSym.
    AttrBuilder<(ins),[{
      return get($_ctxt, {});
    }]>
  ];
  let extraClassDeclaration = [{
    /// Get the inner sym name for fieldID, if it exists.
    mlir::StringAttr getSymIfExists(uint64_t fieldID) const;

    /// Get the inner sym name for fieldID=0, if it exists.
    mlir::StringAttr getSymName() const { return getSymIfExists(0); }

    /// Get the number of inner symbols defined.
    size_t size() const { return getProps().size(); }

    /// Check if this is an empty array, no sym names stored.
    bool empty() const { return getProps().empty(); }

    /// Return an InnerSymAttr with the inner symbol for the specified fieldID removed.
    InnerSymAttr erase(uint64_t fieldID) const;

    using iterator = mlir::ArrayRef<InnerSymPropertiesAttr>::iterator;
    /// Iterator begin for all the InnerSymProperties.
    iterator begin() const { return getProps().begin(); }

    /// Iterator end for all the InnerSymProperties.
    iterator end() const { return getProps().end(); }

    /// Invoke the func, for all sym names. Return success(),
    /// if the callback function never returns failure().
    mlir::LogicalResult walkSymbols(llvm::function_ref<
                              mlir::LogicalResult (::mlir::StringAttr)>) const;
  }];

  let hasCustomAssemblyFormat = 1;
  // Example format:
  // firrtl.wire sym [<@x,1,private>, <@w,2,public>, <@syh,4,public>]
}

#endif // CIRCT_DIALECT_HW_HWTYPES
