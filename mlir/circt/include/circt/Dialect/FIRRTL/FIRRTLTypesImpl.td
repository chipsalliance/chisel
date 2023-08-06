//===- FIRRTLTypesImpl.td - FIRRTL data type definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data type implementations for the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLTYPESIMPL_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLTYPESIMPL_TD

include "FIRRTLDialect.td"
include "circt/Dialect/HW/HWTypeInterfaces.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class FIRRTLImplType<string name,
                     list<Trait> traits = [],
                     string baseCppClass = "::circt::firrtl::FIRRTLBaseType">
    : TypeDef<FIRRTLDialect, name, traits, baseCppClass> {
  // Storage classes must be defined in C++ and
  // inherit from FIRRTLBaseTypeStorage
  let genStorageClass = false;

  // MLIR generates awkward accessor "getIsConst" for the "isConst" parameter,
  // which is common on FIRRTLBaseType anyway, so we generate the other
  // accessors manually
  let genAccessors = false;
}

//===----------------------------------------------------------------------===//
// Type Traits
//===----------------------------------------------------------------------===//

def WidthQualifiedTypeTrait : NativeTypeTrait<"WidthQualifiedTypeTrait"> {
  let cppNamespace = "::circt::firrtl";
}

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

def SIntImpl : FIRRTLImplType<"SInt",
                              [WidthQualifiedTypeTrait, FieldIDTypeInterface],
                              "::circt::firrtl::IntType"> {
  let summary = "A signed integer type, whose width may not be known.";
  let parameters = (ins "int32_t":$widthOrSentinel, "bool":$isConst);
  let storageClass = "WidthTypeStorage";
  let builders = [
    TypeBuilder<(ins "std::optional<int32_t>":$width, CArg<"bool", "false">:$isConst)>,
    TypeBuilder<(ins)>,
  ];
  let genVerifyDecl = true;
  let extraClassDeclaration = [{
    using WidthQualifiedTypeTrait<SIntType>::getWidth;
    using WidthQualifiedTypeTrait<SIntType>::hasWidth;
    int32_t getWidthOrSentinel() const;
    SIntType getConstType(bool isConst);
  }];
}

def UIntImpl : FIRRTLImplType<"UInt",
                              [WidthQualifiedTypeTrait, FieldIDTypeInterface],
                              "::circt::firrtl::IntType"> {
  let summary = "An unsigned integer type, whose width may not be known.";
  let parameters = (ins "int32_t":$widthOrSentinel, "bool":$isConst);
  let storageClass = "WidthTypeStorage";
  let builders = [
    TypeBuilder<(ins "std::optional<int32_t>":$width, CArg<"bool", "false">:$isConst)>,
    TypeBuilder<(ins)>,
  ];
  let genVerifyDecl = true;
  let extraClassDeclaration = [{
    using WidthQualifiedTypeTrait<UIntType>::getWidth;
    using WidthQualifiedTypeTrait<UIntType>::hasWidth;
    int32_t getWidthOrSentinel() const;
    UIntType getConstType(bool isConst);
  }];
}

def ClockTypeImpl : FIRRTLImplType<"Clock", [FieldIDTypeInterface]> {
  let summary = "Clock signal";
  let parameters = (ins "bool":$isConst);
  let storageClass = "FIRRTLBaseTypeStorage";
  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, false);
    }]>
  ];
  let extraClassDeclaration = [{
    ClockType getConstType(bool isConst);
  }];
}

def ResetTypeImpl : FIRRTLImplType<"Reset", [FieldIDTypeInterface]> {
  let summary = "Reset Signal";
  let parameters = (ins "bool":$isConst);
  let storageClass = "FIRRTLBaseTypeStorage";
  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, false);
    }]>
  ];
  let extraClassDeclaration = [{
    ResetType getConstType(bool isConst);
  }];
}

def AsyncResetTypeImpl : FIRRTLImplType<"AsyncReset", [FieldIDTypeInterface]> {
  let summary = "AsyncReset signal";
  let parameters = (ins "bool":$isConst);
  let storageClass = "FIRRTLBaseTypeStorage";
  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, false);
    }]>
  ];
  let extraClassDeclaration = [{
    AsyncResetType getConstType(bool isConst);
  }];
}

def AnalogTypeImpl : FIRRTLImplType<"Analog",
  [WidthQualifiedTypeTrait, FieldIDTypeInterface]> {
  let summary = "Analog signal";
  let parameters = (ins "int32_t":$widthOrSentinel, "bool":$isConst);
  let storageClass = "WidthTypeStorage";
  let builders = [
    TypeBuilder<(ins "std::optional<int32_t>":$width, CArg<"bool", "false">:$isConst)>,
    TypeBuilder<(ins)>,
  ];
  let extraClassDeclaration = [{
    int32_t getWidthOrSentinel() const;
    AnalogType getConstType(bool isConst);
  }];
  let genVerifyDecl = true;
}

class BaseVectorTypeImpl<string name, string ElementType, list<Trait> traits = [], string BaseType = ElementType> : FIRRTLImplType<name, traits, BaseType> {
  let summary = "a fixed size collection of elements, like an array.";
  let parameters = (ins
      TypeParameter<ElementType, "Type of vector elements">:$elementType,
     "size_t":$numElements,
     "bool":$isConst
  );
  let storageClass = name # "TypeStorage";
  let skipDefaultBuilders = true;
  let builders = [
    TypeBuilderWithInferredContext<(ins
      ElementType:$elementType,
      "size_t":$numElements,
       CArg<"bool", "false">:$isConst)
    >
  ];

  // Additional class declarations to emit.
  code firrtlExtraClassDeclaration = "";

  let extraClassDeclaration = [{
    using ElementType = }] # ElementType # [{;

    ElementType getElementType() const;
    size_t getNumElements() const;

    /// Return the recursive properties of the type.
    RecursiveTypeProperties getRecursiveTypeProperties() const;

    /// Get an integer ID for the field. Field IDs start at 1, and are assigned
    /// to each field in a vector in a recursive depth-first walk of all
    /// elements. A field ID of 0 is used to reference the vector itself.
    uint64_t getFieldID(uint64_t index);

    /// Find the element index corresponding to the desired fieldID.  If the
    /// fieldID corresponds to a field in nested under an element, it will
    /// return the index of the parent element.
    uint64_t getIndexForFieldID(uint64_t fieldID);

    /// Find the index of the element that contains the given fieldID.
    /// As well, rebase the fieldID to the element.
    std::pair<uint64_t, uint64_t> getIndexAndSubfieldID(uint64_t fieldID);

    /// Strip off a single layer of this type and return the sub-type and a
    /// field ID targeting the same field, but rebased on the sub-type.
    std::pair<circt::hw::FieldIDTypeInterface, uint64_t>
    getSubTypeByFieldID(uint64_t fieldID);

    /// Get the maximum field ID in this vector.  This is helpful for
    /// constructing field IDs when this VectorType is nested in another
    /// aggregate type.
    uint64_t getMaxFieldID();

    /// Returns the effective field id when treating the index field as the root
    /// of the type.  Essentially maps a fieldID to a fieldID after a subfield
    /// op. Returns the new id and whether the id is in the given child.
    std::pair<uint64_t, bool> rootChildFieldID(uint64_t fieldID, uint64_t index);

    /// Const support.
    bool isConst();
    ElementType getElementTypePreservingConst();

    /// Return a 'const' or non-'const' version of this type.
    }] # name # [{Type getConstType(bool isConst);

  }] # firrtlExtraClassDeclaration;
}

def FVectorImpl : BaseVectorTypeImpl<"FVector","::circt::firrtl::FIRRTLBaseType", [FieldIDTypeInterface]> {
  let firrtlExtraClassDeclaration = [{
    /// Return this type with any flip types recursively removed from itself.
    FIRRTLBaseType getPassiveType();

    /// Return this type with a 'const' modifiers dropped
    FVectorType getAllConstDroppedType();

    /// Return this type with any type alias types recursively removed from itself.
    FIRRTLBaseType getAnonymousType();
  }];
}

def OpenVectorImpl : BaseVectorTypeImpl<"OpenVector","::circt::firrtl::FIRRTLType", [DeclareTypeInterfaceMethods<FieldIDTypeInterface>]> {
  let genVerifyDecl = 1;
}

class BaseBundleTypeImpl<string name, string ElementType, list<Trait> traits = [], string BaseType = ElementType> : FIRRTLImplType<name, traits, BaseType> {
  let summary = "an aggregate of named elements. This is effectively a struct.";
  let parameters = (ins "ArrayRef<BundleElement>":$elements, "bool":$isConst);
  let storageClass = name # "TypeStorage";
  let skipDefaultBuilders = true;
  let builders = [
    TypeBuilder<(ins "ArrayRef<BundleElement>":$elements, CArg<"bool", "false">:$isConst)>
  ];

  // Additional class declarations to emit.
  code firrtlExtraClassDeclaration = "";

  let extraClassDeclaration = [{
    using ElementType = }] # ElementType # [{;

    /// Each element of a bundle, which is a name and type.
    struct BundleElement {
      StringAttr name;
      bool isFlip;
      ElementType type;

      BundleElement(StringAttr name, bool isFlip, ElementType type)
          : name(name), isFlip(isFlip), type(type) {}

      bool operator==(const BundleElement &rhs) const {
        return name == rhs.name && isFlip == rhs.isFlip && type == rhs.type;
      }
      bool operator!=(const BundleElement &rhs) const {
        return !operator==(rhs);
      }

      friend llvm::hash_code hash_value(const BundleElement &arg) {
        return llvm::hash_combine(arg.name, arg.isFlip, arg.type);
      }
    };

    ArrayRef<BundleElement> getElements() const;

    size_t getNumElements() { return getElements().size(); }

    /// Look up an element's index by name.  This returns None on failure.
    std::optional<unsigned> getElementIndex(StringAttr name);
    std::optional<unsigned> getElementIndex(StringRef name);

    /// Look up an element's name by index. This asserts if index is invalid.
    StringAttr getElementNameAttr(size_t index);
    StringRef getElementName(size_t index);

    /// Look up an element by name.  This returns None on failure.
    std::optional<BundleElement> getElement(StringAttr name);
    std::optional<BundleElement> getElement(StringRef name);

    /// Look up an element by index.  This asserts if index is invalid.
    BundleElement getElement(size_t index);

    /// Look up an element type by name.
    ElementType getElementType(StringAttr name);
    ElementType getElementType(StringRef name);

    /// Look up an element type by index.
    ElementType getElementType(size_t index);

    /// Return the recursive properties of the type.
    RecursiveTypeProperties getRecursiveTypeProperties() const;

    /// Get an integer ID for the field. Field IDs start at 1, and are assigned
    /// to each field in a bundle in a recursive pre-order walk of all fields,
    /// visiting all nested bundle fields.  A field ID of 0 is used to reference
    /// the bundle itself. The ID can be used to uniquely identify any specific
    /// field in this bundle.
    uint64_t getFieldID(uint64_t index);

    /// Find the element index corresponding to the desired fieldID.  If the
    /// fieldID corresponds to a field in a nested bundle, it will return the
    /// index of the parent field.
    uint64_t getIndexForFieldID(uint64_t fieldID);

    /// Find the index of the element that contains the given fieldID.
    /// As well, rebase the fieldID to the element.
    std::pair<uint64_t, uint64_t> getIndexAndSubfieldID(uint64_t fieldID);

    /// Strip off a single layer of this type and return the sub-type and a
    /// field ID targeting the same field, but rebased on the sub-type.
    std::pair<circt::hw::FieldIDTypeInterface, uint64_t>
    getSubTypeByFieldID(uint64_t fieldID);

    /// Get the maximum field ID in this bundle.  This is helpful for
    /// constructing field IDs when this BundleType is nested in another
    /// aggregate type.
    uint64_t getMaxFieldID();

    /// Returns the effective field id when treating the index field as the root
    /// of the type.  Essentially maps a fieldID to a fieldID after a subfield
    /// op. Returns the new id and whether the id is in the given child.
    std::pair<uint64_t, bool> rootChildFieldID(uint64_t fieldID,
                                               uint64_t index);

    using iterator = ArrayRef<BundleElement>::iterator;
    iterator begin() const { return getElements().begin(); }
    iterator end() const { return getElements().end(); }

    /// Const support.
    bool isConst();
    ElementType getElementTypePreservingConst(size_t index);

    /// Return a 'const' or non-'const' version of this type.
    }] # name # [{Type getConstType(bool isConst);

  }] # firrtlExtraClassDeclaration;
}

def BundleImpl : BaseBundleTypeImpl<"Bundle","::circt::firrtl::FIRRTLBaseType", [FieldIDTypeInterface]> {
  let firrtlExtraClassDeclaration = [{
    /// Return this type with any flip types recursively removed from itself.
    FIRRTLBaseType getPassiveType();

    /// Return this type with a 'const' modifiers dropped
    BundleType getAllConstDroppedType();

    /// Return this type with any type alias types recursively removed from itself.
    FIRRTLBaseType getAnonymousType();
  }];
}

def OpenBundleImpl : BaseBundleTypeImpl<"OpenBundle","::circt::firrtl::FIRRTLType", [DeclareTypeInterfaceMethods<FieldIDTypeInterface>]> {
  let genVerifyDecl = 1;
}

def FEnumImpl : FIRRTLImplType<"FEnum", [FieldIDTypeInterface]> {
  let summary = "a sum type of named elements.";
  let parameters = (ins "ArrayRef<EnumElement>":$elements, "bool":$isConst);
  let storageClass = "FEnumTypeStorage";
  let genVerifyDecl = true;
  let skipDefaultBuilders = true;
  let builders = [
    TypeBuilder<(ins "ArrayRef<EnumElement>":$elements, CArg<"bool", "false">:$isConst)>
  ];
  let extraClassDeclaration = [{
    /// Each element of an enum, which is a name and type.
    struct EnumElement {
      StringAttr name;
      FIRRTLBaseType type;

      EnumElement(StringAttr name, FIRRTLBaseType type)
          : name(name), type(type) {}

      bool operator==(const EnumElement &rhs) const {
        return name == rhs.name && type == rhs.type;
      }
      bool operator!=(const EnumElement &rhs) const {
        return !operator==(rhs);
      }

      friend llvm::hash_code hash_value(const FEnumType::EnumElement &arg) {
        return mlir::hash_value(arg.name) ^ mlir::hash_value(arg.type);
      }
    };

    ArrayRef<EnumElement> getElements() const;

    size_t getNumElements() { return getElements().size(); }

    FEnumType getConstType(bool isConst);

    /// Return this type with a 'const' modifiers dropped
    FEnumType getAllConstDroppedType();

    /// Return this type with any type alias types recursively removed from itself.
    FIRRTLBaseType getAnonymousType();

    /// Look up an element's index by name.  This returns None on failure.
    std::optional<unsigned> getElementIndex(StringAttr name);
    std::optional<unsigned> getElementIndex(StringRef name);

    /// Look up an element's name by index. This asserts if index is invalid.
    StringAttr getElementNameAttr(size_t index);
    StringRef getElementName(size_t index);

    /// Look up an element by name.  This returns None on failure.
    std::optional<EnumElement> getElement(StringAttr name);
    std::optional<EnumElement> getElement(StringRef name);

    /// Look up an element by index.  This asserts if index is invalid.
    EnumElement getElement(size_t index);

    /// Look up an element type by name.
    FIRRTLBaseType getElementType(StringAttr name);
    FIRRTLBaseType getElementType(StringRef name);

    /// Look up an element type by index.
    FIRRTLBaseType getElementType(size_t index);
    FIRRTLBaseType getElementTypePreservingConst(size_t index);

    /// Return the recursive properties of the type.
    RecursiveTypeProperties getRecursiveTypeProperties() const;

    /// Get an integer ID for the field. Field IDs start at 1, and are assigned
    /// to each field in a enum in a recursive pre-order walk of all fields,
    /// visiting all nested enum fields.  A field ID of 0 is used to reference
    /// the enum itself. The ID can be used to uniquely identify any specific
    /// field in this enum.
    uint64_t getFieldID(uint64_t index);

    /// Find the element index corresponding to the desired fieldID.  If the
    /// fieldID corresponds to a field in a nested enum, it will return the
    /// index of the parent field.
    uint64_t getIndexForFieldID(uint64_t fieldID);

    /// Find the index of the element that contains the given fieldID.
    /// As well, rebase the fieldID to the element.
    std::pair<uint64_t, uint64_t> getIndexAndSubfieldID(uint64_t fieldID);

    /// Strip off a single layer of this type and return the sub-type and a
    /// field ID targeting the same field, but rebased on the sub-type.
    std::pair<circt::hw::FieldIDTypeInterface, uint64_t>
    getSubTypeByFieldID(uint64_t fieldID);

    /// Get the maximum field ID in this bundle.  This is helpful for
    /// constructing field IDs when this FEnumType is nested in another
    /// aggregate type.
    uint64_t getMaxFieldID();

    /// Returns the effective field id when treating the index field as the root
    /// of the type.  Essentially maps a fieldID to a fieldID after a subfield
    /// op. Returns the new id and whether the id is in the given child.
    std::pair<uint64_t, bool> rootChildFieldID(uint64_t fieldID,
                                               uint64_t index);

    using iterator = ArrayRef<EnumElement>::iterator;
    iterator begin() const { return getElements().begin(); }
    iterator end() const { return getElements().end(); }
  }];
}

def RefImpl : FIRRTLImplType<"Ref",
                             [DeclareTypeInterfaceMethods<FieldIDTypeInterface>],
                             "::circt::firrtl::FIRRTLType"> {
  let summary = [{
    A reference type, such as `firrtl.probe<uint<1>>` or `firrtl.rwprobe<uint<2>>`.

    Used for remote reads and writes of the wrapped base type.

    Parameterized over the referenced base type,
    which must be passive and for now must also be ground.

    Not a base type.

    Values of this type are used to capture dataflow paths,
    and do not represent a circuit element or entity.

    Generally read-only (probe), optionally forceable (rwprobe).
  }];
  let parameters = (ins TypeParameter<"::circt::firrtl::FIRRTLBaseType",
                                      "Type of reference target">:$type,
                        "bool":$forceable);
  let genAccessors = true;
  let genStorageClass = true;
  let genVerifyDecl = true;

  let skipDefaultBuilders = true;
  let builders = [
    TypeBuilderWithInferredContext<(ins "::circt::firrtl::FIRRTLBaseType":$type,
                                        CArg<"bool", "false">:$forceable)>
  ];

  let extraClassDeclaration = [{
    /// Return the recursive properties of the type.
    RecursiveTypeProperties getRecursiveTypeProperties() const;
  }];
}

def BaseTypeAliasImpl : FIRRTLImplType<"BaseTypeAlias", [FieldIDTypeInterface],
                                        "::circt::firrtl::FIRRTLBaseType"> {
  let summary = "type alias for firrtl base types";
  let parameters =
    (ins  "StringAttr":$name, TypeParameter<"::circt::firrtl::FIRRTLBaseType",
                                            "An inner type">:$innerType);
  let storageClass = "BaseTypeAliasStorage";
  let genAccessors = true;
  let skipDefaultBuilders = true;
  let extraClassDeclaration = [{
      // FIRRTLBaseType utils.
      FIRRTLBaseType getAnonymousType();
      FIRRTLBaseType getPassiveType();
      FIRRTLBaseType getConstType(bool isConst);
      FIRRTLBaseType getAllConstDroppedType();

      /// Return the recursive properties of the type.
      RecursiveTypeProperties getRecursiveTypeProperties() const;

      // If a given `newInnerType` is identical to innerType, return `*this`
      // because we can reuse the type alias. Otherwise return `newInnerType`.
      FIRRTLBaseType getModifiedType(FIRRTLBaseType newInnerType);

      /// Implement FieldIDTypeInterface.

      /// Strip off a single layer of this type and return the sub-type and a
      /// field ID targeting the same field, but rebased on the sub-type.
      std::pair<circt::hw::FieldIDTypeInterface, uint64_t>
      getSubTypeByFieldID(uint64_t fieldID);

      /// Get the maximum field ID.
      uint64_t getMaxFieldID();

      /// Returns the effective field id when treating the index field as the root
      /// of the type.  Essentially maps a fieldID to a fieldID after a subfield
      /// op. Returns the new id and whether the id is in the given child.
      std::pair<uint64_t, bool> rootChildFieldID(uint64_t fieldID, uint64_t index);
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
           "::circt::firrtl::FIRRTLBaseType":$innerType)>
  ];
}

//===----------------------------------------------------------------------===//
// Non-Hardware Type Definitions
//===----------------------------------------------------------------------===//

def ClassImpl : FIRRTLImplType<"Class", [], "circt::firrtl::FIRRTLType"> {
  let summary = [{
    An instance of a class.

    Example:
    ```mlir
    !firrtl.class<@Module(in p0: !firrtl.uint<8>, out p1: !firrtl.uint<8>)>
    ```
  }];
  let parameters = (ins
    "FlatSymbolRefAttr":$name,
    ArrayRefParameter<"ClassElement">:$elements
  );
  let genStorageClass = true;
  let genAccessors = false;
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "FlatSymbolRefAttr":$name,
      "ArrayRef<ClassElement>":$elements
    )>
  ];
  let extraClassDeclaration = [{
    /// Helper to parse this type as a module interface.
    static ParseResult parseInterface(
      AsmParser &parser, ClassType &result);

    StringRef getName() const;
    FlatSymbolRefAttr getNameAttr() const;

    ArrayRef<ClassElement> getElements() const;
    size_t getNumElements() const { return getElements().size(); }
    const ClassElement &getElement(IntegerAttr index) const;
    const ClassElement &getElement(size_t index) const;
    std::optional<uint64_t> getElementIndex(StringRef fieldName) const;

    /// Helper to print this type as a module interface.
    /// Example:
    /// ```
    /// @MyClass(in %port: !firrtl.uint<8>, out o: !firrtl.uint<8>)
    /// ```
    void printInterface(AsmPrinter &p) const;
  }];
}

def StringImpl : FIRRTLImplType<"String", [], "circt::firrtl::FIRRTLType"> {
  let summary = [{
    An unlimited length string type. Not representable in hardware.
  }];
  let parameters = (ins);
  let genStorageClass = true;
}

def IntegerImpl : FIRRTLImplType<"FInteger", [], "circt::firrtl::FIRRTLType"> {
  let summary = [{
    An unlimited length signed integer type.  Not representable in hardware.
  }];
  let parameters = (ins);
  let genStorageClass = true;
}

def ListImpl : FIRRTLImplType<"List", [], "circt::firrtl::FIRRTLType"> {
  let summary = [{
    A typed property list of any length.  Not representable in hardware.
  }];
  let parameters = (ins TypeParameter<"circt::firrtl::PropertyType", "element type">:$elementType);
  let genStorageClass = true;
  let genAccessors = true;
}

def MapImpl : FIRRTLImplType<"Map", [], "circt::firrtl::FIRRTLType"> {
  let summary = [{
    A typed map of properties.  Not representable in hardware.
  }];
  let parameters = (ins TypeParameter<"circt::firrtl::PropertyType", "key type">:$keyType,
                    TypeParameter<"circt::firrtl::PropertyType", "value type">:$valueType);
  let genStorageClass = true;
  let genAccessors = true;
}

def PathImpl : FIRRTLImplType<"Path", [], "circt::firrtl::FIRRTLType"> {
  let summary = [{
    A path to a hardware entity.  Not representable in hardware.
  }];
  let parameters = (ins);
  let genStorageClass = true;
  let genAccessors = true;
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLTYPESIMPL_TD
