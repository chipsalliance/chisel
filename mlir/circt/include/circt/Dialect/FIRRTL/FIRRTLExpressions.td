//===- FIRRTLExpressions.td - FIRRTL expression ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for FIRRTL expressions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLEXPRESSIONS_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLEXPRESSIONS_TD

include "FIRRTLDialect.td"
include "FIRRTLTypes.td"
include "FIRRTLEnums.td"
include "circt/Dialect/HW/HWAttributesNaming.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Types.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SameOperandsIntTypeKind : NativeOpTrait<"SameOperandsIntTypeKind"> {
  let cppNamespace = "::circt::firrtl";
}

// A common base class for operations that implement type inference and parsed
// argument validation.
class FIRRTLExprOp<string mnemonic, list<Trait> traits = []> :
    FIRRTLOp<mnemonic, traits # [HasCustomSSAName, InferTypeOpInterface, Pure]> {

  // The narrow operation-specific type inference method. Operations can
  // override this with an inline declaration for the class header, or just
  // leave it as is and implement the function in a cpp file.
  code inferTypeDecl = [{
    /// Infer the return type of this operation.
    static FIRRTLType inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      std::optional<Location> loc);
  }];

  // The operation-specific validator for a parsed list of operands and
  // constants. Operations can override this with an inline declaration for the
  // class header, or just leave it as is and implement the function in a cpp
  // file.
  code parseValidator = "";
  code parseValidatorDecl = [{
    /// Check that the parser has consumed the correct number of operands and
    /// constants.
    static LogicalResult validateArguments(ValueRange operands,
                                           ArrayRef<NamedAttribute> attrs,
                                           Location loc)
  }] # !if(!empty(parseValidator), ";", !subst("$_impl", parseValidator, [{ {
    return $_impl(operands, attrs, loc);
  } }]));

  // Additional class declarations to emit alongside the type inference.
  code firrtlExtraClassDeclaration = "";

  let extraClassDeclaration = firrtlExtraClassDeclaration # inferTypeDecl #
      parseValidatorDecl # [{
    /// Infer the return types of this operation. This is called by the
    /// `InferTypeOpInterface`. We simply forward to a narrower
    /// operation-specific implementation which is sufficient for FIRRTL ops.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results) {
      return impl::inferReturnTypes(context, loc, operands, attrs, properties,
        regions, results, &inferReturnType);
    }

    /// Check that the parser has consumed the correct number of operands and
    /// constants, and infer the appropriate return type for the operation.
    static FIRRTLType validateAndInferReturnType(ValueRange operands,
                                                 ArrayRef<NamedAttribute> attrs,
                                                 Location loc) {
      if (failed(validateArguments(operands, attrs, loc)))
        return {};
      return inferReturnType(operands, attrs, loc);
    }

    // Returns when two result types are compatible for this op; method used by
    // InferTypeOpInterface.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      if (l.size () != r.size())
        return false;
      for (auto [l, r] : llvm::zip(l, r)) {
        if (!areAnonymousTypesEquivalent(l, r))
          return false;
      }
      return true;
    }
  }];
}

def ConstantOp : FIRRTLOp<"constant",
      [Pure, ConstantLike, FirstAttrDerivedResultType,
       HasCustomSSAName]> {
  let summary = "Produce a constant value";
  let description = [{
    The constant operation produces a constant value of SInt or UInt type, it
    never produces a zero bit wide result.
    ```
      %result = firrtl.constant 42 : t1
    ```
    }];

  let arguments = (ins APSIntAttr:$value);
  let results = (outs IntType:$result);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "IntType":$type, "const APInt &":$value)>,
    OpBuilder<(ins "const APSInt &":$value)>
  ];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def SpecialConstantOp : FIRRTLOp<"specialconstant",
      [Pure, ConstantLike, HasCustomSSAName]> {
  let summary = "Produce a constant Reset or Clock value";
  let description = [{
    The constant operation produces a constant value of Reset, AsyncReset, or
    Clock type. The value can only be 0 or 1.
    ```
      %result = firrtl.specialconstant 1 : !firrtl.clock
    ```
    }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs
    AnyTypeOf<[ClockType, ResetType, AsyncResetType]>:$result);

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
}

def AggregateConstantOp : FIRRTLOp<"aggregateconstant", [Pure, ConstantLike]> {
  let summary = "Produce a constant of a passive aggregate value";
  let description = [{
    The constant operation produces a constant value of an aggregate type.  The
    type must be passive.  Clock and reset values are supported.
    For nested aggregates, embedded arrays are used.
    ```
      %result = firrtl.aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
      %result = firrtl.aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
      %result = firrtl.aggregateconstant [[1, 2], [3, 5]] : !firrtl.vector<!firrtl.bundle<a: uint<8>, b: uint<5>>, 2>
    ```
  }];

  let extraClassDeclaration = [{
    Attribute getAttributeFromFieldID(uint64_t fieldID);
  }];

  let arguments = (ins ArrayAttr:$fields);
  let results = (outs AggregateType:$result);

  let assemblyFormat = "$fields attr-dict `:` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

def BundleCreateOp : FIRRTLOp<"bundlecreate"> {
  let summary = "Produce a bundle value";
  let description = [{
    Create an bundle from component values.  This is equivalent in terms of
    flow to creating a node.
    ```
      %result = firrtl.bundlecreate %1, %2, %3 : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
    ```
  }];

  let arguments = (ins Variadic<FIRRTLBaseType>:$fields);
  let results = (outs BundleType:$result);

  let assemblyFormat = "$fields attr-dict `:` functional-type($fields, $result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def VectorCreateOp : FIRRTLOp<"vectorcreate"> {
  let summary = "Produce a vector value";
  let description = [{
    Create a vector from component values.  This is equivalent in terms of
    flow to creating a node. The first operand indicates 0-th element of
    the result.
    ```
      %result = firrtl.vectorcreate %1, %2, %3 : !firrtl.vector<uint<8>, 3>

    ```
  }];

  let arguments = (ins Variadic<FIRRTLBaseType>:$fields);
  let results = (outs FVectorType:$result);

  let assemblyFormat = "$fields attr-dict `:` functional-type($fields, $result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def FEnumCreateOp : FIRRTLOp<"enumcreate"> {
  let summary = "Produce a enum value";
  let description = [{
    Create an enum from tag and value.
    ```mlir
      %result = firrtl.enumcreate field-name(%input) : !firrtl.enum<None: uint<0>, Some: uint<8>>

    ```
  }];

  let arguments = (ins I32Attr:$fieldIndex, FIRRTLBaseType:$input);
  let results = (outs FEnumType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "FEnumType":$resultType, "StringRef":$fieldName, "Value":$input), [{
      auto fieldIndex = resultType.getElementIndex(fieldName);
      assert(fieldIndex.has_value() && "invalid type constructor");
      return build($_builder, $_state, resultType, (uint32_t)*fieldIndex, input);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the name attribute of the accessed field.
    StringAttr getFieldNameAttr() {
      return getResult().getType().get().getElementNameAttr(getFieldIndex());
    }

    /// Return the name of the accessed field.
    StringRef getFieldName() {
      return getFieldNameAttr().getValue();
    }
  }];
}


def InvalidValueOp : FIRRTLOp<"invalidvalue",
      [MemoryEffects<[MemAlloc]>, HasCustomSSAName]> {
  let summary = "InvalidValue primitive";
  let description = [{
    The InvalidValue operation returns an invalid value of a specified type:
    ```
      %result = firrtl.invalid : !firrtl.uint<1>
    ```

    This corresponds to the FIRRTL invalidate operation without the implicit
    connect semantics.  Each invalid op produces a unique invalid value.
    InvalidOp is not constant-like
  }];

  let arguments = (ins);
  let results = (outs FIRRTLBaseType:$result);
  let hasCanonicalizeMethod = 1;

  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

class BaseSubfieldOp<string name, Type btype, Type rtype> : FIRRTLExprOp<name> {
  let summary = "Extract a subfield of another value";
  let description = [{
    The subfield expression refers to a subelement of an expression with a
    bundle type.
    ```
      %result = firrtl.}] # name # [{ %input[field-name] : !input-type
    ```
    }];

  let arguments = (ins btype:$input, I32Attr:$fieldIndex);
  let results = (outs rtype:$result);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "StringRef":$fieldName), [{
      auto bundleType = firrtl::type_cast<}] # btype # [{>(input.getType());
      auto fieldIndex = bundleType.getElementIndex(fieldName);
      assert(fieldIndex.has_value() && "subfield operation to unknown field");
      return build($_builder, $_state, input, *fieldIndex);
    }]>
  ];

  let firrtlExtraClassDeclaration = [{
    using InputType = }] # btype # [{;

    /// Return true if the specified field is flipped.
    bool isFieldFlipped();

    /// Return a `FieldRef` to the accessed field.
    FieldRef getAccessedField() {
      return FieldRef(getInput(), firrtl::type_cast<InputType>(getInput().getType())
                                            .getFieldID(getFieldIndex()));
    }

    /// Return the name of the accessed field.
    StringRef getFieldName() {
      return firrtl::type_cast<InputType>(getInput().getType()).getElementName(getFieldIndex());
    }
  }];
}

def SubfieldOp : BaseSubfieldOp<"subfield", BundleType, FIRRTLBaseType> {
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}
def OpenSubfieldOp : BaseSubfieldOp<"opensubfield", OpenBundleType, FIRRTLType>;

def SubindexOp : FIRRTLExprOp<"subindex"> {
  let summary = "Extract an element of a vector value";
  let description = [{
    The subindex expression statically refers, by index, to a subelement
    of an expression with a vector type. The index must be a non-negative
    integer and cannot be equal to or exceed the length of the vector it
    indexes.
    ```
      %result = firrtl.subindex %input[index] : t1
    ```
    }];

  let arguments = (ins FVectorType:$input, I32Attr:$index);
  let results = (outs FIRRTLBaseType:$result);
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat =
     "$input `[` $index `]` attr-dict `:` qualified(type($input))";

  let firrtlExtraClassDeclaration = [{
    /// Return a `FieldRef` to the accessed field.
    FieldRef getAccessedField() {
      return FieldRef(getInput(), getInput().getType().get().getFieldID(getIndex()));
    }
    using InputType = FVectorType;
  }];
}

def OpenSubindexOp : FIRRTLExprOp<"opensubindex"> {
  let summary = "Extract an element of a vector value";
  let description = [{
    The subindex expression statically refers, by index, to a subelement
    of an expression with a vector type. The index must be a non-negative
    integer and cannot be equal to or exceed the length of the vector it
    indexes.
    ```
      %result = firrtl.opensubindex %input[index] : t1
    ```
    }];

  let arguments = (ins OpenVectorType:$input, I32Attr:$index);
  let results = (outs FIRRTLType:$result);

  let assemblyFormat =
     "$input `[` $index `]` attr-dict `:` qualified(type($input))";

  let firrtlExtraClassDeclaration = [{
    /// Return a `FieldRef` to the accessed field.
    FieldRef getAccessedField() {
      return FieldRef(getInput(), getInput().getType().getFieldID(getIndex()));
    }

    using InputType = OpenVectorType;
  }];
}

def SubaccessOp : FIRRTLExprOp<"subaccess"> {
  let summary = "Extract a dynamic element of a vector value";
  let description = [{
    The subaccess expression dynamically refers to a subelement of a
    vector-typed expression using a calculated index. The index must be an
    expression with an unsigned integer type.
    ```
      %result = firrtl.subaccess %input[%idx] : t1, t2
    ```
    }];

  let arguments = (ins FVectorType:$input, UIntType:$index);
  let results = (outs FIRRTLBaseType:$result);

  let assemblyFormat =
     "$input `[` $index `]` attr-dict `:` qualified(type($input)) `,` qualified(type($index))";

  let hasCanonicalizer = true;
}

def IsTagOp : FIRRTLExprOp<"istag"> {
  let summary = "Test the active variant of an enumeration";
  let description = [{
    This operation is used to test the active variant of an enumeration. The
    tag tested for must be one of the possible variants of the input type.  If
    the tag is the currently active variant the result will be 1, otherwise the
    result will be 0.

    Example:
    ```mlir
      %0 = firrtl.istag A %v : !firrtl.enum<A: UInt<0>, B: UInt<0>>
    ```
  }];
  let arguments = (ins FEnumType:$input, I32Attr:$fieldIndex);
  let results = (outs UInt1Type:$result);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$input, "StringRef":$fieldName), [{
      auto enumType = firrtl::type_cast<FEnumType>(input.getType());
      auto fieldIndex = enumType.getElementIndex(fieldName);
      assert(fieldIndex.has_value() && "subtag operation to unknown field");
      return build($_builder, $_state, input, *fieldIndex);
    }]>
  ];

  let firrtlExtraClassDeclaration = [{
    /// Return a `FieldRef` to the accessed field.
    FieldRef getAccessedField() {
      return FieldRef(getInput(), firrtl::type_cast<FEnumType>(getInput().getType())
                                            .getFieldID(getFieldIndex()));
    }

    /// Return the name of the accessed field.
    StringAttr getFieldNameAttr() {
      return firrtl::type_cast<FEnumType>(getInput().getType())
        .getElementNameAttr(getFieldIndex());
    }

    /// Return the name of the accessed field.
    StringRef getFieldName() {
      return getFieldNameAttr().getValue();
    }
  }];
}

def SubtagOp : FIRRTLExprOp<"subtag"> {
  let summary = "Extract an element of a enum value";
  let description = [{
    The subtag expression refers to a subelement of a
    enum-typed expression.
    ```
      %result = firrtl.subtag %input[field-name] : !input-type
    ```
    }];

  let arguments = (ins FEnumType:$input, I32Attr:$fieldIndex);
  let results = (outs FIRRTLBaseType:$result);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "StringRef":$fieldName), [{
      auto enumType = firrtl::type_cast<FEnumType>(input.getType());
      auto fieldIndex = enumType.getElementIndex(fieldName);
      assert(fieldIndex.has_value() && "subtag operation to unknown field");
      return build($_builder, $_state, input, *fieldIndex);
    }]>
  ];

  let firrtlExtraClassDeclaration = [{
    /// Return a `FieldRef` to the accessed field.
    FieldRef getAccessedField() {
      return FieldRef(getInput(), getInput().getType().get()
                                            .getFieldID(getFieldIndex()));
    }

    /// Return the name of the accessed field.
    StringAttr getFieldNameAttr() {
      return getInput().getType().get().getElementNameAttr(getFieldIndex());
    }

    /// Return the name of the accessed field.
    StringRef getFieldName() {
      return getFieldNameAttr().getValue();
    }
  }];
}

def TagExtractOp : FIRRTLExprOp<"tagextract", [InferTypeOpInterface]> {
  let summary = "Extract the tag from a value";
  let description = [{
    The tagextract expression returns the binary value of the current tag of an
    enum value.
    ```
      %result = firrtl.tagextract %input : !input-type
    ```
    }];

  let arguments = (ins FEnumType:$input);
  let results = (outs UIntType:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input))";
}

def MultibitMuxOp : FIRRTLExprOp<"multibit_mux"> {
  let summary = "Multibit multiplexer";
  let description = [{
    The multibit mux expression dynamically selects operands. The
    index must be an expression with an unsigned integer type.
    ```
      %result = firrtl.multibit_mux %index,
                %v_{n-1}, ..., %v_2, %v_1, %v_0  : t1, t2
    ```
    The order of operands is defined in the same way as hw dialect.
    For the example above, if `%index` is 0, then the value is `%v_0`.
    }];

  let arguments = (ins UIntType:$index, Variadic<FIRRTLBaseType>:$inputs);
  let results = (outs FIRRTLBaseType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = true;
  let hasCanonicalizeMethod = true;
}

//===----------------------------------------------------------------------===//
// Primitive Operations
//===----------------------------------------------------------------------===//

/// PrimOp is a FIRRTLOp without side effects.
class PrimOp<string mnemonic, list<Trait> traits = []> :
    FIRRTLExprOp<mnemonic, traits> {
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

// Base class for binary primitive operators.
class BinaryPrimOp<string mnemonic, Type lhsType, Type rhsType, Type resultType,
                   list<Trait> traits = []> : PrimOp<mnemonic, traits> {
  let arguments = (ins lhsType:$lhs, rhsType:$rhs);
  let results = (outs resultType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:`
       `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
  }];

  // Give concrete operations a chance to set a type inference callback. If left
  // empty, a declaration for `inferBinaryReturnType` will be emitted that the
  // operation is expected to implement.
  code inferType = "";
  let inferTypeDecl = !if(!empty(inferType), [{
    /// Infer the return type of this binary operation.
    static FIRRTLType inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                            std::optional<Location> loc);
    }], "") # !subst("$_infer", !if(!empty(inferType), "inferBinaryReturnType",
      inferType), [{
    /// Infer the return type of this operation.
    static FIRRTLType inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      std::optional<Location> loc) {
      return $_infer(firrtl::type_cast<FIRRTLType>(operands[0].getType()),
                     firrtl::type_cast<FIRRTLType>(operands[1].getType()),
                     loc);
    }
  }]);
  let parseValidator = "impl::validateBinaryOpArguments";
}

// A binary operation on two integer-typed arguments of the same kind.
class IntBinaryPrimOp<string mnemonic, Type resultType,
                      list<Trait> traits = []> :
   BinaryPrimOp<mnemonic, IntType, IntType, resultType,
                traits # [SameOperandsIntTypeKind]>;

let inferType = "impl::inferAddSubResult" in {
  let hasCanonicalizer = true in {
    def AddPrimOp : IntBinaryPrimOp<"add", IntType, [Commutative]>;
    def SubPrimOp : IntBinaryPrimOp<"sub", IntType>;
  }
}

def MulPrimOp : IntBinaryPrimOp<"mul", IntType, [Commutative]>;
def DivPrimOp : IntBinaryPrimOp<"div", IntType> {
  let description = [{
    Divides the first argument (the numerator) by the second argument
    (the denominator) truncating the result (rounding towards zero).

    **If the denominator is zero, the result is undefined.**

    The compiler may optimize this undefined behavior in any way it
    wants. Notably `div(a, a)` will be optimized to `1`. This may cause
    erroneous formal equivalence mismatches between unoptimized and
    optimized FIRRTL dialects that are separately converted to Verilog.
  }];
}
def RemPrimOp : IntBinaryPrimOp<"rem", IntType>;
let inferType = "impl::inferBitwiseResult" in {
  let hasCanonicalizer = true in {
    def AndPrimOp : IntBinaryPrimOp<"and", UIntType, [Commutative]>;
    def OrPrimOp  : IntBinaryPrimOp<"or",  UIntType, [Commutative]>;
    def XorPrimOp : IntBinaryPrimOp<"xor", UIntType, [Commutative]>;
  }
}

// Element-wise and/or/xor operations whose result and operands are 1d vectors with
// a same type. Apply bitwise operation to each element.
let inferType = "impl::inferElementwiseResult" in
let hasFolder = 0 in {
  def ElementwiseAndPrimOp : BinaryPrimOp<"elementwise_and", 1DVecIntType,
                                1DVecIntType, 1DVecUIntType, [Commutative]>;
  def ElementwiseOrPrimOp  : BinaryPrimOp<"elementwise_or", 1DVecIntType,
                                1DVecIntType, 1DVecUIntType, [Commutative]>;
  def ElementwiseXorPrimOp : BinaryPrimOp<"elementwise_xor", 1DVecUIntType,
                                1DVecIntType, 1DVecUIntType, [Commutative]>;
}

// Comparison Operations
let inferType = "impl::inferComparisonResult" in {
  let hasCanonicalizer = true in {
    def LEQPrimOp : IntBinaryPrimOp<"leq", UInt1Type>;
    def LTPrimOp  : IntBinaryPrimOp<"lt",  UInt1Type>;
    def GEQPrimOp : IntBinaryPrimOp<"geq", UInt1Type>;
    def GTPrimOp  : IntBinaryPrimOp<"gt",  UInt1Type>;
  }
  let hasCanonicalizeMethod = true in {
    def EQPrimOp  : IntBinaryPrimOp<"eq",  UInt1Type, [Commutative]>;
    def NEQPrimOp : IntBinaryPrimOp<"neq", UInt1Type, [Commutative]>;
  }
}

def CatPrimOp : IntBinaryPrimOp<"cat", UIntType> {
  let hasCanonicalizer = true;
}
def DShlPrimOp  : BinaryPrimOp<"dshl", IntType, UIntType, IntType>  {
  let description = [{
    A dynamic shift left operation. The width of `$result` is expanded to
    `width($lhs) + 1 << width($rhs) - 1`.
  }];
  let hasCanonicalizer = true;
}
def DShlwPrimOp : BinaryPrimOp<"dshlw", IntType, UIntType, IntType> {
  let description = [{
    A dynamic shift left operation same as 'dshl' but with different width rule.
    The width of `$result` is equal to `$lhs`.
  }];
}
def DShrPrimOp  : BinaryPrimOp<"dshr", IntType, UIntType, IntType> {
  let hasCanonicalizer = true;
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Base class for binary primitive operators.
class UnaryPrimOp<string mnemonic, Type srcType, Type resultType,
                  list<Trait> traits = []> : PrimOp<mnemonic, traits> {
  let arguments = (ins srcType:$input);
  let results = (outs resultType:$result);

  let assemblyFormat =
    "$input attr-dict `:` functional-type($input, $result)";

  // Give concrete operations a chance to set a type inference callback. If left
  // empty, a declaration for `inferUnaryReturnType` will be emitted that the
  // operation is expected to implement.
  code inferType = "";
  let inferTypeDecl = !if(!empty(inferType), [{
    /// Infer the return type of this unary operation.
    static FIRRTLType inferUnaryReturnType(FIRRTLType input,
                                           std::optional<Location> loc);
    }], "") # !subst("$_infer", !if(!empty(inferType), "inferUnaryReturnType",
      inferType), [{
    /// Infer the return type of this operation.
    static FIRRTLType inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      std::optional<Location> loc) {
      return $_infer(firrtl::type_cast<FIRRTLType>(operands[0].getType()), loc);
    }
  }]);

  let parseValidator = "impl::validateUnaryOpArguments";
}

def SizeOfIntrinsicOp : UnaryPrimOp<"int.sizeof", FIRRTLBaseType, UInt32Type>;

let hasCanonicalizer=1 in {
  def AsSIntPrimOp : UnaryPrimOp<"asSInt", FIRRTLBaseType, SIntType>;
  def AsUIntPrimOp : UnaryPrimOp<"asUInt", FIRRTLBaseType, UIntType>;
  def CvtPrimOp : UnaryPrimOp<"cvt", IntType, SIntType>;
}
def AsAsyncResetPrimOp
  : UnaryPrimOp<"asAsyncReset", OneBitCastableType, AsyncResetType>;
def AsClockPrimOp : UnaryPrimOp<"asClock", OneBitCastableType, ClockType>;
def NegPrimOp : UnaryPrimOp<"neg", IntType, SIntType>;
let hasCanonicalizer = true in
def NotPrimOp : UnaryPrimOp<"not", IntType, UIntType>;

let inferType = "impl::inferReductionResult" in {
  let hasCanonicalizer = 1 in {
    def AndRPrimOp : UnaryPrimOp<"andr", IntType, UInt1Type> {
      let description = [{
        Horizontally reduce a value to one bit, using the 'and' operation to merge
        bits.  `andr(x)` is equivalent to `concat(x, 1b1) == ~0`.  As such, it
        returns 1 for zero-bit-wide operands.
      }];
    }
    def OrRPrimOp : UnaryPrimOp<"orr", IntType, UInt1Type> {
      let description = [{
        Horizontally reduce a value to one bit, using the 'or' operation to merge
        bits.  `orr(x)` is equivalent to `concat(x, 1b0) != 0`.  As such, it
        returns 0 for zero-bit-wide operands.
      }];
    }
    def XorRPrimOp : UnaryPrimOp<"xorr", IntType, UInt1Type> {
      let description = [{
        Horizontally reduce a value to one bit, using the 'xor' operation to merge
        bits.  `xorr(x)` is equivalent to `popcount(concat(x, 1b0)) & 1`.  As
        such, it returns 0 for zero-bit-wide operands.
      }];
    }
  }
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

def BitsPrimOp : PrimOp<"bits"> {
  let arguments = (ins IntType:$input, I32Attr:$hi, I32Attr:$lo);
  let results = (outs UIntType:$result);

  let assemblyFormat =
    "$input $hi `to` $lo attr-dict `:` functional-type($input, $result)";

  let description = [{
    The `bits` operation extracts the bits between `hi` (inclusive) and `lo`
    (inclusive) from `input`.  `hi` must be greater than or equal to `lo`. Both
    `hi` and `lo` must be non-negative and less than the bit width of `input`.
    The result is `hi - lo + 1` bits wide.
  }];

  let hasCanonicalizer = true;
}

def HeadPrimOp : PrimOp<"head"> {
  let arguments = (ins IntType:$input, I32Attr:$amount);
  let results = (outs UIntType:$result);

  let assemblyFormat =
    "$input `,` $amount attr-dict `:` functional-type($input, $result)";

  let hasCanonicalizeMethod = true;
  let parseValidator = "impl::validateOneOperandOneConst";
}

def MuxPrimOp : PrimOp<"mux"> {
  let arguments = (ins UInt1OrUnsizedType:$sel, PassiveType:$high,
                       PassiveType:$low);
  let results = (outs PassiveType:$result);

  let assemblyFormat =
    "`(` operands `)` attr-dict `:` functional-type(operands, $result)";

  let hasCanonicalizer = true;
}

def PadPrimOp : PrimOp<"pad"> {
  let arguments = (ins IntType:$input, I32Attr:$amount);
  let results = (outs IntType:$result);

  let assemblyFormat =
    "$input `,` $amount attr-dict `:` functional-type($input, $result)";

  let description = [{
    Pad the input out to an `amount` wide integer, sign extending or zero
    extending according to `input`s type.  If `amount` is less than the existing
    width of `input`, then input is unmodified.
  }];

  let parseValidator = "impl::validateOneOperandOneConst";
}

class ShiftPrimOp<string mnemonic> : PrimOp<mnemonic> {
  let arguments = (ins IntType:$input, I32Attr:$amount);
  let results = (outs IntType:$result);

  let assemblyFormat =
    "$input `,` $amount attr-dict `:` functional-type($input, $result)";

  let parseValidator = "impl::validateOneOperandOneConst";
}

def ShlPrimOp : ShiftPrimOp<"shl"> {
  let description = [{
    The `shl` operation concatenates `amount` zero bits to the least significant
    end of `input`. `amount` must be non-negative.
  }];
}

def ShrPrimOp : ShiftPrimOp<"shr"> {
  let description = [{
    The `shr` operation truncates least significant `amount` bits from `input`.
    If `amount` is greater than of equal to `width(input)`, the value will be
    zero for unsigned types and the sign bit for signed types. `amount` must be
    non-negative.
  }];

  let hasCanonicalizeMethod = true;
}

def TailPrimOp : PrimOp<"tail"> {
  let arguments = (ins IntType:$input, I32Attr:$amount);
  let results = (outs UIntType:$result);

  let assemblyFormat =
    "$input `,` $amount attr-dict `:` functional-type($input, $result)";

  let description = [{
    The `tail` operation truncates the `amount` most significant bits from
    `input`.  `amount` must be non-negative and less than or equal to the bit
    width of e. The result is `width(input)-amount` bits wide.
  }];

  let hasCanonicalizeMethod = true;
  let parseValidator = "impl::validateOneOperandOneConst";
}

//===----------------------------------------------------------------------===//
// Target Specific Intrinsics
//===----------------------------------------------------------------------===//

def Mux2CellIntrinsicOp : PrimOp<"int.mux2cell"> {
  let summary = [{
    an intrinsic lowered into 2-to-1 MUX cell in synthesis tools.
  }];

  let description = [{
    This intrinsic exposes a low-level API to use 2-to-1 MUX cell in backend
    synthesis tool. At FIRRTL level, this operation participates
    the inference process in the same way as a normal mux operation.
  }];

  let arguments = (ins UInt1OrUnsizedType:$sel, PassiveType:$high,
                       PassiveType:$low);
  let results = (outs PassiveType:$result);

  let assemblyFormat =
    "`(` operands `)` attr-dict `:` functional-type(operands, $result)";
}

def Mux4CellIntrinsicOp : PrimOp<"int.mux4cell"> {
  let summary = [{
    an intrinsic lowered into 4-to-1 MUX cell in synthesis tools.
  }];

  let description = [{
    This intrinsic exposes a low-level API to use 4-to-1 MUX cell in backend
    synthesis tool. At FIRRTL level, this operation participates
    the inference process as a sugar of mux operation chains.
  }];

  let arguments = (ins UInt2OrUnsizedType:$sel, PassiveType:$v3,
                       PassiveType:$v2, PassiveType:$v1,
                       PassiveType:$v0);
  let results = (outs PassiveType:$result);

  let assemblyFormat =
    "`(` operands `)` attr-dict `:` functional-type(operands, $result)";
}

//===----------------------------------------------------------------------===//
// Verif and SV specific
//===----------------------------------------------------------------------===//

def IsXIntrinsicOp : FIRRTLOp<"int.isX",
    [HasCustomSSAName, Pure]> {
  let summary = "Test for 'x";
  let description = [{
    The `int.isX` expression checks that the operand is not a verilog literal
    'x.  FIRRTL doesn't have a notion of 'x per-se, but x can come in to the
    system from external modules and from SV constructs.  Verification
    constructs need to explicitly test for 'x.
    }];

  let arguments = (ins FIRRTLBaseType:$arg);
  let results = (outs NonConstUInt1Type:$result);
  let hasFolder = 1;
  let assemblyFormat = "$arg attr-dict `:` type($arg)";
}

def PlusArgsTestIntrinsicOp : FIRRTLOp<"int.plusargs.test",
    [HasCustomSSAName, Pure]> {
  let summary = "SystemVerilog `$test$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs NonConstUInt1Type:$found);
  let assemblyFormat = "$formatString attr-dict";
}

def PlusArgsValueIntrinsicOp : FIRRTLOp<"int.plusargs.value",
    [HasCustomSSAName, Pure]> {
  let summary = "SystemVerilog `$value$plusargs` call";

  let arguments = (ins StrAttr:$formatString);
  let results = (outs NonConstUInt1Type:$found, AnyType:$result);
  let assemblyFormat = "$formatString attr-dict `:` type($result)";
}

def HasBeenResetIntrinsicOp : FIRRTLOp<"int.has_been_reset", [Pure]> {
  let summary = "Check that a proper reset has been seen.";
  let description = [{
    The result of `firrtl.int.has_been_reset` reads as 0 immediately after simulation
    startup and after each power-cycle in a power-aware simulation. The result
    remains 0 before and during reset and only switches to 1 after the reset is
    deasserted again.

    See the corresponding `verif.has_been_reset` operation.
  }];
  let arguments = (ins NonConstClockType:$clock, AnyResetType:$reset);
  let results = (outs NonConstUInt1Type:$result);
  let hasFolder = 1;
  let assemblyFormat = "$clock `,` $reset attr-dict `:` type($reset)";
}

//===----------------------------------------------------------------------===//
// Verbatim
//===----------------------------------------------------------------------===//

def VerbatimExprOp : FIRRTLOp<"verbatim.expr",
                              [Pure, HasCustomSSAName]> {
  let summary = "Expression that expands to a value given SystemVerilog text";
  let description = [{
    This operation produces a typed value expressed by a string of
    SystemVerilog.  This can be used to access macros and other values that are
    only sensible as Verilog text.

    The text string is expected to have the highest precedence, so you should
    include parentheses in the string if it isn't a single token.  This is also
    assumed to not have side effects (use `sv.verbatim` if you need them).

    `firrtl.verbatim.expr` allows operand substitutions with `{{0}}` syntax.
  }];

  let arguments = (ins StrAttr:$text, Variadic<AnyType>:$substitutions,
                       DefaultValuedAttr<NameRefArrayAttr,"{}">:$symbols);
  let results = (outs FIRRTLBaseType:$result);
  let assemblyFormat = [{
    $text (`(` $substitutions^ `)`)?
    `:` functional-type($substitutions, $result) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Twine":$text,
                   CArg<"ValueRange", "{}">:$substitutions,
                   CArg<"ArrayRef<Attribute>", "{}">:$symbols), [{
      build(odsBuilder, odsState, resultType, odsBuilder.getStringAttr(text),
            substitutions, odsBuilder.getArrayAttr(symbols));
    }]>
  ];
}

def VerbatimWireOp : FIRRTLOp<"verbatim.wire",
                              [Pure, HasCustomSSAName]> {
  let summary = "Expression with wire semantics that expands to a value given "
                "SystemVerilog text";
  let description = [{
    This operation produces a typed value with wire semantics, expressed by a
    string of SystemVerilog. This can be used to access macros and other values
    that are only sensible as Verilog text.

    The text string is expected to have the highest precedence, so you should
    include parentheses in the string if it isn't a single token. This is also
    assumed to not have side effects (use `sv.verbatim` if you need them).

    `firrtl.verbatim.wire` allows operand substitutions with `{{0}}` syntax.
  }];

  let arguments = (ins StrAttr:$text, Variadic<AnyType>:$substitutions,
                       DefaultValuedAttr<NameRefArrayAttr,"{}">:$symbols);
  let results = (outs FIRRTLBaseType:$result);
  let assemblyFormat = [{
    $text (`(` $substitutions^ `)`)?
    `:` functional-type($substitutions, $result) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Twine":$text,
                   CArg<"ValueRange", "{}">:$substitutions,
                   CArg<"ArrayRef<Attribute>", "{}">:$symbols), [{
      build(odsBuilder, odsState, resultType, odsBuilder.getStringAttr(text),
            substitutions, odsBuilder.getArrayAttr(symbols));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Special inference Operations
//===----------------------------------------------------------------------===//

// This assumes operands are ground types without explicitly checking
class SameGroundTypeOperandConstness<string a, string b> 
  : PredOpTrait<
      "operand constness must match",
      CPred<"isConst($" # a # ".getType()) == isConst($" # b # ".getType())">>;

def UninferredResetCastOp : FIRRTLOp<"resetCast", [HasCustomSSAName, Pure, SameGroundTypeOperandConstness<"input", "result">]> {
  let description = [{
    Cast between reset types.  This is used to enable strictconnects early in
    the pipeline by isolating all uninferred reset connections to a single op.
  }];
  let arguments = (ins AnyResetType:$input);
  let results = (outs AnyResetType:$result);
  let hasFolder = 1;
  let assemblyFormat =
    "$input attr-dict `:` functional-type($input, $result)";
}

def ConstCastOp : FIRRTLOp<"constCast", [HasCustomSSAName, Pure]> {
  let description = [{
    Cast from a 'const' to a non-'const' type.
    ```
    %result = firrtl.constCast %in : (!firrtl.const.t1) -> !firrtl.t1
    ```
  }];
  let arguments = (ins PassiveType:$input);
  let results = (outs PassiveType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
    "$input attr-dict `:` functional-type($input, $result)";
}

//===----------------------------------------------------------------------===//
// Conversions to/from fixed-width signless integer types in standard dialect.
//===----------------------------------------------------------------------===//

// This operation converts from a struct to a bundle
// type, or visa-versa.  FIRRTL source/destination types must be passive.
def HWStructCastOp : FIRRTLOp<"hwStructCast", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input attr-dict `:` functional-type($input, $result)";

  let hasVerifier = 1;
}

def BitCastOp: FIRRTLOp<"bitcast", [Pure]> {
  let summary = [{
    Reinterpret one value to another value of the same size and
    potentially different type. This op is lowered to hw::BitCastOp.
  }];

  let arguments = (ins FIRRTLBaseType:$input);
  let results = (outs PassiveType:$result);
  let hasVerifier = 1;
  let hasFolder = 1;

  let assemblyFormat = "$input attr-dict `:` functional-type($input, $result)";
}

//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

def StringConstantOp : FIRRTLOp<"string", [Pure, ConstantLike]> {
  let summary = "Produce a constant string value";
  let description = [{
    Produces a constant value of string type.

    Example:
    ```mlir
    %0 = firrtl.string "hello world"
    ```
  }];
  let arguments = (ins StrAttr:$value);
  let results = (outs StringType:$result);
  let hasFolder = 1;
  let assemblyFormat = "$value attr-dict";
}

def FIntegerConstantOp : FIRRTLOp<"integer", [Pure, ConstantLike]> {
  let summary = "Produce a constant integer value";
  let description = [{
    Produces a constant value of integer type.

    Example:
    ```mlir
    %0 = firrtl.integer 42
    ```
  }];
  let arguments = (ins APSIntAttr:$value);
  let results = (outs FIntegerType:$result);
  let hasCustomAssemblyFormat = true;
}

//===----------------------------------------------------------------------===//
// RefOperations: Operations on the RefType.
//
// The RefType is used to capture dataflow paths.
// The operations on RefType support capturing the dataflow to/from SSA values.
// The following ops are supported on the RefType:
//
//  - RefSendOp
//    - Takes as input a `BaseType` and returns a `RefType`
//    - Used to send a read-only Reference handle of a local value.
//    - Represents the source of a dataflow by getting it's reference.
//  - RefResolveOp
//    - Takes as input a `RefType` and returns a `BaseType`
//    - Represents the sink of a dataflow from the reference handle
//    - The `RefType` is constrained to express a "single source to
//    one-or-more sink" dataflow
//    - Hence, multiple `RefResolveOp`s can be remotely connected to a single
//    `RefSendOp`.
//    - If the Reference ops are lowered to cross-module-reference, then it
//    represents the location of the XMR.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RefOps
//===----------------------------------------------------------------------===//

def RefCastOp : FIRRTLOp<"ref.cast",
    [HasCustomSSAName,
     Pure,
     CompatibleRefTypes<"result","input">]> {
  let summary = "Cast between compatible reference types";
  let description = [{
    Losslessly cast between compatible reference types.
    Source and destination must be recursively identical or destination
    has uninferred variants of the corresponding element in source.
    ```
      %result = firrtl.ref.cast %ref : (t1) -> t2
    ```
    }];

  let arguments = (ins RefType:$input);
  let results = (outs RefType:$result);

  let hasFolder = 1;

  let assemblyFormat =
     "$input attr-dict `:` functional-type($input, $result)";
}

def RefResolveOp: FIRRTLExprOp<"ref.resolve",
                              [RefTypeConstraint<"ref","result">]> {
  let summary = "FIRRTL Resolve a Reference";
  let description = [{
    Resolve a remote reference for reading a remote value.
    It takes a RefType input and returns the corresponding BaseType value.
    If an XMR is emitted for this reference, it will be at the location
    of this operation.
  }];
  let arguments = (ins RefType:$ref);
  let results = (outs FIRRTLBaseType:$result);

  let hasCanonicalizer = true;

  let assemblyFormat = "$ref attr-dict `:` qualified(type($ref))";
}

def RefSendOp: FIRRTLExprOp<"ref.send", [RefResultTypeConstraint<"base", "result">]> {
  let summary = "FIRRTL Send through Reference";
  let description = [{
    Endpoint of a remote reference. Send a value through a reference
    to be read from the firrtl.ref.resolve op.
    It takes a BaseType input and returns the corresponding RefType value.
  }];
  let arguments = (ins FIRRTLBaseType:$base);
  let results = (outs RefType:$result);

  let assemblyFormat = "$base attr-dict `:` qualified(type($base))";
}

def RefSubOp : FIRRTLExprOp<"ref.sub"> {
  let summary = "Extract an element of an aggregate RefType value";
  let description = [{
    The refsub expression statically refers, by index, to a sub-element
    of an expression with a RefType. The index must be a non-negative
    integer and cannot be equal to or exceed the underlying vector size
    or number of elements in bundle.
    ```
      %result = firrtl.ref.sub %input[index] : t1
    ```
    }];

  let arguments = (ins RefType:$input, I32Attr:$index);
  let results = (outs RefType:$result);

  let assemblyFormat =
     "$input `[` $index `]` attr-dict `:` qualified(type($input))";
}

def RWProbeOp : FIRRTLExprOp<"ref.rwprobe",
                  [DeclareOpInterfaceMethods<InnerRefUserOpInterface>]> {
  let summary = "FIRRTL RWProbe";
  let description = [{
    Create a RWProbe for the target.
    Target must be local.
    ```
      %result = firrtl.ref.rwprobe @mod::@sym : t
    ```
  }];

  let arguments = (ins InnerRefAttr:$target, TypeAttrOf<RWProbeTarget>:$type);
  let results = (outs RefType:$result);
  let assemblyFormat = "$target attr-dict `:` $type";
}

//===----------------------------------------------------------------------===//
// LTL Dialect Sequence and Property Intrinsics
//===----------------------------------------------------------------------===//

class LTLIntrinsicOp<string mnemonic, list<Trait> traits = []> :
    FIRRTLOp<"int.ltl." # mnemonic, traits # [Pure]> {
  let summary = "FIRRTL variant of `ltl." # mnemonic # "`";
  let description = "See `ltl." # mnemonic # "` op in the LTL dialect.";
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let results = (outs UInt1Type:$result);
}

class UnaryLTLIntrinsicOp<string mnemonic, list<Trait> traits = []> :
    LTLIntrinsicOp<mnemonic, traits> {
  let arguments = (ins UInt1Type:$input);
}

class BinaryLTLIntrinsicOp<string mnemonic, list<Trait> traits = []> :
    LTLIntrinsicOp<mnemonic, traits> {
  let arguments = (ins UInt1Type:$lhs, UInt1Type:$rhs);
}


// Generic
def LTLAndIntrinsicOp : BinaryLTLIntrinsicOp<"and", [Commutative]>;
def LTLOrIntrinsicOp : BinaryLTLIntrinsicOp<"or", [Commutative]>;

// Sequences
def LTLDelayIntrinsicOp : LTLIntrinsicOp<"delay"> {
  let arguments = (ins UInt1Type:$input,
                       I64Attr:$delay,
                       OptionalAttr<I64Attr>:$length);
  let assemblyFormat = [{
    $input `,` $delay (`,` $length^)? attr-dict `:`
    functional-type(operands, results)
  }];
}
def LTLConcatIntrinsicOp : BinaryLTLIntrinsicOp<"concat">;

// Properties
def LTLNotIntrinsicOp : UnaryLTLIntrinsicOp<"not">;
def LTLImplicationIntrinsicOp : BinaryLTLIntrinsicOp<"implication">;
def LTLEventuallyIntrinsicOp : UnaryLTLIntrinsicOp<"eventually">;

// Clocking
def LTLClockIntrinsicOp : LTLIntrinsicOp<"clock"> {
  let arguments = (ins UInt1Type:$input, ClockType:$clock);
  let assemblyFormat = [{
    $input `,` $clock attr-dict `:` functional-type(operands, results)
  }];
}
def LTLDisableIntrinsicOp : BinaryLTLIntrinsicOp<"disable">;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLEXPRESSIONS_TD
