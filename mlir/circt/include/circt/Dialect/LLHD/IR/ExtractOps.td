//===- ExtractOps.td - LLHD extract operations -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the LLHD ops for field and slice extractions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Constraint definitions
//===----------------------------------------------------------------------===//

class SigPtrIndexBitWidthConstraint<string index, string input>
  : TypesMatchWith<"Index width should be exactly clog2 (size of array)",
                   input, index,
                   [{
                     IntegerType::get($_self.getContext(), std::max<uint64_t>(
                       llvm::Log2_64_Ceil(llhd::getLLHDTypeWidth($_self)), 1))
                   }]>;

class SigArrayElementTypeConstraint<string result, string input>
  : TypesMatchWith<"result must be a signal of the array element type",
                   input, result,
                   "llhd::SigType::get(llhd::getLLHDElementType($_self))">;

class PtrArrayElementTypeConstraint<string result, string input>
  : TypesMatchWith<"result must be a pointer of the array element type",
                   input, result,
                   "llhd::PtrType::get(llhd::getLLHDElementType($_self))">;

class SameSigPtrArrayElementTypeConstraint<string result, string input>
  : PredOpTrait<"arrays element type must match",
                CPred<"llhd::getLLHDElementType($" # result # ".getType()) == "
                      # "llhd::getLLHDElementType($" # input # ".getType())">>;

class SmallerOrEqualResultTypeWidthConstraint<string result, string input>
  : PredOpTrait<"width of " # result # " type has to be smaller than or " #
                "equal to the " # input # " type",
                CPred<"llhd::getLLHDTypeWidth($" # result # ".getType()) <= " #
                      "llhd::getLLHDTypeWidth($" # input  # ".getType())">>;

//===----------------------------------------------------------------------===//
// Integer Operations
//===----------------------------------------------------------------------===//

def LLHD_SigExtractOp : LLHD_Op<"sig.extract",
    [Pure,
     SmallerOrEqualResultTypeWidthConstraint<"result", "input">,
     SigPtrIndexBitWidthConstraint<"lowBit", "input">]> {

  let summary = "Extract a range of bits from an integer signal";
  let description = [{
    The `llhd.sig.extract` operation allows to access a range of bits
    of the `$input` operand, starting at the index given by the `$lowBit`
    operand. The result length is defined by the result type.
  }];

  let arguments = (ins LLHD_SigType<[HWIntegerType]>:$input,
                       HWIntegerType:$lowBit);
  let results = (outs LLHD_SigType<[HWIntegerType]>: $result);

  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:` functional-type($input, $result)
  }];

  let extraClassDeclaration = [{
    unsigned getResultWidth()  { return getLLHDTypeWidth(getResult().getType()); }
    unsigned getInputWidth() { return getLLHDTypeWidth(getInput().getType()); }
  }];
  let hasFolder = true;
}

def LLHD_PtrExtractOp : LLHD_Op<"ptr.extract",
    [Pure,
     SmallerOrEqualResultTypeWidthConstraint<"result", "input">,
     SigPtrIndexBitWidthConstraint<"lowBit", "input">]> {

  let summary = "Extract a range of bits from an integer pointer";
  let description = [{
    The `llhd.ptr.extract` operation allows to access a range of bits
    of the `$input` operand, starting at the index given by the `$lowBit`
    operand. The result length is defined by the result type.
  }];

  let arguments = (ins LLHD_PtrType<[HWIntegerType]>:$input,
                       HWIntegerType:$lowBit);
  let results = (outs LLHD_PtrType<[HWIntegerType]>: $result);

  let assemblyFormat = [{
    $input `from` $lowBit attr-dict `:` functional-type($input, $result)
  }];

  let extraClassDeclaration = [{
    unsigned getResultWidth()  { return getLLHDTypeWidth(getResult().getType()); }
    unsigned getInputWidth() { return getLLHDTypeWidth(getInput().getType()); }
  }];
  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// Array Operations
//===----------------------------------------------------------------------===//

def LLHD_SigArraySliceOp : LLHD_Op<"sig.array_slice",
    [Pure,
     SmallerOrEqualResultTypeWidthConstraint<"result", "input">,
     SameSigPtrArrayElementTypeConstraint<"result", "input">,
     SigPtrIndexBitWidthConstraint<"lowIndex", "input">]> {

  let summary = "Get a range of consecutive values from a signal of an array";
  let description = [{
    The `llhd.sig.array_slice` operation allows to access a sub-range of the
    `$input` operand, starting at the index given by the `$lowIndex`
    operand. The resulting slice length is defined by the result type.
    Returns a signal aliasing the elements of the slice.

    Width of 'lowIndex' is defined to be the precise number of bits required to
    index the 'input' array. More precisely: for an input array of size M,
    the width of 'lowIndex' is ceil(log2(M)). Lower and upper bound indexes
    which are larger than the size of the 'input' array results in undefined
    behavior.

    Example:

    ```mlir
    %3 = llhd.sig.array_slice %input at %lowIndex :
      (!llhd.sig<!hw.array<4xi8>>) -> !llhd.sig<!hw.array<2xi8>>
    ```
  }];

  let arguments = (ins LLHD_SigType<[ArrayType]>: $input,
                       HWIntegerType: $lowIndex);
  let results = (outs LLHD_SigType<[ArrayType]>: $result);

  let assemblyFormat = [{
    $input `at` $lowIndex attr-dict `:` functional-type($input, $result)
  }];

  let extraClassDeclaration = [{
    unsigned getResultWidth()  { return getLLHDTypeWidth(getResult().getType()); }
    unsigned getInputWidth() { return getLLHDTypeWidth(getInput().getType()); }
    hw::ArrayType getInputArrayType() {
      return getInput().getType().cast<llhd::SigType>().getUnderlyingType()
        .cast<hw::ArrayType>();
    }
    hw::ArrayType getResultArrayType() {
      return getResult().getType().cast<llhd::SigType>().getUnderlyingType()
        .cast<hw::ArrayType>();
    }
  }];
  let hasFolder = true;
  let hasCanonicalizeMethod = true;
}

def LLHD_PtrArraySliceOp : LLHD_Op<"ptr.array_slice",
    [Pure,
     SmallerOrEqualResultTypeWidthConstraint<"result", "input">,
     SameSigPtrArrayElementTypeConstraint<"result", "input">,
     SigPtrIndexBitWidthConstraint<"lowIndex", "input">]> {

  let summary = "Get a range of consecutive values from a pointer to an array";
  let description = [{
    The `llhd.ptr.array_slice` operation allows to access a sub-range of the
    `$input` operand, starting at the index given by the `$lowIndex`
    operand. The resulting slice length is defined by the result type.
    Returns a pointer aliasing the elements of the slice.

    Width of 'lowIndex' is defined to be the precise number of bits required to
    index the 'input' array. More precisely: for an input array of size M,
    the width of 'lowIndex' is ceil(log2(M)). Lower and upper bound indexes
    which are larger than the size of the 'input' array results in undefined
    behavior.

    Example:

    ```mlir
    %3 = llhd.ptr.array_slice %input at %lowIndex :
      (!llhd.ptr<!hw.array<4xi8>>) -> !llhd.ptr<!hw.array<2xi8>>
    ```
  }];

  let arguments = (ins LLHD_PtrType<[ArrayType]>: $input,
                       HWIntegerType: $lowIndex);
  let results = (outs LLHD_PtrType<[ArrayType]>: $result);

  let assemblyFormat = [{
    $input `at` $lowIndex attr-dict `:` functional-type($input, $result)
  }];

  let extraClassDeclaration = [{
    unsigned getResultWidth()  { return getLLHDTypeWidth(getResult().getType()); }
    unsigned getInputWidth() { return getLLHDTypeWidth(getInput().getType()); }
    hw::ArrayType getInputArrayType() {
      return getInput().getType().cast<llhd::PtrType>().getUnderlyingType()
        .cast<hw::ArrayType>();
    }
    hw::ArrayType getResultArrayType() {
      return getResult().getType().cast<llhd::PtrType>().getUnderlyingType()
        .cast<hw::ArrayType>();
    }
  }];
  let hasFolder = true;
  let hasCanonicalizeMethod = true;
}

def LLHD_SigArrayGetOp : LLHD_Op<"sig.array_get",
    [Pure,
     SigPtrIndexBitWidthConstraint<"index", "input">,
     SigArrayElementTypeConstraint<"result", "input">]> {

  let summary = "Extract an element from a signal of an array.";
  let description = [{
    The `llhd.sig.array_get` operation allows to access the element of the
    `$input` operand at position `$index`. A new subsignal aliasing the element
    will be returned.

    Example:

    ```mlir
    // Returns a !llhd.sig<i8>
    %0 = llhd.sig.array_get %arr[%index] : !llhd.sig<!hw.array<4xi8>>
    ```
  }];

  let arguments = (ins LLHD_SigType<[ArrayType]>:$input, HWIntegerType:$index);
  let results = (outs LLHD_SigType<[HWNonInOutType]>: $result);

  let assemblyFormat = "$input `[` $index `]` attr-dict `:` qualified(type($input))";

  let extraClassDeclaration = [{
    hw::ArrayType getArrayType() {
      return getInput().getType().cast<llhd::SigType>()
      .getUnderlyingType().cast<hw::ArrayType>();
    }
  }];
}

def LLHD_PtrArrayGetOp : LLHD_Op<"ptr.array_get",
    [Pure,
     SigPtrIndexBitWidthConstraint<"index", "input">,
     PtrArrayElementTypeConstraint<"result", "input">]> {

  let summary = "Extract an element from a pointer to an array.";
  let description = [{
    The `llhd.ptr.array_get` operation allows to access the element of the
    `$input` operand at position `$index`. A new pointer aliasing the element
    will be returned.

    Example:

    ```mlir
    // Returns a !llhd.ptr<i8>
    %0 = llhd.ptr.array_get %arr[%index] : !llhd.ptr<!hw.array<4xi8>>
    ```
  }];

  let arguments = (ins LLHD_PtrType<[ArrayType]>:$input, HWIntegerType:$index);
  let results = (outs LLHD_PtrType<[HWNonInOutType]>: $result);

  let assemblyFormat = "$input `[` $index `]` attr-dict `:` qualified(type($input))";

  let extraClassDeclaration = [{
    hw::ArrayType getArrayType() {
      return getInput().getType().cast<llhd::PtrType>()
      .getUnderlyingType().cast<hw::ArrayType>();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Structure Operations
//===----------------------------------------------------------------------===//

def LLHD_SigStructExtractOp : LLHD_Op<"sig.struct_extract", [Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>, InferTypeOpInterface]> {
  let summary = "Extract a field from a signal of a struct.";
  let description = [{
    The `llhd.sig.struct_extract` operation allows access to the field of the
    `$input` operand given by its name via the `$field` attribute.
    A new subsignal aliasing the field will be returned.

    Example:

    ```mlir
    // Returns a !llhd.sig<i8>
    %0 = llhd.sig.struct_extract %struct["foo"]
      : !llhd.sig<!hw.struct<foo: i8, bar: i16>>
    ```
  }];

  let arguments = (ins LLHD_SigType<[StructType]>:$input, StrAttr:$field);
  let results = (outs LLHD_SigType<[HWNonInOutType]>:$result);

  let assemblyFormat = "$input `[` $field `]` attr-dict `:` qualified(type($input))";

  let extraClassDeclaration = [{
    hw::StructType getStructType() {
      return getInput().getType().cast<llhd::SigType>().getUnderlyingType()
        .cast<hw::StructType>();
    }
  }];
}

def LLHD_PtrStructExtractOp : LLHD_Op<"ptr.struct_extract", [Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>, InferTypeOpInterface]> {
  let summary = "Extract a field from a pointer to a struct.";
  let description = [{
    The `llhd.ptr.struct_extract` operation allows access to the field of the
    `$input` operand given by its name via the `$field` attribute.
    A new pointer aliasing the field will be returned.

    Example:

    ```mlir
    // Returns a !llhd.ptr<i8>
    %0 = llhd.ptr.struct_extract %struct["foo"]
      : !llhd.ptr<!hw.struct<foo: i8, bar: i16>>
    ```
  }];

  let arguments = (ins LLHD_PtrType<[StructType]>:$input, StrAttr:$field);
  let results = (outs LLHD_PtrType<[HWNonInOutType]>:$result);

  let assemblyFormat = "$input `[` $field `]` attr-dict `:` qualified(type($input))";

  let extraClassDeclaration = [{
    hw::StructType getStructType() {
      return getInput().getType().cast<llhd::PtrType>().getUnderlyingType()
        .cast<hw::StructType>();
    }
  }];
}
