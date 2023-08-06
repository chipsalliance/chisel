//===- SeqTypes.td - Seq data type definitions -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Data types for the Seq dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/BuiltinTypeInterfaces.td"

class SeqType<string name, list<Trait> traits = []> : TypeDef<SeqDialect, name, traits> { }

def HLMemType : SeqType<"HLMem",[
    ShapedTypeInterface
  ]> {
  let summary = "Multi-dimensional memory type";
  let description = [{
    Syntax:
    ```
    hlmem-type ::== `hlmem` `<` dim-list element-type `>`
    ```

    The HLMemType represents the type of an addressable memory structure. The
    type is inherently multidimensional. Dimensions must be known integer values.

    Note: unidimensional memories are represented as <1x{element type}> - 
    <{element type}> is illegal.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let extraClassDeclaration = [{
    using ShapedType::Trait<HLMemType>::clone;
    using ShapedType::Trait<HLMemType>::getElementTypeBitWidth;
    using ShapedType::Trait<HLMemType>::getRank;
    using ShapedType::Trait<HLMemType>::getNumElements;
    using ShapedType::Trait<HLMemType>::isDynamicDim;
    using ShapedType::Trait<HLMemType>::hasStaticShape;
    using ShapedType::Trait<HLMemType>::getNumDynamicDims;
    using ShapedType::Trait<HLMemType>::getDimSize;
    using ShapedType::Trait<HLMemType>::getDynamicDimIndex;

    // Returns the set of types required to access this memory type.
    llvm::SmallVector<Type> getAddressTypes() const;

    /// Returns if this type is ranked (always true).
    bool hasRank() const { return true; }

    /// Clone this vector type with the given shape and element type. If the
    /// provided shape is `None`, the current shape of the type is used.
    HLMemType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                         Type elementType) const;
  }];

  let mnemonic = "hlmem";
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def FirMemType : SeqType<"FirMem"> {
  let summary = "A FIRRTL-flavored memory";
  let description = [{
    The `!seq.firmem` type represents a FIRRTL-flavored memory declared by a
    `seq.firmem` operation. It captures the parameters of the memory that are
    relevant to the read, write, and read-write ports, such as width and depth.
  }];
  let mnemonic = "firmem";
  let parameters = (ins
    "uint64_t":$depth,
    "uint32_t":$width,
    OptionalParameter<"std::optional<uint32_t>">:$maskWidth
  );
  let assemblyFormat = "`<` $depth `x` $width (`,` `mask` $maskWidth^)? `>`";
}
