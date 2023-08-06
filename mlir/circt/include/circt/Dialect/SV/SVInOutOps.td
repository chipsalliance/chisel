//===- SVInOutOps.td - SV ops pertaining to inout type -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the SystemVerilog dialect ops for working with inout types.
// These are wires, NoRegionArguments, etc.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Types.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Note that net declarations like 'wire' should not appear in an always block.
def WireOp : SVOp<"wire", [NonProceduralOp,
          DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
          DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>]> {
  let summary = "Define a new wire";
  let description = [{
    Declare a SystemVerilog Net Declaration of 'wire' type.
     See SV Spec 6.7, pp97.
    }];

  let arguments = (ins StrAttr:$name, OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs InOutType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType,
                   CArg<"StringAttr", "StringAttr()">:$name,
                   CArg<"hw::InnerSymAttr", "hw::InnerSymAttr()">:$innerSym)>,
    OpBuilder<(ins "::mlir::Type":$elementType, CArg<"StringRef">:$name), [{
      return build($_builder, $_state, elementType,
                   $_builder.getStringAttr(name));
    }]>
  ];

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) attr-dict
    `:` qualified(type($result))
  }];
  let hasCanonicalizeMethod = true;

  let extraClassDeclaration = [{
    Type getElementType() {
      return getResult().getType().cast<InOutType>().getElementType();
    }
  }];
}

def RegOp : SVOp<"reg", [
          DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
          DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>]> {
  let summary = "Define a new `reg` in SystemVerilog";
   let description = [{
     Declare a SystemVerilog Variable Declaration of 'reg' type.
     See SV Spec 6.8, pp100.
   }];
  let arguments = (ins StrAttr:$name, OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs InOutType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType,
                    CArg<"StringAttr", "StringAttr()">:$name,
                    CArg<"hw::InnerSymAttr", "hw::InnerSymAttr()">:$innerSym)>
  ];

  // We handle the name in a custom way, so we use a customer parser/printer.
  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) attr-dict
    `:` qualified(type($result))
  }];
  let hasCanonicalizeMethod = true;

  let extraClassDeclaration = [{
    Type getElementType() {
      return getResult().getType().cast<InOutType>().getElementType();
    }
  }];
}

def XMROp : SVOp<"xmr", []> {
  let summary = "Encode a reference to a non-local net.";
  let description = [{
    This represents a non-local hierarchical name to a net, sometimes called a
    cross-module reference.  A hierarchical name may be absolute, when prefixed
    with '$root', in which case it is resolved from the set of top-level modules
    (any non-instantiated modules).  Non-absolute paths are resolved by
    attempting resolution of the path locally, then recursively up the instance
    graph. See SV Spec 23.6, pp721.

    It is impossible to completely resolve a hierarchical name without making a
    closed-world assumption in the compiler.  We therefore don't try to link
    hierarchical names to what they resolve to at compile time.  A frontend
    generating this op should ensure that any instance or object in the intended
    path has public visibility so paths are not invalidated.
  }];
  let arguments = (ins UnitAttr:$isRooted, StrArrayAttr:$path, StrAttr:$terminal);
  let results = (outs InOutType:$result);
  let assemblyFormat = "(`isRooted` $isRooted^)? custom<XMRPath>($path, $terminal) attr-dict `:` qualified(type($result))";
}

def XMRRefOp : SVOp<"xmr.ref", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Encode a reference to something with a hw.hierpath.";
  let description = [{
    This represents a hierarchical path, but using something which the compiler
    can understand.  In contrast to the XMROp (which models pure Verilog
    hierarchical paths which may not map to anything knowable in the circuit),
    this op uses a `hw.hierpath` to refer to something which exists in the
    circuit.

    Generally, this operation is always preferred for situations where
    hierarchical paths cannot be known statically and may change.

    `verbatimSuffix` should only be populated when the final operation on the
    path is an instance of an external module.
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$ref,
    DefaultValuedAttr<StrAttr, "{}">:$verbatimSuffix
  );
  let results = (outs InOutType:$result);
  let assemblyFormat = [{
    $ref ( $verbatimSuffix^ )? attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    hw::HierPathOp getReferencedPath(const hw::HWSymbolCache *cache);
  }];
}

def ReadInOutOp
 : SVOp<"read_inout",
        [Pure, InOutElementConstraint<"result", "input">]> {
  let summary = "Get the value of from something of inout type (e.g. a wire or"
                " inout port) as the value itself.";
  let arguments = (ins InOutType:$input);
  let results = (outs HWValueType:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input))";
}

def InOutArrayType
  : Type<CPred<"getAnyHWArrayElementType(getInOutElementType($_self)) "
                        "!= Type()">,
         "an inout type with array element", "::circt::hw::InOutType">;

class InOutIndexConstraint<string value, string inoutValue>
  : TypesMatchWith<"type should be element of inout type",
                   inoutValue, value,
                   "InOutType::get(getAnyHWArrayElementType("
                                            "getInOutElementType($_self)))">;

def ArrayIndexInOutOp
 : SVOp<"array_index_inout",
        [Pure, InOutIndexConstraint<"result", "input">]> {
  let summary = "Index an inout memory to produce an inout element";
  let description = "See SV Spec 11.5.2.";
  let arguments = (ins InOutArrayType:$input, HWIntegerType:$index);
  let results = (outs InOutType:$result);
  let assemblyFormat =
    "$input`[`$index`]` attr-dict `:` qualified(type($input)) `,` qualified(type($index))";
}

def IndexedPartSelectInOutOp : SVOp<"indexed_part_select_inout",
                         [Pure, InferTypeOpInterface]> {
  let summary = "Address several contiguous bits of an inout type (e.g. a wire"
                " or inout port). This is an indexed part-select operator."
                "The base is an integer expression and the width is an "
                " integer constant. The bits start from base and the number "
                "of bits selected is equal to width. If $decrement is true, "
                " then part select decrements starting from $base."
                "See SV Spec 11.5.1.";

  let arguments = (ins InOutType:$input, HWIntegerType:$base, I32Attr:$width,
                      UnitAttr:$decrement);
  let results = (outs InOutType:$result);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];

  let hasFolder = true;
  let assemblyFormat = "$input`[`$base (`decrement` $decrement^)?`:` $width`]`"
                        " attr-dict `:` qualified(type($input)) `,` qualified(type($base))";
}

def InOutStructType
  : Type<CPred<"circt::hw::type_isa_and_nonnull<hw::StructType>(getInOutElementType($_self))">,
         "an inout type with struct field", "::circt::hw::InOutType">;

def StructFieldInOutOp : SVOp<"struct_field_inout",
                              [Pure, InferTypeOpInterface]> {
  let summary = "Create an subfield inout memory to produce an inout element.";
  let description = "See SV Spec 7.2.";

  let arguments = (ins InOutStructType:$input, StrAttr:$field);
  let results = (outs InOutType:$result);

  let assemblyFormat =
    "$input `[` $field `]` attr-dict `:` qualified(type($input))";

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];

}

def LogicOp : SVOp<"logic", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
      DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>]> {
  let summary = "Define a logic";
  let description = [{
    Declare a SystemVerilog Variable Declaration of 'logic' type.
    See SV Spec 6.8, pp100.
    }];

  let arguments = (ins StrAttr:$name, OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs InOutType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Type":$elementType,
                   CArg<"StringAttr", "StringAttr()">:$name,
                   CArg<"hw::InnerSymAttr", "hw::InnerSymAttr()">:$innerSym)>,
    OpBuilder<(ins "::mlir::Type":$elementType, CArg<"StringRef">:$name), [{
      return build($_builder, $_state, elementType,
                   $_builder.getStringAttr(name));
    }]>
  ];

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) attr-dict
    `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    Type getElementType() {
      return getResult().getType().cast<InOutType>().getElementType();
    }
  }];
}
