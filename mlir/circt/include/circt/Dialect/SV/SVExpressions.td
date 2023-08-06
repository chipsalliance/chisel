//===- SVExpressions.td - SV expression ops ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemVerilog expressions.
//
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/InferTypeOpInterface.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/SV/SVTypes.td"

def HWValueOrInOutType : AnyTypeOf<[HWValueType, InOutType]>;
def HasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface,
                         ["getAsmResultNames"]>;


def SystemFunctionOp : SVOp<"system"> {
  let summary = "Simple SV System Function calls";
  let description = [{
    This operation calls the indicated system verilog system function.  This
    supports functions which take normal expression arguments.

    See section 20 of the 2012 SV spec.
  }];
  let arguments = (ins StrAttr:$fnName, Variadic<AnyType>:$args);
  let results = (outs AnyType:$out);

  let assemblyFormat = "$fnName `(` $args `)` attr-dict `:` functional-type($args, $out)";
}


def VerbatimExprOp : SVOp<"verbatim.expr", [Pure, HasCustomSSAName]> {
  let summary = "Expression that expands to a value given SystemVerilog text";
  let description = [{
    This operation produces a typed value expressed by a string of
    SystemVerilog.  This can be used to access macros and other values that are
    only sensible as Verilog text.

    The text string is expected to have the highest precedence, so you should
    include parentheses in the string if it isn't a single token.  This is also
    assumed to not have side effects (use sv.verbatim.expr.se) if you need them.

    sv.verbatim.expr allows operand substitutions with {{0}} syntax.
    }];

  let arguments = (ins StrAttr:$format_string, Variadic<AnyType>:$substitutions,
                  DefaultValuedAttr<NameRefArrayAttr,"{}">:$symbols);
  let results = (outs HWValueOrInOutType:$result);
  let assemblyFormat = [{
    $format_string (`(` $substitutions^ `)`)?
      `:` functional-type($substitutions, $result) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Twine":$formatString,
                CArg<"ValueRange", "{}">:$substitutions),
               "build(odsBuilder, odsState, resultType, "
               "odsBuilder.getStringAttr(formatString), substitutions,"
               "odsBuilder.getArrayAttr({}));">,
    OpBuilder<(ins "Type":$resultType, "StringAttr":$formatString,
                CArg<"ValueRange", "{}">:$substitutions),
               "build(odsBuilder, odsState, resultType, "
               "formatString, substitutions,"
               "odsBuilder.getArrayAttr({}));">
  ];
}

def VerbatimExprSEOp : SVOp<"verbatim.expr.se", [HasCustomSSAName]> {
  let summary = "Expression that expands to a value given SystemVerilog text";
  let description = [{
    This operation produces a typed value expressed by a string of
    SystemVerilog.  This can be used to access macros and other values that are
    only sensible as Verilog text.

    The text string is expected to have the highest precedence, so you should
    include parentheses in the string if it isn't a single token.  This is
    allowed to have side effects.

    sv.verbatim.se.expr allows operand substitutions with {{0}} syntax.
    }];

  let arguments = (ins StrAttr:$format_string, Variadic<AnyType>:$substitutions,
                 DefaultValuedAttr<NameRefArrayAttr,"{}">:$symbols );
  let results = (outs HWValueOrInOutType:$result);
  let assemblyFormat = [{
    $format_string (`(` $substitutions^ `)`)?
      `:` functional-type($substitutions, $result) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Twine":$formatString),
               "build(odsBuilder, odsState, resultType, "
               "odsBuilder.getStringAttr(formatString), ValueRange{},"
               "odsBuilder.getArrayAttr({}));">
  ];
}

def MacroRefExprOp : SVOp<"macro.ref", [Pure, HasCustomSSAName,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Expression to refer to a SystemVerilog macro";
  let description = [{
    This operation produces a value by referencing a named macro.

    Presently, it is assumed that the referenced macro is a constant with no
    side effects.  This expression is subject to CSE.  It can be duplicated
    and emitted inline by the Verilog emitter.
    }];

  let arguments = (ins FlatSymbolRefAttr:$macroName, Variadic<AnyType>:$inputs);
  let results = (outs HWValueOrInOutType:$result);

  let assemblyFormat = "$macroName `(` $inputs `)` attr-dict `:` functional-type($inputs, $result)";

  let builders = [
    OpBuilder<(ins "Type":$resultType, "StringRef":$ident),
               "build(odsBuilder, odsState, resultType, "
                "FlatSymbolRefAttr::get($_builder.getContext(), ident), {});">
  ];

  let extraClassDeclaration = [{
    MacroDeclOp getReferencedMacro(const hw::HWSymbolCache *cache);
  }];
}

def MacroRefExprSEOp : SVOp<"macro.ref.se", [HasCustomSSAName,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Expression to refer to a SystemVerilog macro";
  let description = [{
    This operation produces a value by referencing a named macro.

    Presently, it is assumed that the referenced macro is not constant and has
    side effects.  This expression is not subject to CSE.  It can not be
    duplicated, but can be emitted inline by the Verilog emitter.
    }];

  let arguments = (ins FlatSymbolRefAttr:$macroName, Variadic<AnyType>:$inputs);
  let results = (outs HWValueOrInOutType:$result);

  let assemblyFormat = "$macroName `(` $inputs `)` attr-dict `:` functional-type($inputs, $result)";

  let builders = [
    OpBuilder<(ins "Type":$resultType, "StringRef":$ident),
               "build(odsBuilder, odsState, resultType, "
                "FlatSymbolRefAttr::get($_builder.getContext(), ident), {});">
  ];

  let extraClassDeclaration = [{
    MacroDeclOp getReferencedMacro(const hw::HWSymbolCache *cache);
  }];
}

def ConstantXOp : SVOp<"constantX", [Pure, HasCustomSSAName]> {
  let summary = "A constant of value 'x'";
  let description = [{
    This operation produces a constant value of 'x'.  This 'x' follows the
    System Verilog rules for 'x' propagation.
    }];

  let arguments = (ins);
  let results = (outs HWValueType:$result);
  let assemblyFormat = " attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    int64_t getWidth() {
      return hw::getBitWidth(getType());
    }
  }];
}

def ConstantZOp : SVOp<"constantZ", [Pure, HasCustomSSAName]> {
  let summary = "A constant of value 'z'";
  let description = [{
    This operation produces a constant value of 'z'.  This 'z' follows the
    System Verilog rules for 'z' propagation.
    }];

  let arguments = (ins);
  let results = (outs HWValueType:$result);
  let assemblyFormat = " attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    int64_t getWidth() {
      return hw::getBitWidth(getType());
    }
  }];
}

def ConstantStrOp : SVOp<"constantStr", [Pure]> {
  let summary = "A constant of string value";
  let description = [{
    This operation produces a constant string literal.
    }];

  let arguments = (ins StrAttr:$str);
  let results = (outs HWStringType:$result);
  let assemblyFormat = " $str attr-dict";
}

def LocalParamOp : SVOp<"localparam",
      [FirstAttrDerivedResultType, Pure, HasCustomSSAName]> {
  let summary = "Declare a localparam";
  let description = [{
    The localparam operation produces a `localparam` declaration. See SV spec
    6.20.4 p125.
    }];

  let arguments = (ins AnyAttr:$value, StrAttr:$name);
  let results = (outs HWValueType:$result);

  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) attr-dict `:` qualified(type($result))
  }];

  let hasVerifier = 1;
}

def IndexedPartSelectOp
 : SVOp<"indexed_part_select",
        [Pure, InferTypeOpInterface]> {
  let summary = "Read several contiguous bits of an int type."
                "This is an indexed part-select operator."
                "The base is an integer expression and the width is an "
                " integer constant. The bits start from base and the number "
                "of bits selected is equal to width. If $decrement is true, "
                " then part select decrements starting from $base."
                "See SV Spec 11.5.1.";
  let arguments = (ins HWIntegerType:$input, HWIntegerType:$base, I32Attr:$width,
                      UnitAttr:$decrement);
  let results = (outs HWIntegerType:$result);

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

  let assemblyFormat = "$input`[`$base (`decrement` $decrement^)?`:` $width`]`"
                        " attr-dict `:` qualified(type($input)) `,` qualified(type($base))";
}

def SampledOp : SVOp<"system.sampled", [Pure, SameOperandsAndResultType]> {
  let summary = "`$sampled` system function to sample a value";
  let description = [{
    Sample a value using System Verilog sampling semantics (see Section 16.5.1
    of the SV 2017 specification for more information).

    A use of `$sampled` is to safely read the value of a net/variable in a
    concurrent assertion action block such that the value will be the same as
    the value used when the assertion is triggered.  See Section 16.9.3 of the
    SV 2017 specification for more information.
  }];
  let arguments = (ins AnyType:$expression);
  let results = (outs AnyType:$sampledValue);
  let assemblyFormat = "$expression attr-dict `:` qualified(type($expression))";
}
