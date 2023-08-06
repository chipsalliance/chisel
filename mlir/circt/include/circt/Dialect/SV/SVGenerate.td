//===- SVGenerate.td - SV generate block ops ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemVerilog generate blocks.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/BuiltinAttributeInterfaces.td"

def GenerateOp : SVOp<"generate",
                      [SingleBlock, NoTerminator, NoRegionArguments]> {
  let summary = "A generate block";
  let description = "See SystemVerilog 2017 27.";

  let arguments = (ins StrAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name attr-dict `:` $body
  }];
}

/// An array of case patterns. This is either a typed attribute of some kind,
/// an integer constant or a parameter expression for example, or a `unit` attr
/// that indicates the `default` case.
def PatternArrayAttr : TypedArrayAttrBase<
  AnyAttrOf<[TypedAttrInterface, UnitAttr]>, "case pattern array">;

def GenerateCaseOp : SVOp<"generate.case",
                          [SingleBlock, NoTerminator, NoRegionArguments,
                           HasParent<"GenerateOp">]> {

  let summary = "A 'case' statement inside of a generate block";
  let description = "See SystemVerilog 2017 27.5.";

  let regions = (region VariadicRegion<SizedRegion<1>>:$caseRegions);
  let arguments = (ins TypedAttrInterface:$cond,
                       PatternArrayAttr:$casePatterns, StrArrayAttr:$caseNames);

  let assemblyFormat = [{
    $cond attr-dict ` ` `[`
      custom<CaseRegions>($casePatterns, $caseNames, $caseRegions)
    `]`
  }];

  let hasVerifier = true;
}
