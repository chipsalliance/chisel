//===- VerifOps.td - Verif dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VERIF_VERIFOPS_TD
#define CIRCT_DIALECT_VERIF_VERIFOPS_TD

include "circt/Dialect/Verif/VerifDialect.td"
include "circt/Dialect/LTL/LTLTypes.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VerifOp<string mnemonic, list<Trait> traits = []> :
  Op<VerifDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Assertions
//===----------------------------------------------------------------------===//

class AssertLikeOp<string mnemonic, list<Trait> traits = []> :
  VerifOp<mnemonic, traits> {
  let arguments = (ins LTLAnyPropertyType:$property,
                       OptionalAttr<StrAttr>:$label);
  let assemblyFormat = [{
    $property (`label` $label^)? attr-dict `:` type($property)
  }];
}

def AssertOp : AssertLikeOp<"assert"> {
  let summary = "Assert that a property holds.";
}

def AssumeOp : AssertLikeOp<"assume"> {
  let summary = "Assume that a property holds.";
}

def CoverOp : AssertLikeOp<"cover"> {
  let summary = "Ensure that a property can hold.";
}

//===----------------------------------------------------------------------===//
// Printing Formatted Messages
//===----------------------------------------------------------------------===//

def FormatVerilogStringOp : VerifOp<"format_verilog_string", [
    Pure
  ]> {
  let summary = "Creates a formatted string.";
  let description = [{
    Creates a formatted string suitable for printing via the `verif.print` op.
    The formatting syntax is expected to be identical to verilog string
    formatting to keep things simple for emission.
    If we in the future would like to be less tied to verilog formatting,
    please ask your friendly neighbourhood compiler engineer to e.g. implement
    a `FormatStringOp` which itself may lower to a `FormatVerilogStringOp`.
  }];
  let arguments = (ins
    StrAttr:$formatString,
    Variadic<AnyType>:$substitutions);
  
  let results = (outs HWStringType:$str);
  let assemblyFormat = [{
    $formatString `(` $substitutions `)` `:` type($substitutions) attr-dict
  }];
}

def PrintOp : VerifOp<"print", []> {
  let summary = "Prints a message.";
  let arguments = (ins HWStringType:$string);
  let assemblyFormat = [{
    $string attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Reset and Power-Cycling Detection
//===----------------------------------------------------------------------===//

def HasBeenResetOp : VerifOp<"has_been_reset", [Pure]> {
  let summary = "Check that a proper reset has been seen.";
  let description = [{
    The result of `verif.has_been_reset` reads as 0 immediately after simulation
    startup and after each power-cycle in a power-aware simulation. The result
    remains 0 before and during reset and only switches to 1 after the reset is
    deasserted again.

    This is a useful utility to disable the evaluation of assertions and other
    verification constructs in the IR before the circuit being tested has been
    properly reset. Verification failures due to uninitialized or randomized
    initial state can thus be prevented.

    Using the result of `verif.has_been_reset` to enable verification is more
    powerful and proper than just disabling verification during reset. The
    latter does not properly handle the period of time between simulation
    startup or power-cycling and the start of reset. `verif.has_been_reset` is
    guaranteed to produce a 0 value in that period, as well as during the reset.
  }];
  let arguments = (ins I1:$clock, I1:$reset, BoolAttr:$async);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $clock `,` custom<KeywordBool>($async, "\"async\"", "\"sync\"")
    $reset attr-dict
  }];
  let hasFolder = true;
}

#endif // CIRCT_DIALECT_VERIF_VERIFOPS_TD
