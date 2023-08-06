//===- SeqAttributes.td - Attributes for Seq dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQATTRIBUTES_TD
#define CIRCT_DIALECT_SEQ_SEQATTRIBUTES_TD

include "circt/Dialect/Seq/SeqDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

let cppNamespace = "circt::seq" in {

// Read-under-write behavior
def RUW_Undefined : I32EnumAttrCase<"Undefined", 0, "undefined">;
def RUW_Old : I32EnumAttrCase<"Old", 1, "old">;
def RUW_New : I32EnumAttrCase<"New", 2, "new">;
def RUWAttr : I32EnumAttr<"RUW", "Read-Under-Write Behavior",
                          [RUW_Undefined, RUW_Old, RUW_New]>;

// Write-under-write behavior
def WUW_Undefined : I32EnumAttrCase<"Undefined", 0, "undefined">;
def WUW_PortOrder : I32EnumAttrCase<"PortOrder", 1, "port_order">;
def WUWAttr : I32EnumAttr<"WUW", "Write-Under-Write Behavior",
                          [WUW_Undefined, WUW_PortOrder]>;

}

/// An attribute holding information about memory initialization.
def FirMemInitAttr : AttrDef<SeqDialect, "FirMemInit"> {
  let mnemonic = "firmem.init";
  let summary = "Memory initialization information";
  let description = [{
    This attribute captures what the initial contents of a memory should be.
    At the moment this is modeled primarily with simulation in mind, where the
    memory contents are pre-loaded from a file at simulation startup.

    The `filename` specifies a file on disk that contains the initial contents
    for this memory. If `isBinary` is set, the file is interpreted as a binary
    file, otherwise it is treated as hexadecimal. This is modeled after the
    `$readmemh` and `$readmemb` SystemVerilog functions. If `isInline` is set,
    the initialization is emitted directly in the memory model; otherwise it is
    split out into a separate module that can be bound in.
  }];
  let parameters = (ins
    "mlir::StringAttr":$filename,
    "bool":$isBinary,
    "bool":$isInline
  );
  let assemblyFormat = "`<` $filename `,` $isBinary `,` $isInline `>`";
}

#endif // CIRCT_DIALECT_SEQ_SEQATTRIBUTES_TD
