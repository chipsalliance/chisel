//===- ESIStdServices.td - ESI standard services ops -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/ESI/ESIServices.td"
include "circt/Dialect/HW/HWTypes.td"

def RandomAccessMemoryDeclOp: ESI_Op<"mem.ram",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">,
          Symbol, DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "Random access memory service";
  let description = [{
    Declares a service which is backed by a memory of some sort. Allows random
    access of the inner elements.

    Ports:
      read(address: clog2(depth)) -> data: innerType
      write({address: clog2(depth), data: innerType}) -> done: i0

    Users can ensure R/W ordering by waiting for the write "done" message before
    issuing a potentially dependant read. Ordering of R/W messages in flight is
    undefined.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$innerType, I64Attr:$depth);

  let assemblyFormat = [{
    $sym_name $innerType `x` $depth attr-dict
  }];
}
