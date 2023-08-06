//===- ESITypes.td - ESI type specifications ---------------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI extends the HW types by adding a List type.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_TYPES_TD
#define CIRCT_DIALECT_ESI_TYPES_TD

include "circt/Dialect/ESI/ESIDialect.td"

def ChannelType : DialectType<ESI_Dialect,
    CPred<"isa<::circt::esi::ChannelType>($_self)">,
    "an ESI channel",
    "::circt::esi::ChannelType">;

def ESIListType : ESI_Type<"List"> {
  let summary = "a runtime-variably sized list";
  let description = [{
    In software, a chunk of memory with runtime-specified length. In hardware, a
    stream of runtime-specified amount of data transmitted over many cycles in
    compile-time specified specified windows (chunks).
  }];

  let mnemonic = "list";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = [{
    `<` $elementType `>`
  }];
}

#endif // CIRCT_DIALECT_ESI_TYPES_TD
