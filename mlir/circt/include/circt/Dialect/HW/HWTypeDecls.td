//===- HWTypeDecls.td - HW data type declaration ops and types ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for type declarations in the HW type system.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWTYPEDECLS_TD
#define CIRCT_DIALECT_HW_HWTYPEDECLS_TD

include "circt/Dialect/HW/HWDialect.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Declaration operations
//===----------------------------------------------------------------------===//

def TypeScopeOp : HWOp<"type_scope",
      [Symbol, SymbolTable, SingleBlock, NoTerminator, NoRegionArguments]> {
  let summary = "Type declaration wrapper.";
  let description = [{
    An operation whose one body block contains type declarations. This op
    provides a scope for type declarations at the top level of an MLIR module.
    It is a symbol that may be looked up within the module, as well as a symbol
    table itself, so type declarations may be looked up.
  }];

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins SymbolNameAttr:$sym_name);

  let assemblyFormat = "$sym_name $body attr-dict";

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }
  }];
}

def TypedeclOp : HWOp<"typedecl", [Symbol, HasParent<"TypeScopeOp">]> {
  let summary = "Type declaration.";
  let description = "Associate a symbolic name with a type.";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    OptionalAttr<StrAttr>:$verilogName
  );

  let assemblyFormat = "$sym_name (`,` $verilogName^)? `:` $type attr-dict";

  let extraClassDeclaration = [{
    StringRef getPreferredName();

    // Returns the type alias type which this typedecl op defines.
    Type getAliasType();
  }];
}

#endif // CIRCT_DIALECT_HW_HWTYPEDECLS_TD
