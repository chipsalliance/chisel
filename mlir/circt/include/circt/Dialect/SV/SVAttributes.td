//===- SVAttributes.td - Attributes for SV dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SV dialect specific attributes.
//
//===----------------------------------------------------------------------===//

def MacroIdentAttr : AttrDef<SVDialect, "MacroIdent"> {
  let summary = "Macro identifier";
  let description = [{
    Represents a reference to a macro identifier.
  }];
  let parameters = (ins "::mlir::StringAttr":$ident);
  let mnemonic = "macro.ident";

  let assemblyFormat = "$ident";

  let builders = [
    AttrBuilder<(ins "::llvm::StringRef":$ident), [{
      return $_get($_ctxt, ::mlir::StringAttr::get($_ctxt, ident));
    }]>,
  ];
  let extraClassDeclaration = [{
    ::llvm::StringRef getName() { return getIdent().getValue(); }
  }];
}

def SVAttributeAttr : AttrDef<SVDialect, "SVAttribute"> {
  let summary = "a Verilog Attribute";
  let mnemonic = "attribute";
  let description = [{
    This attribute is used to encode a Verilog _attribute_.  A Verilog attribute
    (not to be confused with an LLVM or MLIR attribute) is a syntactic mechanism
    for adding metadata to specific declarations, statements, and expressions in
    the Verilog language.  _There are no "standard" attributes_.  Specific tools
    define and handle their own attributes.

    Verilog attributes have a mandatory name and an optional constant
    expression.  This is encoded as a key (name) value (expression) pair.
    Multiple attributes may be specified, either with multiple separate
    attributes or by comman-separating name--expression pairs.

    Currently, SV attributes don't block most optimizations; therefore, users should
    not expect that sv attributes always appear in the output verilog.
    However, we must block optimizations that updating ops in-place
    since it is mostly invalid to transfer SV attributes one to another.

    For more information, refer to Section 5.12 of the SystemVerilog (1800-2017)
    specification.
  }];
  let parameters = (ins
    "::mlir::StringAttr":$name,
    OptionalParameter<"::mlir::StringAttr">:$expression,
    DefaultValuedParameter<
      "mlir::BoolAttr", "mlir::BoolAttr::get($_ctxt, false)">:$emitAsComment
  );

  let builders = [
    AttrBuilder<(ins "::llvm::StringRef":$name,
                     CArg<"bool", "false">: $emitAsComment), [{
      return get(context, StringAttr::get(context, name), StringAttr(),
                 BoolAttr::get(context, emitAsComment));
    }]>,
    AttrBuilder<(ins "::mlir::StringAttr":$name,
                     CArg<"bool", "false">: $emitAsComment), [{
      return get(context, name, StringAttr(),
                 BoolAttr::get(context, emitAsComment));
    }]>,
    AttrBuilder<(ins "::llvm::StringRef":$name,
                     "::llvm::StringRef":$expression,
                     CArg<"bool", "false">: $emitAsComment), [{
      return get(context, StringAttr::get(context, name),
                 StringAttr::get(context, expression),
                 BoolAttr::get(context, emitAsComment));
    }]>,
    AttrBuilder<(ins "::mlir::StringAttr":$name,
                     "::mlir::StringAttr":$expression,
                     CArg<"bool", "false">: $emitAsComment), [{
      return get(context, name, expression,
                 BoolAttr::get(context, emitAsComment));
    }]>
  ];
  let hasCustomAssemblyFormat = true;
  let extraClassDeclaration = [{
    static inline llvm::StringRef
        getSVAttributesAttrName() { return "sv.attributes"; }
  }];
}
