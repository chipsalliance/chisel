//===- OMOpInterfaces.td - Object Model dialect op interfaces -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Object Model dialect operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMOPINTERFACES_TD
#define CIRCT_DIALECT_OM_OMOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def ClassLike : OpInterface<"ClassLike"> {
  let cppNamespace = "circt::om";

  let description = [{
    Common functionality for class-like operations.
  }];

  let methods = [
    InterfaceMethod<"Get the class-like symbol name",
      "llvm::StringRef", "getSymName", (ins)>,
    InterfaceMethod<"Get the class-like symbol name attribute",
      "mlir::StringAttr", "getSymNameAttr", (ins)>,
    InterfaceMethod<"Get the class-like symbol name attribute name",
      "mlir::StringAttr", "getSymNameAttrName", (ins)>,
    InterfaceMethod<"Get the class-like formal parameter names attribute",
      "mlir::ArrayAttr", "getFormalParamNames", (ins)>,
    InterfaceMethod<"Get the class-like formal parameter names attribute name",
      "mlir::StringAttr", "getFormalParamNamesAttrName", (ins)>,
    InterfaceMethod<"Get the class-like body region",
      "mlir::Region &", "getBody", (ins)>,
    InterfaceMethod<"Get the class-like body block",
      "mlir::Block *", "getBodyBlock", (ins),
      /*methodBody=*/[{ return $_op.getBodyBlock(); }]>
  ];
}

def ClassFieldLike : OpInterface<"ClassFieldLike"> {
  let cppNamespace = "circt::om";

  let description = [{
    Common functionality for class-like field operations.
  }];

  let methods = [
    InterfaceMethod<"Get the class-like field's type",
      "mlir::Type", "getType", (ins)>
  ];
}

#endif // CIRCT_DIALECT_OM_OMOPINTERFACES_TD
