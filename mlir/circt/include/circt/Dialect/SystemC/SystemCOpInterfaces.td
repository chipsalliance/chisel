//===- SystemCOpInterfaces.td - Operation Interfaces -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the SystemC operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCOPINTERFACES
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCOPINTERFACES

include "mlir/IR/OpBase.td"

def SystemCNameDeclOpInterface : OpInterface<"SystemCNameDeclOpInterface"> {
  let cppNamespace = "circt::systemc";
  let description = [{
    Provide common information for operations that declare a SystemC variable
    name.
  }];

  let methods = [
    InterfaceMethod<"Get the declared name",
    "::llvm::StringRef", "getName", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getName(); }]>,

    InterfaceMethod<"Get the declared name",
    "::mlir::StringAttr", "getNameAttr", (ins),
    /*methodBody=*/[{}],
    /*defaultImplementation=*/[{ return $_op.getNameAttr(); }]>,
  ];

  let verify = [{
    assert($_op->hasAttrOfType<mlir::StringAttr>("name"));
    if ($_op->getAttrOfType<mlir::StringAttr>("name").getValue().empty())
      return $_op->emitOpError("'name' attribute must not be empty");
    return mlir::success();
  }];
}

#endif
