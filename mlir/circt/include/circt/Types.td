//===- Types.td - Common type predicates and definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file collects type predicates and definitions commonly used in the CIRCT
// dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TYPES
#define CIRCT_TYPES

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def APSIntAttr : Attr<CPred<"$_self.isa<::mlir::IntegerAttr>()">,
                      "arbitrary integer attribute with sign"> {
  let storageType = [{ ::mlir::IntegerAttr }];
  let returnType = [{ ::llvm::APSInt }];
  let constBuilderCall = "IntegerAttr::get($_builder.getContext(), $0)";
  let convertFromStorage = "$_self.getAPSInt()";
}

// Compared to the standard ArrayAttr, the accessor to an ArrayRefAttr will
// convert the attribute to an ArrayRef, while an ArrayAttr accessor just
// returns itself.
def ArrayRefAttr :
  ArrayAttrBase<
    CPred<"::llvm::isa<::mlir::ArrayAttr>($_self)">, "array attribute"> {
  let returnType = [{ ::llvm::ArrayRef<Attribute> }];
  let convertFromStorage = [{ $_self.getValue() }];
}

#endif // CIRCT_TYPES
