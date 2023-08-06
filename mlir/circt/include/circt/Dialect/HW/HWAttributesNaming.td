//===- HWAttributesNaming.td - Attributes for HW dialect ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines HW dialect attributes used in other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWATTRIBUTESNAMING
#define CIRCT_DIALECT_HW_HWATTRIBUTESNAMING

include "circt/Dialect/HW/HWDialect.td"
include "mlir/IR/AttrTypeBase.td"

def InnerRefAttr : AttrDef<HWDialect, "InnerRef"> {
  let summary = "Refer to a name inside a module";
  let description = [{
    This works like a symbol reference, but to a name inside a module.
  }];
  let mnemonic = "innerNameRef";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$moduleRef,
                        "::mlir::StringAttr":$name);

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$module,
                                        "::mlir::StringAttr":$name), [{
      return $_get(
          module.getContext(), mlir::FlatSymbolRefAttr::get(module), name);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the InnerRefAttr for an operation and add the sym on it.
    static InnerRefAttr getFromOperation(mlir::Operation *op,
                                         mlir::StringAttr symName,
                                         mlir::StringAttr moduleName);

    /// Return the name of the referenced module.
    mlir::StringAttr getModule() const { return getModuleRef().getAttr(); }
  }];
}

def GlobalRefAttr : AttrDef<HWDialect, "GlobalRef"> {
  let summary = "Refer to a non-local symbol";
  let description = [{
    This works like a symbol reference, but to a global symbol with a possible
    unique instance path.
  }];
  let mnemonic = "globalNameRef";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$glblSym);
  let builders = [
    AttrBuilderWithInferredContext<(ins "::circt::hw::GlobalRefOp":$ref),[{
      return get(ref.getContext(), SymbolRefAttr::get(ref));
    }]>,
  ];

  let assemblyFormat = "`<` $glblSym `>`";

  let extraClassDeclaration = [{
    static constexpr char DialectAttrName[] = "circt.globalRef";
  }];
}

#endif // CIRCT_DIALECT_HW_HWATTRIBUTESNAMING
