//===- OMPasses.td - OM dialect passes ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def LinkModules: Pass<"om-link-modules", "mlir::ModuleOp"> {
  let summary = "Link separated OM modules into a single module";
  let description = [{
    Flatten nested modules and resolve external classes.
  }];
  let constructor = "circt::om::createOMLinkModulesPass()";
}
