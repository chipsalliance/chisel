//===- OMDialect.td - Object Model dialect definition ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Object Model dialect definition.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMDIALECT
#define CIRCT_DIALECT_OM_OMDIALECT

def OMDialect : Dialect {
  let name = "om";
  let cppNamespace = "::circt::om";

  let summary = [{
    The Object Model dialect captures design intent from various domains in an
    object model.
  }];

  let description = [{
    For more information about the Object Model dialect, see the
    [Object Model Dialect Rationale](./RationaleOM.md).
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let dependentDialects = [
    "circt::hw::HWDialect"
  ];

  let extraClassDeclaration = [{
    /// Register all OM types.
    void registerTypes();

    /// Register all OM types.
    void registerAttributes();
  }];
}

#endif // CIRCT_DIALECT_OM_OMDIALECT
