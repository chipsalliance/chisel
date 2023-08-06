//===- SeqDialect.td - Seq dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the SeqDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQ_SEQDIALECT
#define CIRCT_DIALECT_SEQ_SEQDIALECT

def SeqDialect : Dialect {
  let name = "seq";

  let summary = "Types and operations for seq dialect";
  let description = [{
    The `seq` dialect is intended to model digital sequential logic.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::circt::seq";

  let extraClassDeclaration = [{
    /// Register all types.
    void registerTypes();
    /// Register all attributes.
    void registerAttributes();
  }];
}

#endif // CIRCT_DIALECT_SEQ_SEQDIALECT
