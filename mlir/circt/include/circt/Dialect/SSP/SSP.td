//===- SSP.td - SSP dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the SSP (static scheduling problem) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SSP_TD
#define SSP_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def SSPDialect : Dialect {
  let name = "ssp";
  let summary = "Static scheduling problem instances and solutions.";
  let description = [{
    A dialect to abstractly represent instances and solutions of [static
    scheduling problems](https://circt.llvm.org/docs/Scheduling/), intended as
    an import/export format for testing, benchmarking, and rapid prototyping of
    problem definitions and algorithm implementations. See the
    [rationale](https://circt.llvm.org/docs/Dialects/SSP/RationaleSSP/) for more
    information.
  }];

  let cppNamespace = "::circt::ssp";

  let useDefaultAttributePrinterParser = true;
  let extraClassDeclaration = [{
    /// Register all SSP attributes.
    void registerAttributes();
  }];
}

include "circt/Dialect/SSP/SSPAttributes.td"
include "circt/Dialect/SSP/SSPOps.td"

#endif // SSP_TD
