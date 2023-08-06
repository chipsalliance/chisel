//===- IbisInterfaces.td - Ibis Interfaces ---------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the Ibis dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_INTERFACES_TD
#define CIRCT_DIALECT_IBIS_INTERFACES_TD

include "mlir/IR/OpBase.td"

def PortOpInterface : OpInterface<"PortOpInterface"> {
  let cppNamespace = "circt::ibis";
  let description =
      [{"An interface for operations which describe ports."}];

  let methods = [
  ];
}

#endif // CIRCT_DIALECT_IBIS_INTERFACES_TD
