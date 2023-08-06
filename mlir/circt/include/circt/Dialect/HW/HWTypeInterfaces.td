//===- HWTypeInterfaces.td - HW Type Interfaces ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the type interfaces of the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWTYPEINTERFACES_TD
#define CIRCT_DIALECT_HW_HWTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def FieldIDTypeInterface : TypeInterface<"FieldIDTypeInterface"> {
  let cppNamespace = "circt::hw";
  let description = [{
    Common methods for types which can be indexed by a FieldID.
  }];

  let methods = [
    InterfaceMethod<"Get the maximum field ID for this type",
      "uint64_t", "getMaxFieldID">,

    InterfaceMethod<[{
      Get the sub-type of a type for a field ID, and the subfield's ID. Strip
      off a single layer of this type and return the sub-type and a field ID
      targeting the same field, but rebased on the sub-type.
    }], "std::pair<circt::hw::FieldIDTypeInterface, uint64_t>",
    "getSubTypeByFieldID", (ins "uint64_t":$fieldID)>,

    InterfaceMethod<[{
      Return the final type targeted by this field ID by recursively walking all
      nested aggregate types. This is the identity function for ground types.
    }], "circt::hw::FieldIDTypeInterface", "getFinalTypeByFieldID",
    (ins "uint64_t":$fieldID)>,

    InterfaceMethod<[{
      Returns the effective field id when treating the index field as the
      root of the type.  Essentially maps a fieldID to a fieldID after a
      subfield op. Returns the new id and whether the id is in the given
      child.
    }], "std::pair<uint64_t, bool>", "rootChildFieldID",
    (ins "uint64_t":$fieldID, "uint64_t":$index)>,
  ];
}

#endif // CIRCT_DIALECT_HW_HWTYPEINTERFACES_TD
