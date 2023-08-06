//===- Calyx.td - Calyx dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Calyx dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CALYX_TD
#define CALYX_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "circt/Dialect/Calyx/CalyxInterfaces.td"

def CalyxDialect : Dialect {
  let name = "calyx";
  let summary = "Types and operations for the Calyx dialect";
  let description = [{
    Calyx is an intermediate language and infrastructure for building
    compilers that generate custom hardware accelerators. For more
    information, visit the
    [documentation](https://capra.cs.cornell.edu/calyx/).
  }];
  let extraClassDeclaration = [{
    /// Register all Calyx types.
    void registerTypes();
    /// Register all Calyx attributes.
    void registerAttributes();
  }];
  // Depends on the HWDialect to support external primitives using hw.module.extern
  let dependentDialects = ["circt::hw::HWDialect"];
  let cppNamespace = "::circt::calyx";
}

class SameTypeConstraint<string lhs, string rhs>
 : TypesMatchWith<"lhs and rhs types should be equivalent",
                   lhs, rhs, [{ $_self }]>;

/// Base class for the operation in this dialect.
class CalyxOp<string mnemonic, list<Trait> traits = []> :
  Op<CalyxDialect, mnemonic, traits>;

/// Base class for Calyx cells.
class CalyxCell<string mnemonic, list<Trait> traits = []> :
  CalyxOp<mnemonic, !listconcat(traits, [
    Symbol,
    DeclareOpInterfaceMethods<CellOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ])> {}

/// Base class for Calyx containers.
class CalyxContainer<string mnemonic, list<Trait> traits = []> :
  CalyxOp<mnemonic, !listconcat(traits, [
    NoRegionArguments,
    NoTerminator,
    SingleBlock
  ])> {
  let assemblyFormat = "$body attr-dict";
  let regions = (region SizedRegion<1>: $body);
  
  code commonClassDeclaration = [{
    /// Returns the body of a Calyx container.
    Block *getBodyBlock() {
      Region* region = &getOperation()->getRegion(0);
      assert(region->hasOneBlock() && "The body should have one Block.");
      return &region->front();
    }
  }];
  let extraClassDeclaration = commonClassDeclaration;
}

/// Base class for ports associated with a Calyx GroupOp.
class CalyxGroupPort<string mnemonic, list<Trait> traits = []> :
  CalyxOp<mnemonic, !listconcat(traits, [
    HasParent<"GroupOp">
  ])> {
  let arguments = (ins
    I1:$src,
    Optional<I1>:$guard
  );
  let hasVerifier = 1;
}

include "circt/Dialect/Calyx/CalyxStructure.td"
include "circt/Dialect/Calyx/CalyxControl.td"
include "circt/Dialect/Calyx/CalyxPrimitives.td"

#endif // CALYX_TD
