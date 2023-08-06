//===- ESIStructure.td - ESI modules ---------------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI system and other ESI structural components.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_STRUCTURE_TD
#define CIRCT_DIALECT_ESI_STRUCTURE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "circt/Dialect/ESI/ESIDialect.td"
include "circt/Dialect/HW/HWOpInterfaces.td"

def ESIPureModuleOp : ESI_Op<"pure_module",
      [Symbol, NoTerminator, RegionKindInterface, NoRegionArguments,
       SingleBlock, HasParent<"mlir::ModuleOp">,
       DeclareOpInterfaceMethods<HWModuleLike>]> {
  let summary = "ESI pure module";
  let description = [{
    A module containing only ESI channels and modules with only ESI ports. All
    non-local connectivity is done through ESI services. If this module is the
    top level in the design, then the design's actual top level ports are
    defined by a BSP.

    Useful on its own for simulation and BSPs which don't define a top-level.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs); 
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{ $sym_name attr-dict-with-keyword $body }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph;}

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();
  }];
}

def ESIPureModuleInputOp : ESI_Op<"pure_module.input",
      [HasParent<"ESIPureModuleOp">]> {

  let summary = "Inputs become input ports when the module is lowered";
  let description = [{
    To create input ports when lowering a pure module op into an HWModuleOp, use
    this op. This op is typically created by a service implementation generator.

    If two 'input' ops exist in the same block, the names match, and the type
    matches they'll become one port during lowering. Two or more may not exist
    with the same name and different types. Useful for 'clk' and 'rst'.
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{ $name attr-dict `:` type($value) }];
}

def ESIPureModuleOutputOp : ESI_Op<"pure_module.output",
      [HasParent<"ESIPureModuleOp">]> {

  let summary = "Outputs become output ports when the module is lowered";
  let description = [{
    To create output ports when lowering a pure module op into an HWModuleOp, use
    this op. This op is typically created by a service implementation generator.

    Two 'output' ops with the same name cannot exist in the same block.
  }];

  let arguments = (ins StrAttr:$name, AnyType:$value);
  let results = (outs);
  let assemblyFormat = [{ $name `,` $value attr-dict `:` type($value) }];
}

def ESIPureModuleParamOp : ESI_Op<"pure_module.param",
      [HasParent<"ESIPureModuleOp">]> {

  let summary = "Params become module parameters when the module is lowered";
  let description = [{
    Allows attaching parameters to modules which become HW module parameters
    when lowering. Currently, they are ignored. Some low-level BSPs instantiate
    modules with parameters. This allows the modules produced to accept
    parameters so those BSPs can instantiate them.
  }];

  let arguments = (ins StrAttr:$name, TypeAttr:$type);
  let results = (outs);
  let assemblyFormat = [{ $name `:` $type attr-dict }];
}

#endif // CIRCT_DIALECT_ESI_STRUCTURE_TD
