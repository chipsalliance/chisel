//===- FIRRTLStructure.td - Circuit and Module Ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the FIRRTL Circuit and Module MLIR ops.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLSTRUCTURE_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLSTRUCTURE_TD

include "FIRRTLAttributes.td"
include "FIRRTLDialect.td"
include "FIRRTLEnums.td"
include "FIRRTLOpInterfaces.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Types.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def CircuitOp : FIRRTLOp<"circuit",
      [IsolatedFromAbove, SymbolTable, SingleBlock, NoTerminator,
       NoRegionArguments, InnerRefNamespace]> {
  let summary = "FIRRTL Circuit";
  let description = [{
    The "firrtl.circuit" operation represents an overall Verilog circuit,
    containing a list of modules.
  }];
  let arguments = (ins StrAttr:$name,
                   DefaultValuedAttr<AnnotationArrayAttr, "{}">:$annotations);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name,
               CArg<"ArrayAttr","ArrayAttr()">:$annotations)>
  ];

  let extraClassDeclaration = [{
    OpBuilder getBodyBuilder() {
      assert(!getBody().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = getBody().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    /// Return body of this circuit.
    Block *getBodyBlock();

    // Return the main module that is the entry point of the circuit.  This may
    // be either an FModuleOp or an FExtModuleOp.  A SymbolTable is optional.
    FModuleLike getMainModule(mlir::SymbolTable* symtbl = nullptr);
  }];

  let assemblyFormat = "$name `` custom<CircuitOpAttrs>(attr-dict) $body";
  let hasRegionVerifier = 1;
}

class FIRRTLModuleLike<string mnemonic, list<Trait> traits = []> :
  FIRRTLOp<mnemonic, traits # [
    IsolatedFromAbove, Symbol, HasParent<"CircuitOp">,
    DeclareOpInterfaceMethods<FModuleLike>,
    DeclareOpInterfaceMethods<HWModuleLike>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    OpAsmOpInterface, InnerSymbolTable]> {

  /// Additional class definitions inside the module op.
  code extraModuleClassDefinition = [{}];
  code extraModuleClassDeclaration = [{}];

  let extraClassDeclaration = extraModuleClassDeclaration # [{

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();
  }];

  let extraClassDefinition = extraModuleClassDefinition # [{
    size_t $cppClass::getNumPorts() {
      return getPortTypesAttr().size();
    }

    circt::hw::InnerSymAttr $cppClass::getPortSymbolAttr(size_t portIndex) {
      auto syms = getPortSymbols();
      if (syms.empty() ||
          syms[portIndex].template cast<hw::InnerSymAttr>().empty())
        return hw::InnerSymAttr();
      return syms[portIndex].template cast<hw::InnerSymAttr>();
    }
  }];

}


def FModuleOp : FIRRTLModuleLike<"module", [SingleBlock, NoTerminator]> {
  let summary = "FIRRTL Module";
  let description = [{
    The "firrtl.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments =
            (ins ConventionAttr:$convention,
                 DefaultValuedAttr<AnnotationArrayAttr, "{}">:$annotations);

  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ConventionAttr":$convention,
               "ArrayRef<PortInfo>":$ports,
               CArg<"ArrayAttr", "ArrayAttr()">:$annotations)>,
  ];

  let extraModuleClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }

    using iterator = Block::iterator;
    iterator begin() { return getBodyBlock()->begin(); }
    iterator end() { return getBodyBlock()->end(); }

    Block::BlockArgListType getArguments() {
      return getBodyBlock()->getArguments();
    }

    // Return the block argument for the port with the specified index.
    BlockArgument getArgument(size_t portNumber);

    OpBuilder getBodyBuilder() {
      assert(!getBody().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = getBody().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
    ArrayAttr getParameters();
  }];

}

def FExtModuleOp : FIRRTLModuleLike<"extmodule"> {
  let summary = "FIRRTL external module";
  let description = [{
    The "firrtl.extmodule" operation represents an external reference to a
    Verilog module, including a given name and a list of ports.
    LowerAnnotations can add RefType ports to it. Each RefType port
    must have a corresponding entry in the internalPaths attribute.
    The internalPaths attribute is used to represent opaque internal paths
    into the external module, to be used for generating XMRs. Each RefType
    port must be removed by LowerXMR pass.
  }];
  let arguments = (ins
                   OptionalAttr<StrAttr>:$defname,
                   ConventionAttr:$convention,
                   ParamDeclArrayAttr:$parameters,
                   DefaultValuedAttr<AnnotationArrayAttr,
                   "ArrayAttr()">:$annotations,
                   DefaultValuedAttr<ArrayAttr, "{}">:$internalPaths
                  );
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name,
                      "ConventionAttr":$convention,
                      "ArrayRef<PortInfo>":$ports,
                      CArg<"StringRef", "StringRef()">:$defnamAttr,
                      CArg<"ArrayAttr", "ArrayAttr()">:$annotations,
                      CArg<"ArrayAttr", "ArrayAttr()">:$parameters,
                      CArg<"ArrayAttr", "ArrayAttr()">:$internalPaths)>
  ];

  let extraModuleClassDeclaration = [{
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def FIntModuleOp : FIRRTLModuleLike<"intmodule"> {
  let summary = "FIRRTL intrinsic module";
  let description = [{
    The "firrtl.intmodule" operation represents a compiler intrinsic.
  }];
  let arguments = (ins
                   OptionalAttr<StrAttr>:$intrinsic,
                   ParamDeclArrayAttr:$parameters,
                   DefaultValuedAttr<AnnotationArrayAttr,
                   "ArrayAttr()">:$annotations,
                   DefaultValuedAttr<ArrayAttr, "{}">:$internalPaths
                  );
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name,
                      "ArrayRef<PortInfo>":$ports,
                      CArg<"StringRef", "StringRef()">:$intrinsicNameAttr,
                      CArg<"ArrayAttr", "ArrayAttr()">:$annotations,
                      CArg<"ArrayAttr", "ArrayAttr()">:$parameters,
                      CArg<"ArrayAttr", "ArrayAttr()">:$internalPaths)>
  ];

  let extraModuleClassDeclaration = [{
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def FMemModuleOp : FIRRTLModuleLike<"memmodule"> {
  let summary = "FIRRTL Generated Memory Module";
  let description = [{
    The "firrtl.memmodule" operation represents an external reference to a
    memory module. See the "firrtl.mem" op for a deeper explantation of the
    parameters.

    A "firrtl.mem" operation is typically lowered to this operation when they
    are not directly lowered to registers by the compiler.
  }];
  let arguments =
    (ins UI32Attr:$numReadPorts, UI32Attr:$numWritePorts,
         UI32Attr:$numReadWritePorts, UI32Attr:$dataWidth, UI32Attr:$maskBits,
         UI32Attr:$readLatency, UI32Attr:$writeLatency, UI64Attr:$depth,
         ArrayAttr:$extraPorts, AnnotationArrayAttr:$annotations);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports,
                   "uint32_t":$numReadPorts,  "uint32_t":$numWritePorts,
                   "uint32_t":$numReadWritePorts, "uint32_t":$dataWidth,
                   "uint32_t":$maskBits, "uint32_t":$readLatency,
                   "uint32_t":$writeLatency, "uint64_t":$depth,
                   CArg<"ArrayAttr", "ArrayAttr()">:$annotations)>
  ];

  let extraModuleClassDeclaration = [{
    /// Return true if this memory has a mask.
    bool isMasked() { return getMaskBits() > 1; }
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
    ArrayAttr getParameters();
  }];

  let hasCustomAssemblyFormat = 1;
}

def ClassOp : FIRRTLModuleLike<"class", [
  SingleBlock, NoTerminator,
  DeclareOpInterfaceMethods<FModuleLike, [
    // Class Ops do not support port annotations.
    // Override this method to return an empty array attr.
    "getPortAnnotationsAttr"]>
]> {
  let summary = "FIRRTL Class";
  let description = [{
    The "firrtl.class" operation defines a class of property-only objects,
    including a given name, a list of ports, and a body that represents the
    connections within the class.

    A class may only have property ports, and its body may only be ops that act
    on properties, such as propassign ops.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, APIntAttr:$portDirections,
                       ArrayRefAttr:$portNames, ArrayRefAttr:$portTypes,
                       ArrayRefAttr:$portSyms, ArrayRefAttr:$portLocations);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins
      "StringAttr":$name,
      "ArrayRef<PortInfo>":$ports)>];

  let extraModuleClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }

    using iterator = Block::iterator;
    iterator begin() { return getBodyBlock()->begin(); }
    iterator end() { return getBodyBlock()->end(); }

    Block::BlockArgListType getArguments() {
      return getBodyBlock()->getArguments();
    }

    // Return the block argument for the port with the specified index.
    BlockArgument getArgument(size_t portNumber);

    OpBuilder getBodyBuilder() {
      assert(!getBody().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = getBody().front();
      return OpBuilder::atBlockEnd(&bodyBlock);
    }

    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
    ArrayAttr getParameters();

    /// Verify that the given type agrees with this class.
    LogicalResult verifyType(
      ClassType type, function_ref<InFlightDiagnostic()> emitError);

    /// Get the type for instances of this class.
    ClassType getInstanceType();
  }];
}

def GroupDeclOp : FIRRTLOp<
  "declgroup",
  [IsolatedFromAbove, Symbol, SymbolTable, SingleBlock, NoTerminator,
   ParentOneOf<["firrtl::CircuitOp", "firrtl::GroupDeclOp"]>]
> {
  let summary = "A declaration of an optional group";
  let description = [{
    The `firrtl.declgroup` operation declares an optional group and a lowering
    convetion for that group.  Optional groups are a feature of FIRRTL that add
    verification or debugging code to an existing module at runtime.

    A `firrtl.declgroup` operation only declares the group and any groups nested
    under the current declaration.  Functionality is added to modules using the
    `firrtl.group` operation.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, GroupConventionAttr:$convention);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name $convention attr-dict-with-keyword $body
  }];
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLSTRUCTURE_TD
