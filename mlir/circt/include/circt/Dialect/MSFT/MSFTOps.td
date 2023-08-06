//===- MSFTOps.td - MSFT operations definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"

def InstanceOp : MSFTOp<"instance", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
        DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        DeclareOpInterfaceMethods<HWInstanceLike>
    ]> {
  let summary = "Instantiate a module";

  let arguments = (ins InnerSymAttr:$inner_sym,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnyType>:$inputs,
                       OptionalAttr<ParamDeclArrayAttr>:$parameters,
                       OptionalAttr<SymbolRefAttr>:$targetDesignPartition);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "ArrayRef<Type>":$resultTypes, "StringAttr":$symName,
                   "FlatSymbolRefAttr":$moduleName, "ArrayRef<Value>":$inputs)>
  ];

  let extraClassDeclaration = [{
    // Return the name of the specified result or empty string if it cannot be
    // determined.
    StringAttr getResultName(size_t i);

    /// Check that the operands and results match the module specified.
    LogicalResult verifySignatureMatch(const circt::hw::ModulePortInfo&);

    // Update the results.
    InstanceOp getWithNewResults(MSFTModuleOp mod,
                                 ArrayRef<unsigned> newToOldMap);

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();
  }];

  /// sym keyword for optional symbol simplifies parsing
  let assemblyFormat = [{
    $inner_sym $moduleName `(` $inputs `)` custom<ParameterList>($parameters)
      attr-dict `:` functional-type($inputs, results)
  }];
}

def OneOrNoBlocksRegion : Region<
  CPred<"::llvm::hasNItemsOrLess($_self, 1)">,
  "region with at most 1 block">;

def AppIDArrayAttr : TypedArrayAttrBase<AppIDAttr, "Array of AppIDs">;

class MSFTModuleOpBase<string mnemonic, list<Trait> traits = []> :
    MSFTOp<mnemonic, traits # [
      IsolatedFromAbove, Symbol, FunctionOpInterface, OpAsmOpInterface,
      HasParent<"mlir::ModuleOp">, DeclareOpInterfaceMethods<HWModuleLike>
    ]> {
  /// Additional class declarations inside the module op.
  code extraModuleClassDeclaration = ?;

  let extraClassDeclaration = extraModuleClassDeclaration # [{

    // Get the module's symbolic name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the module's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError(
            "requires '" + getFunctionTypeAttrName().getValue() +
            "' attribute of function type");
      return success();
    }

    void getAsmBlockArgumentNames(Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
      if (region.empty())
        return;
      Block *block = getBodyBlock();
      ArrayAttr argNames = getArgNamesAttr();
      for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
        auto name = argNames[i].cast<StringAttr>().getValue();
        if (!name.empty())
          setNameFn(block->getArgument(i), name);
      }
    }

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();
  }];
}

def MSFTModuleOp : MSFTModuleOpBase<"module",
      [RegionKindInterface, HasParent<"mlir::ModuleOp">,
       SingleBlockImplicitTerminator<"OutputOp">,
       DeclareOpInterfaceMethods<HWMutableModuleLike>]>{
  let summary = "MSFT HW Module";
  let description = [{
    A lot like `hw.module`, but with a few differences:
      - Can exist without a body. The body is filled in by a generator post op
      creation.
      - Provides methods for mutation.
  }];
  let arguments = (ins
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs,
      StrArrayAttr:$argNames,
      StrArrayAttr:$resultNames,
      LocationArrayAttr:$argLocs,
      LocationArrayAttr:$resultLocs,
      DictionaryAttr:$parameters,
      OptionalAttr<StrAttr>:$fileName,
      OptionalAttr<StrArrayAttr>:$childAppIDBases);
  let results = (outs);
  let regions = (region OneOrNoBlocksRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "hw::ModulePortInfo":$ports,
                   "ArrayRef<NamedAttribute>":$params)>
  ];

  let extraModuleClassDeclaration = [{
    using mlir::detail::FunctionOpInterfaceTrait<MSFTModuleOp>::front;
    using mlir::detail::FunctionOpInterfaceTrait<MSFTModuleOp>::getFunctionBody;

    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) {
      return RegionKind::Graph;
    }

    // Insert and remove ports.
    void modifyPorts(
      ArrayRef<std::pair<unsigned, circt::hw::PortInfo>> insertInputs,
      ArrayRef<std::pair<unsigned, circt::hw::PortInfo>> insertOutputs,
      ArrayRef<unsigned> eraseInputs, ArrayRef<unsigned> eraseOutputs);

    // Adds input and output ports. Returns a list of new block arguments for
    // the new inputs.
    SmallVector<BlockArgument> addPorts(
      ArrayRef<std::pair<StringAttr, Type>> inputs,
      ArrayRef<std::pair<StringAttr, Value>> outputs);

    // Remove the ports at the specified indexes. Returns the new to old result
    // mapping.
    SmallVector<unsigned>
    removePorts(llvm::BitVector inputs, llvm::BitVector outputs);

    Block *getBodyBlock() { return &getFunctionBody().front(); }
  }];

  let extraClassDefinition = [{
    size_t $cppClass::getNumPorts() {
      // This is excessively expensive
      auto ports = getPortList();
      return ports.size();
    }

    circt::hw::InnerSymAttr $cppClass::getPortSymbolAttr(size_t portIndex) {
      for (NamedAttribute argAttr :
           mlir::function_interface_impl::getArgAttrs(*this, portIndex))
        if (auto sym = argAttr.getValue().dyn_cast<circt::hw::InnerSymAttr>())
          return sym;
      return circt::hw::InnerSymAttr();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def MSFTModuleExternOp : MSFTOp<"module.extern",
      [Symbol, FunctionOpInterface, HasParent<"mlir::ModuleOp">, DeclareOpInterfaceMethods<HWModuleLike>]> {
  let summary = "MSFT external Module";
  let description = [{
    Identical to `hw.module.extern`, and trivially lowers to that. This op
    exists so that we can use `msft.instance` to refer to both `msft.module` and
    `msft.module.extern`, rather than mixing `hw.instance` with `msft.instance`.
  }];
  let arguments = (ins TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
                       ParamDeclArrayAttr:$parameters,
                       OptionalAttr<StrAttr>:$verilogName);
  let regions = (region SizedRegion<0>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "const hw::ModulePortInfo &":$ports,
                   CArg<"StringRef", "StringRef()">:$verilogName,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();
  }];
}

def DesignPartitionOp : MSFTOp<"partition",
                               [Symbol, HasParent<"msft::MSFTModuleOp">]> {
  let summary = "A target \"module\" for moving entities";
  let description = [{
    Sometimes EDA tools require designs to have a module hierarchy which doesn't
    match the logical structure a designer would like to have. "Design
    partitions" allow the designer to "tag" entities (instances, registers, etc.)
    with a target design partition. During lowering, CIRCT will modify the
    hierarchy to move the tagged entities into the design partition module. The
    target design partition can then be used by subsequent EDA tools.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$verilogName);
  let assemblyFormat = "$sym_name `,` $verilogName attr-dict";
}

def OutputOp : MSFTOp<"output", [Terminator, ParentOneOf<["MSFTModuleOp", "LinearOp"]>,
                                Pure, ReturnLike]> {
  let summary = "termination operation";

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [
    OpBuilder<(ins)>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` qualified(type($operands)))?";
}
