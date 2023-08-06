//===- HWStructure.td - HW structure ops -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for structure.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWSTRUCTURE_TD
#define CIRCT_DIALECT_HW_HWSTRUCTURE_TD

include "circt/Dialect/HW/HWAttributes.td"
include "circt/Dialect/HW/HWDialect.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Base class factoring out some of the additional class declarations common to
/// the module-like operations.
class HWModuleOpBase<string mnemonic, list<Trait> traits = []> :
    HWOp<mnemonic, traits # [
      DeclareOpInterfaceMethods<PortList>,
      DeclareOpInterfaceMethods<HWModuleLike>,
      DeclareOpInterfaceMethods<HWMutableModuleLike>,
      FunctionOpInterface, Symbol,
      OpAsmOpInterface, HasParent<"mlir::ModuleOp">]> {
  /// Additional class declarations inside the module op.
  code extraModuleClassDeclaration = ?;

  let extraClassDeclaration = extraModuleClassDeclaration # [{
    /// Insert and remove input and output ports of this module. Does not modify
    /// the block arguments of the module body. The insertion and removal
    /// indices must be in ascending order. The indices refer to the port
    /// positions before any insertion or removal occurs. Ports inserted at the
    /// same index will appear in the module in the same order as they were
    /// listed in the insertion arrays.
    void modifyPorts(
      ArrayRef<std::pair<unsigned, PortInfo>> insertInputs,
      ArrayRef<std::pair<unsigned, PortInfo>> insertOutputs,
      ArrayRef<unsigned> eraseInputs,
      ArrayRef<unsigned> eraseOutputs
    );

    /// Attribute name for port symbols.
    static StringRef getPortSymbolAttrName() {
      return "hw.exportPort";
    }

    void setPortSymbolAttr(size_t portIndex, ::circt::hw::InnerSymAttr sym);
  }];

  /// Additional class definitions inside the module op.
  code extraModuleClassDefinition = [{}];

  let extraClassDefinition = extraModuleClassDefinition # [{
    size_t $cppClass::getNumPorts() {
      // This is wildly inefficient
      return getPortList().size();
    }

    ::circt::hw::InnerSymAttr $cppClass::getPortSymbolAttr(size_t portIndex) {
      DictionaryAttr portAttrs;
      if (portIndex < getNumInputs()) {
         portAttrs = ::mlir::function_interface_impl::getArgAttrDict(*this,
                                                                     portIndex);
      } else {
        portIndex -= getNumInputs();
        portAttrs = ::mlir::function_interface_impl::getResultAttrDict(
            *this, portIndex);
      }
      if (portAttrs)
        return portAttrs.getAs<::circt::hw::InnerSymAttr>(
          getPortSymbolAttrName());
      return {};
    }

    void $cppClass::setPortSymbolAttr(size_t portIndex, ::circt::hw::InnerSymAttr sym) {
      auto portSymAttr = StringAttr::get(getContext(), getPortSymbolAttrName());
      if (portIndex < getNumInputs()) {
        if (sym)
          ::mlir::function_interface_impl::setArgAttr(*this, portIndex,
                                                      portSymAttr, sym);
        else
          ::mlir::function_interface_impl::removeArgAttr(*this, portIndex,
                                                         portSymAttr);
        return;
      }
      if (sym)
        ::mlir::function_interface_impl::setResultAttr(
            *this, portIndex - getNumInputs(), portSymAttr, sym);
      else
        ::mlir::function_interface_impl::removeResultAttr(
            *this, portIndex - getNumInputs(), portSymAttr);
    }
  }];

}

def HWTestModuleOp : HWOp<"testmodule", [Symbol, PortList,
                                         OpAsmOpInterface, 
                                         HasParent<"mlir::ModuleOp">, 
                                         IsolatedFromAbove,
                                         SingleBlockImplicitTerminator<"OutputOp">]> {
  let arguments = (ins TypeAttrOf<ModuleType>:$module_type,
                       OptionalAttr<DictArrayAttr>:$port_attrs,
                       OptionalAttr<LocationArrayAttr>:$port_locs,
                       ParamDeclArrayAttr:$parameters,
                       OptionalAttr<StrAttr>:$comment);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);

    Block *getBodyBlock() { return &getRegion().front(); }

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();
  }];

}

def HWModuleOp : HWModuleOpBase<"module",
      [IsolatedFromAbove, RegionKindInterface,
       SingleBlockImplicitTerminator<"OutputOp">]>{
  let summary = "HW Module";
  let description = [{
    The "hw.module" operation represents a Verilog module, including a given
    name, a list of ports, a list of parameters, and a body that represents the
    connections within the module.
  }];
  let arguments = (ins TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       StrArrayAttr:$argNames,
                       StrArrayAttr:$resultNames,
                       LocationArrayAttr:$argLocs,
                       LocationArrayAttr:$resultLocs,
                       ParamDeclArrayAttr:$parameters,
                       StrAttr:$comment);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes,
                   CArg<"StringAttr", "{}">:$comment)>,
    OpBuilder<(ins "StringAttr":$name, "const ModulePortInfo &":$ports,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes,
                   CArg<"StringAttr", "{}">:$comment,
		   CArg<"bool", "true">:$shouldEnsureTerminator)>,
    OpBuilder<(ins "StringAttr":$name, "const ModulePortInfo &":$ports,
                   "HWModuleBuilder":$modBuilder,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes,
                   CArg<"StringAttr", "{}">:$comment)>
  ];

  let extraModuleClassDeclaration = [{
    using mlir::detail::FunctionOpInterfaceTrait<HWModuleOp>::front;
    using mlir::detail::FunctionOpInterfaceTrait<HWModuleOp>::getFunctionBody;

    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph;}

    /// Returns the number of in or inout ports.
    size_t getNumInOrInoutPorts() { return getArgumentTypes().size(); }

    /// Return the PortInfo for the specified input or inout port.
    PortInfo getInOrInoutPort(size_t i) {
      return getModuleInOrInoutPort(*this, i);
    }

    /// Return the PortInfo for the specified output port.
    PortInfo getOutputPort(size_t i) {
      return getModuleOutputPort(*this, i);
    }

    /// Append an input with a given name and type to the port list.
    /// If the name is not unique, a unique name is created and returned.
    std::pair<StringAttr, BlockArgument>
    appendInput(const Twine &name, Type ty) {
      return insertInput(getNumInOrInoutPorts(), name, ty);
    }

    std::pair<StringAttr, BlockArgument>
    appendInput(StringAttr name, Type ty) {
      return insertInput(getNumInOrInoutPorts(), name.getValue(), ty);
    }

    /// Prepend an input with a given name and type to the port list.
    /// If the name is not unique, a unique name is created and returned.
    std::pair<StringAttr, BlockArgument>
    prependInput(const Twine &name, Type ty) {
      return insertInput(0, name, ty);
    }

    std::pair<StringAttr, BlockArgument>
    prependInput(StringAttr name, Type ty) {
      return insertInput(0, name.getValue(), ty);
    }

    /// Insert an input with a given name and type into the port list.
    /// The input is added at the specified index.
    std::pair<StringAttr, BlockArgument>
    insertInput(unsigned index, StringAttr name, Type ty);

    std::pair<StringAttr, BlockArgument>
    insertInput(unsigned index, const Twine &name, Type ty) {
      ::mlir::StringAttr nameAttr = ::mlir::StringAttr::get(getContext(), name);
      return insertInput(index, nameAttr, ty);
    }

    /// Append an output with a given name and type to the port list.
    /// If the name is not unique, a unique name is created.
    void appendOutput(StringAttr name, Value value) {
      return insertOutputs(getResultTypes().size(), {{name, value}});
    }

    void appendOutput(const Twine &name, Value value) {
      ::mlir::StringAttr nameAttr = ::mlir::StringAttr::get(getContext(), name);
      return insertOutputs(getResultTypes().size(), {{nameAttr, value}});
    }

    /// Prepend an output with a given name and type to the port list.
    /// If the name is not unique, a unique name is created.
    void prependOutput(StringAttr name, Value value) {
      return insertOutputs(0, {{name, value}});
    }

    void prependOutput(const Twine &name, Value value) {
      ::mlir::StringAttr nameAttr = ::mlir::StringAttr::get(getContext(), name);
      return insertOutputs(0, {{nameAttr, value}});
    }

    /// Inserts a list of output ports into the port list at a specific
    /// location, shifting all subsequent ports.  Rewrites the output op
    /// to return the associated values.
    void insertOutputs(unsigned index,
                       ArrayRef<std::pair<StringAttr, Value>> outputs);

    Block *getBodyBlock() { return &getFunctionBody().front(); }

    // Get the module's symbolic name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the module's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getFunctionTypeAttrName().getValue() +
                           "' attribute of function type");
      return success();
    }

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def HWModuleExternOp : HWModuleOpBase<"module.extern"> {
  let summary = "HW external Module";
  let description = [{
    The "hw.module.extern" operation represents an external reference to a
    Verilog module, including a given name and a list of ports.

    The 'verilogName' attribute (when present) specifies the spelling of the
    module name in Verilog we can use.  TODO: This is a hack because we don't
    have proper parameterization in the hw.dialect.  We need a way to represent
    parameterized types instead of just concrete types.
  }];
  let arguments = (ins TypeAttrOf<FunctionType>:$function_type,
		       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
                       ParamDeclArrayAttr:$parameters,
                       OptionalAttr<StrAttr>:$verilogName);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports,
                   CArg<"StringRef", "StringRef()">:$verilogName,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "StringAttr":$name, "const ModulePortInfo &":$ports,
                   CArg<"StringRef", "StringRef()">:$verilogName,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraModuleClassDeclaration = [{

    /// Return the PortInfo for the specified input or inout port.
    PortInfo getInOrInoutPort(size_t i) {
      return getModuleInOrInoutPort(*this, i);
    }

    /// Return the PortInfo for the specified output port.
    PortInfo getOutputPort(size_t i) {
      return getModuleOutputPort(*this, i);
    }

    /// Return the name to use for the Verilog module that we're referencing
    /// here.  This is typically the symbol, but can be overridden with the
    /// verilogName attribute.
    StringRef getVerilogModuleName() {
      return getVerilogModuleNameAttr().getValue();
    }

    /// Return the name to use for the Verilog module that we're referencing
    /// here.  This is typically the symbol, but can be overridden with the
    /// verilogName attribute.
    StringAttr getVerilogModuleNameAttr();

    // Get the module's symbolic name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the module's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }

    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getFunctionTypeAttrName().getValue() +
                           "' attribute of function type");
      return success();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def HWGeneratorSchemaOp : HWOp<"generator.schema",
      [Symbol, HasParent<"mlir::ModuleOp">]> {
  let summary = "HW Generator Schema declaration";
  let description = [{
    The "hw.generator.schema" operation declares a kind of generated module by
    declaring the schema of meta-data required.
    A generated module instance of a schema is independent of the external
    method of producing it.  It is assumed that for well known schema instances,
    multiple external tools might exist which can process it.  Generator nodes
    list attributes required by hw.module.generated instances.

    For example:
    generator.schema @MEMORY, "Simple-Memory", ["ports", "write_latency", "read_latency"]
    module.generated @mymem, @MEMORY(ports)
      -> (ports) {write_latency=1, read_latency=1, ports=["read","write"]}
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$descriptor,
                       StrArrayAttr:$requiredAttrs);
  let results = (outs);
  let assemblyFormat = "$sym_name `,` $descriptor `,` $requiredAttrs attr-dict";
}

def HWModuleGeneratedOp : HWModuleOpBase<"module.generated", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>,
      IsolatedFromAbove]> {
  let summary = "HW Generated Module";
  let description = [{
    The "hw.module.generated" operation represents a reference to an external
    module that will be produced by some external process.
    This represents the name and list of ports to be generated.

    The 'verilogName' attribute (when present) specifies the spelling of the
    module name in Verilog we can use.  See hw.module for an explanation.
  }];
  let arguments = (ins FlatSymbolRefAttr:$generatorKind,
                       TypeAttrOf<FunctionType>:$function_type,
		       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
                       ParamDeclArrayAttr:$parameters,
                       OptionalAttr<StrAttr>:$verilogName);
  let results = (outs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "FlatSymbolRefAttr":$genKind, "StringAttr":$name,
                   "ArrayRef<PortInfo>":$ports,
                   CArg<"StringRef", "StringRef()">:$verilogName,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "FlatSymbolRefAttr":$genKind, "StringAttr":$name,
                   "const ModulePortInfo &":$ports,
                   CArg<"StringRef", "StringRef()">:$verilogName,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraModuleClassDeclaration = [{
    /// Return the name to use for the Verilog module that we're referencing
    /// here.  This is typically the symbol, but can be overridden with the
    /// verilogName attribute.
    StringRef getVerilogModuleName() {
      return getVerilogModuleNameAttr().getValue();
    }

    /// Return the name to use for the Verilog module that we're referencing
    /// here.  This is typically the symbol, but can be overridden with the
    /// verilogName attribute.
    StringAttr getVerilogModuleNameAttr();

    /// Lookup the generator kind for the symbol.  This returns null on
    /// invalid IR.
    Operation *getGeneratorKindOp();

    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getFunctionTypeAttrName().getValue() +
                           "' attribute of function type");
      return success();
    }

    TypeAttr getTypeAttr();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def InstanceOp : HWOp<"instance", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
        DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>,
        DeclareOpInterfaceMethods<HWInstanceLike>]> {
  let summary = "Create an instance of a module";
  let description = [{
    This represents an instance of a module. The inputs and results are
    the referenced module's inputs and outputs.  The `argNames` and
    `resultNames` attributes must match the referenced module.

    Any parameters in the "old" format (slated to be removed) are stored in the
    `oldParameters` dictionary.
  }];

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnyType>:$inputs,
                       StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
                       ParamDeclArrayAttr:$parameters,
                       OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    /// Create a instance that refers to a known module.
    OpBuilder<(ins "Operation*":$module, "StringAttr":$name,
                   "ArrayRef<Value>":$inputs,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"InnerSymAttr", "{}">:$innerSym)>,
    /// Create a instance that refers to a known module.
    OpBuilder<(ins "Operation*":$module, "StringRef":$name,
                   "ArrayRef<Value>":$inputs,
                   CArg<"ArrayAttr", "{}">:$parameters,
                   CArg<"InnerSymAttr", "{}">:$innerSym), [{
      build($_builder, $_state, module, $_builder.getStringAttr(name), inputs,
            parameters, innerSym);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Return the name of the specified input port or null if it cannot be
    /// determined.
    StringAttr getArgumentName(size_t i);

    /// Return the name of the specified result or null if it cannot be
    /// determined.
    StringAttr getResultName(size_t i);

    /// Change the name of the specified input port.
    void setArgumentName(size_t i, StringAttr name);

    /// Change the name of the specified output port.
    void setResultName(size_t i, StringAttr name);

    /// Change the names of all the input ports.
    void setArgumentNames(ArrayAttr names) {
      setArgNamesAttr(names);
    }

    /// Change the names of all the result ports.
    void setResultNames(ArrayAttr names) {
      setResultNamesAttr(names);
    }

    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    Operation *getReferencedModule(const HWSymbolCache *cache);

    /// Return the value for a port in port order.
    Value getValue(size_t idx);

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// An InstanceOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();

  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def OutputOp : HWOp<"output", [Terminator, HasParent<"HWModuleOp, HWTestModuleOp">,
                                Pure, ReturnLike]> {
  let summary = "HW termination operation";
  let description = [{
    "hw.output" marks the end of a region in the HW dialect and the values
    to put on the output ports.
  }];

  let arguments = (ins Variadic<AnyType>:$outputs);

  let builders = [
    OpBuilder<(ins), "build($_builder, $_state, std::nullopt);">
  ];

  let assemblyFormat = "attr-dict ($outputs^ `:` qualified(type($outputs)))?";

  let hasVerifier = 1;
}

def GlobalRefOp : HWOp<"globalRef", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    IsolatedFromAbove, Symbol]> {
  let summary = "A global reference to uniquely identify each"
                                    "instance of an operation";
  let description = [{
    This works like a symbol reference to an operation by specifying the
    instance path to uniquely identify it globally.
    It can be used to attach per instance metadata (non-local attributes).
    This also lets components of the path point to a common entity.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, NameRefArrayAttr:$namepath);

  let assemblyFormat = [{ $sym_name $namepath attr-dict }];
}

def HierPathOp : HWOp<"hierpath",
      [IsolatedFromAbove, Symbol,
       DeclareOpInterfaceMethods<InnerRefUserOpInterface>]> {
  let summary = "Hierarchical path specification";
  let description = [{
    The "hw.hierpath" operation represents a path through the hierarchy.
    This is used to specify namable things for use in other operations, for
    example in verbatim substitution. Non-local annotations also use these.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, NameRefArrayAttr:$namepath);
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Drop the module from the namepath. If its a InnerNameRef, then drop
    /// the Module-Instance pair, else drop the final module from the namepath.
    /// Return true if any update is made.
    bool dropModule(StringAttr moduleToDrop);

    /// Inline the module in the namepath.
    /// Update the symbol name for the inlined module instance, by prepending
    /// the symbol name of the instance at which the inling was done.
    /// Return true if any update is made.
    bool inlineModule(StringAttr moduleToDrop);

    /// Replace the oldMod module with newMod module in the namepath of the NLA.
    /// Return true if any update is made.
    bool updateModule(StringAttr oldMod, StringAttr newMod);

    /// Replace the oldMod module with newMod module in the namepath of the NLA.
    /// Since the module is being updated, the symbols inside the module should
    /// also be renamed. Use the rename Map to update the corresponding
    /// inner_sym names in the namepath. Return true if any update is made.
    bool updateModuleAndInnerRef(StringAttr oldMod, StringAttr newMod,
         const llvm::DenseMap<StringAttr, StringAttr> &innerSymRenameMap);

    /// Truncate the namepath for this NLA, at atMod module.
    /// If includeMod is false, drop atMod and beyond, else include it and drop
    /// everything after it.
    /// Return true if any update is made.
    bool truncateAtModule(StringAttr atMod, bool includeMod = true);

    /// Return just the module part of the namepath at a specific index.
    StringAttr modPart(unsigned i);

    /// Return the root module.
    StringAttr root();

    /// Return just the reference part of the namepath at a specific index.
    /// This will return an empty attribute if this is the leaf and the leaf is
    /// a module.
    StringAttr refPart(unsigned i);

    /// Return the leaf reference.  This returns an empty attribute if the leaf
    /// reference is a module.
    StringAttr ref();

    /// Return the leaf Module.
    StringAttr leafMod();

    /// Returns true, if the NLA path contains the module.
    bool hasModule(StringAttr modName);

    /// Returns true, if the NLA path contains the InnerSym {modName, symName}.
    bool hasInnerSym(StringAttr modName, StringAttr symName) const;

    /// Returns true if this NLA targets a module or instance of a module (as
    /// opposed to an instance's port or something inside an instance).
    bool isModule();

    /// Returns true if this NLA targets something inside a module (as opposed
    /// to a module or an instance of a module);
    bool isComponent();
    }];
}

// Edge behavior for trigger blocks. Currently these map 1:1 to SV event
// control kinds.

/// AtPosEdge triggers on a rise from 0 to 1/X/Z, or X/Z to 1.
def AtPosEdge: I32EnumAttrCase<"AtPosEdge", 0, "posedge">;
/// AtNegEdge triggers on a drop from 1 to 0/X/Z, or X/Z to 0.
def AtNegEdge: I32EnumAttrCase<"AtNegEdge", 1, "negedge">;
/// AtEdge(v) is syntactic sugar for "AtPosEdge(v) or AtNegEdge(v)".
def AtEdge   : I32EnumAttrCase<"AtEdge", 2, "edge">;

def EventControlAttr : I32EnumAttr<"EventControl", "edge control trigger",
                                   [AtPosEdge, AtNegEdge, AtEdge]> {
  let cppNamespace = "circt::hw";
}

def TriggeredOp : HWOp<"triggered", [
    IsolatedFromAbove, SingleBlock, NoTerminator]> {
  let summary = "A procedural region with a trigger condition";
  let description = [{
    A procedural region that can be triggered by an event. The trigger
    condition is a 1-bit value that is activated based on some event control
    attribute.
    The operation is `IsolatedFromAbove`, and thus requires values passed into
    the trigger region to be explicitly passed in through the `inputs` list.
  }];

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins EventControlAttr:$event, I1:$trigger, Variadic<AnyType>:$inputs);
  let results = (outs);

  let assemblyFormat = [{
    $event $trigger  `(` $inputs `)` `:` type($inputs) $body attr-dict
  }];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }

    // Return the input arguments inside the trigger region.
    ArrayRef<BlockArgument> getInnerInputs() {
      return getBodyBlock()->getArguments();
    }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "EventControlAttr":$event, "Value":$trigger, "ValueRange":$inputs)>
  ];
}

#endif // CIRCT_DIALECT_HW_HWSTRUCTURE_TD
