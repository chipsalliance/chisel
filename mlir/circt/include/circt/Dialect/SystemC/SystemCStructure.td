//===- SystemCStructure.td - SystemC structure ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes ops with regions for SystemC to structure other ops, such as
// modules, methods, etc.
//
//===----------------------------------------------------------------------===//

def SCModuleOp : SystemCOp<"module", [
    Symbol,
    DeclareOpInterfaceMethods<HWModuleLike>,
    FunctionOpInterface,
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    RegionKindInterface,
    DeclareOpInterfaceMethods<CallableOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    HasParent<"mlir::ModuleOp">
  ]> {
  let summary = "Define a SystemC SC_MODULE.";
  let description = [{
    Represents the SC_MODULE macro as described in IEEE 1666-2011 §5.2.5.
    Models input, output and inout ports as module arguments (as opposed to
    `sc_signal`s which are modeled by a separate `systemc.signal` operation),
    but are nonetheless emitted as regular struct fields.
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    StrArrayAttr:$portNames
  );
  let regions = (region SizedRegion<1>: $body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<hw::PortInfo>":$ports,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "StringAttr":$name, "const hw::ModulePortInfo &":$ports,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "StringAttr":$name, "ArrayAttr":$portNames,
                   "ArrayRef<Type>":$portTypes,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>
  ];

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) {
      return RegionKind::Graph;
    }

    /// Returns the argument types of this function.
    ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    // Use FunctionOpInterface traits's getFunctionBody method.
    using mlir::detail::FunctionOpInterfaceTrait<SCModuleOp>::getFunctionBody;

    /// Return the block corresponding to the region.
    Block *getBodyBlock() { return &getFunctionBody().front(); }

    /// Return the symbol name of this module as string.
    StringRef getModuleName();

    using PortDirectionRange = llvm::iterator_range<
      llvm::filter_iterator<Region::BlockArgListType::iterator,
                            std::function<bool(const BlockArgument &)>>>;

    PortDirectionRange getPortsOfDirection(hw::ModulePort::Direction direction);
    PortDirectionRange getInputPorts() {
      return getPortsOfDirection(hw::ModulePort::Direction::Input);
    }
    PortDirectionRange getOutputPorts() {
      return getPortsOfDirection(hw::ModulePort::Direction::Output);
    }
    PortDirectionRange getInOutPorts() {
      return getPortsOfDirection(hw::ModulePort::Direction::InOut);
    }

    // Return the Ctor operation in this module's body or create one if none
    // exists yet.
    systemc::CtorOp getOrCreateCtor();

    // Return the Destructor operation in this module's body or create one if
    // none exists yet.
    systemc::DestructorOp getOrCreateDestructor();

    //===------------------------------------------------------------------===//
    // PortList Methods
    //===------------------------------------------------------------------===//
    ::circt::hw::ModulePortInfo getPortList();
  }];

  let extraClassDefinition = [{
    size_t $cppClass::getNumPorts() {
      return getPortNames().size();
    }

    circt::hw::InnerSymAttr $cppClass::getPortSymbolAttr(size_t portIndex) {
      for (NamedAttribute argAttr :
           mlir::function_interface_impl::getArgAttrs(*this, portIndex))
        if (auto sym = argAttr.getValue().dyn_cast<circt::hw::InnerSymAttr>())
          return sym;
      return circt::hw::InnerSymAttr();
    }
  }];
}

def CtorOp : SystemCOp<"ctor", [SingleBlock, NoTerminator,
                                HasParent<"SCModuleOp">]> {
  let summary = "A constructor definition.";
  let description = [{
    Represents the SC_CTOR macro as described in IEEE 1666-2011 §5.2.7.
    The name of the module being constructed does not have to be passed
    to this operation, but is automatically added during emission.
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";

  let builders = [
    OpBuilder<(ins), [{
      Region *region = $_state.addRegion();
      region->push_back(new Block);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the block corresponding to the region.
    Block *getBodyBlock() { return &getBody().front(); }
  }];

  let hasVerifier = true;
  let skipDefaultBuilders = 1;
}

def SCFuncOp : SystemCOp<"func", [
  HasCustomSSAName,
  SystemCNameDeclOpInterface,
  SingleBlock,
  NoTerminator,
  HasParent<"SCModuleOp">
]> {
  let summary = "A (void)->void member function of a SC_MODULE.";
  let description = [{
    This operation does not represent a specific SystemC construct, but a
    regular C++ member function with no arguments and a void return type.
    These are used to implement module-internal logic and are registered to the
    module using the SC_METHOD, SC_THREAD, and SC_CTHREAD macros.
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs NullaryVoidFunctionType:$handle);
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "StringAttr":$name), [{
      $_state.addAttribute(getNameAttrName($_state.name), name);
      Region *region = $_state.addRegion();
      $_state.addTypes(FunctionType::get($_builder.getContext(), {}, {}));
      region->push_back(new Block);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the block corresponding to the region.
    Block *getBodyBlock() { return &getBody().front(); }
  }];

  let assemblyFormat = [{
    `` custom<ImplicitSSAName>($name) attr-dict-with-keyword $body
  }];

  let hasVerifier = true;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// Interoperability operations
//===----------------------------------------------------------------------===//

def InteropVerilatedOp : SystemCOp<"interop.verilated", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Instantiates a verilated module.";
  let description = [{
    Instantiates a verilated module represented by a hw.module operation
    (usually the extern variant).

    This operation also encodes the interoparability layer to connect its
    context (i.e. the surrounding operation, input values, result values, and
    types) to the C++ code of the verilated module.
    When residing in a context that understands C++ (e.g., inside a SystemC
    module), this refers to the instantiation of the class, assignment of the
    input ports, the call to the eval() function and reading the output ports.

    Additionally, properties of the verilated module can be specified in
    a config attribute which influences the interop layer code generation
    (not yet implemented).
  }];

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName,
                       StrArrayAttr:$inputNames,
                       StrArrayAttr:$resultNames,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    /// Create a instance that refers to a known module.
    OpBuilder<(ins "Operation*":$module, "StringAttr":$name,
                   "ArrayRef<Value>":$inputs)>,
    /// Create a instance that refers to a known module.
    OpBuilder<(ins "Operation*":$module, "StringRef":$name,
                   "ArrayRef<Value>":$inputs), [{
      build($_builder, $_state, module, $_builder.getStringAttr(name), inputs);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Return the name of the specified input port or null if it cannot be
    /// determined.
    StringAttr getInputName(size_t i) {
      return hw::instance_like_impl::getName(getInputNames(), i);
    }

    /// Return the name of the specified result or null if it cannot be
    /// determined.
    StringAttr getResultName(size_t i) {
      return hw::instance_like_impl::getName(getResultNames(), i);
    }

    /// Change the name of the specified input port.
    void setInputName(size_t i, StringAttr name) {
      setInputNamesAttr(
        hw::instance_like_impl::updateName(getInputNames(), i, name));
    }

    /// Change the name of the specified output port.
    void setResultName(size_t i, StringAttr name) {
      setResultNamesAttr(
        hw::instance_like_impl::updateName(getResultNames(), i, name));
    }

    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    Operation *getReferencedModule(const hw::HWSymbolCache *cache) {
      return hw::instance_like_impl::getReferencedModule(cache, *this,
                                                        getModuleNameAttr());
    }
    Operation *getReferencedModule() { return getReferencedModule(nullptr); }

    /// Get the instances's name.
    StringAttr getName() { return getInstanceNameAttr(); }

    /// Set the instance's name.
    void setName(StringAttr name) { setInstanceNameAttr(name); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// An InstanceOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }
  }];

  let assemblyFormat = [{
    $instanceName $moduleName
    custom<InputPortList>($inputs, type($inputs), $inputNames) `->`
    custom<OutputPortList>(type($results), $resultNames) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Operations that model C++-level features.
//===----------------------------------------------------------------------===//

def DestructorOp : SystemCOp<"cpp.destructor", [
    SingleBlock,
    NoTerminator,
    HasParent<"SCModuleOp">
]> {
  let summary = "A C++ destructor definition.";
  let description = [{
    This operation models a C++ destructor of a class or struct. It is not an
    operation modelling some abstract SystemC construct, but still required to
    support more complex functionality such as having a pointer to an external
    object inside a SystemC module, e.g., for interoperability purposes.
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";

  let builders = [
    OpBuilder<(ins), [{
      Region *region = $_state.addRegion();
      region->push_back(new Block);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the block corresponding to the region.
    Block *getBodyBlock() { return &getBody().front(); }
  }];

  let hasVerifier = true;
  let skipDefaultBuilders = 1;
}

def FuncOp : SystemCOp<"cpp.func", [
  AutomaticAllocationScope,
  CallableOpInterface,
  FunctionOpInterface,
  IsolatedFromAbove,
  OpAsmOpInterface,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  Symbol
]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external function declaration (used when
    referring to a function declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for function arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Argument names are stored in a 'argNames' attribute, but used directly as
    the SSA value's names. They are verified to be unique and can be used to
    print them, e.g., as C function argument names.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

    Example:

    ```mlir
    // External function definitions.
    systemc.cpp.func @abort()
    systemc.cpp.func externC @scribble(i32, i64) -> i64

    // A function that returns its argument twice:
    systemc.cpp.func @count(%argumentName: i64) -> (i64, i64) {
      return %argumentName, %argumentName: i64, i64
    }

    // A function with an attribute
    systemc.cpp.func @exampleFnAttr() attributes {dialectName.attrName = false}
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       StrArrayAttr:$argNames,
                       UnitAttr:$externC,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "ArrayAttr":$argNames, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name,
                         ArrayAttr argNames, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static FuncOp create(Location location, StringRef name,
                         ArrayAttr argNames, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static FuncOp create(Location location, StringRef name,
                         ArrayAttr argNames, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    FuncOp clone(IRMapping &mapper);
    FuncOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(FuncOp dest, IRMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "systemc"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def ReturnOp : SystemCOp<"cpp.return", [
  Pure,
  HasParent<"systemc::FuncOp">,
  ReturnLike, Terminator
]> {
  let summary = "Function return operation";
  let description = [{
    The `systemc.cpp.return` operation represents a return operation within a
    function. The operand number and types must match the signature of the
    function that contains the operation.

    Example:

    ```mlir
    systemc.cpp.func @foo() : i32 {
      ...
      systemc.cpp.return %0 : i32
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$returnValues);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($returnValues^ `:` type($returnValues))?";
  let hasVerifier = 1;
}
