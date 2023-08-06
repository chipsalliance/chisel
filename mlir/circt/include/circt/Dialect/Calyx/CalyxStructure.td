//===- CalyxStructure.td - Calyx Structure -------------*- tablegen -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the Calyx structures.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"

// "Forward-declare" these HW attributes rather than including or duplicating
// them here. This lets us to refer to them in ODS, but delegates to HW in C++.
// These are used to represent parameters for the PrimitiveOp.
def ParamDeclAttr : Attr<CPred<"$_self.isa<hw::ParamDeclAttr>()">>;
def ParamDeclArrayAttr : TypedArrayAttrBase<ParamDeclAttr, "parameter array">;

def UndefinedOp : CalyxOp<"undef", [
    Pure
  ]> {
  let summary = "Calyx Undefined Value";
  let description = [{
    The "undef" operation represents an undefined value
    that may be used when a specific source or destination
    does not have an assignment yet. This is used to avoid
    pulling in the entire LLVMIR dialect for a single
    operation.

    ```mlir
      %0 = calyx.undef : i1
    ```
  }];
  let results = (outs AnyType:$res);
  let assemblyFormat = "attr-dict `:` qualified(type($res))";
}

def ComponentOp : CalyxOp<"component", [
    HasParent<"mlir::ModuleOp">,
    RegionKindInterface,
    SymbolTable, /* contains Cell names. */
    Symbol,
    FunctionOpInterface,
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    OpAsmOpInterface,
    DeclareOpInterfaceMethods<ComponentOpInterface>
  ]> {
  let summary = "Calyx Component";
  let description = [{
    The "calyx.component" operation represents an overall Calyx component containing:
    (1) In- and output port definitions that define the interface.
    (2) The cells, wires, and control schedule.

    A Calyx component requires attributes `clk`, `go`, and `reset` on separate input ports,
    and `done` on an output port.

    ```mlir
      calyx.component @C(%in: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i16, %done: i1 {done}) {
        ...
        calyx.wires { ... }
        calyx.control { ... }
      }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    ArrayAttr:$portNames,
    ArrayAttr:$portAttributes,
    APIntAttr:$portDirections
  );
  let results = (outs);
  let regions = (region SizedRegion<1>: $body);

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports)>
  ];

  let extraClassDeclaration = [{
      using mlir::detail::FunctionOpInterfaceTrait<ComponentOp>::front;
      using mlir::detail::FunctionOpInterfaceTrait<ComponentOp>::getFunctionBody;

      // Implement RegionKindInterface.
      static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph;}

      /// Returns the argument types of this function.
      ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

      /// Returns the result types of this function.
      ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

      /// Verify the type attribute of this function. Returns failure and emits
      /// an error if the attribute is invalid.
      LogicalResult verifyType() {
        auto type = getFunctionTypeAttr().getValue();
        if (!type.isa<FunctionType>())
          return emitOpError("requires '" +
	                     getFunctionTypeAttrName().getValue() +
                             "' attribute of function type");
        return success();
      }

      /// Returns the WiresOp of a Calyx Component.
      WiresOp getWiresOp();

      /// Returns the ControlOp of a Calyx Component.
      ControlOp getControlOp();

      /// Gets the Done port of a Calyx component.
      Value getDonePort();

      /// Gets the Go port of a Calyx component.
      Value getGoPort();

      /// Gets the Clock port of a Calyx component.
      Value getClkPort();

      /// Gets the Reset port of a Calyx component.
      Value getResetPort();

      void getAsmBlockArgumentNames(mlir::Region &region,
                                    mlir::OpAsmSetValueNameFn setNameFn);
  }];

  let extraClassDefinition = [{
      Block* $cppClass::getBodyBlock() {
        Region* region = getRegion();
        assert(region->hasOneBlock() && "The body should have one Block.");
        return &region->front();
      }

      StringRef $cppClass::getName() {
        return cast<FunctionOpInterface>(**this).getName();
      }

      bool $cppClass::isComb() {
        return false;
      }

      /// Returns the region of a Calyx component.
      Region* $cppClass::getRegion() { return &getFunctionBody(); }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def CombComponentOp : CalyxOp<"comb_component", [
    HasParent<"mlir::ModuleOp">,
    SymbolTable, /* contains Cell names. */
    Symbol,
    FunctionOpInterface,
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    OpAsmOpInterface,
    DeclareOpInterfaceMethods<ComponentOpInterface>
  ]> {
  let summary = "Calyx Combinational Component";
  let description = [{
    The "calyx.comb_component" operation represents a Calyx combinational component containing:
    (1) In- and output port definitions that define the interface.
    (2) Combinational cells and wires.

    ```mlir
      calyx.comb_component @C(%in: i32) -> (%out: i16) {
        ...
        calyx.wires { ... }
      }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    ArrayAttr:$portNames,
    ArrayAttr:$portAttributes,
    APIntAttr:$portDirections
  );
  let results = (outs);
  let regions = (region SizedRegion<1>: $body);

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports)>
  ];

  let extraClassDeclaration = [{
      using mlir::detail::FunctionOpInterfaceTrait<CombComponentOp>::front;
      using mlir::detail::FunctionOpInterfaceTrait<CombComponentOp>::getFunctionBody;

      /// Returns the argument types of this function.
      ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

      /// Returns the result types of this function.
      ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

      /// Verify the type attribute of this function. Returns failure and emits
      /// an error if the attribute is invalid.
      LogicalResult verifyType() {
        auto type = getFunctionTypeAttr().getValue();
        if (!type.isa<FunctionType>())
          return emitOpError("requires '" +
	                     getFunctionTypeAttrName().getValue() +
                             "' attribute of function type");
        return success();
      }

      /// Returns the WiresOp of a Calyx Component.
      WiresOp getWiresOp();

      void getAsmBlockArgumentNames(mlir::Region &region,
                                    mlir::OpAsmSetValueNameFn setNameFn);
  }];

  let extraClassDefinition = [{
      Block* $cppClass::getBodyBlock() {
        Region* region = getRegion();
        assert(region->hasOneBlock() && "The body should have one Block.");
        return &region->front();
      }

      StringRef $cppClass::getName() {
        return cast<FunctionOpInterface>(**this).getName();
      }

      bool $cppClass::isComb() {
        return true;
      }

      /// Returns the region of a Calyx component.
      Region* $cppClass::getRegion() { return &getFunctionBody(); }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def WiresOp : CalyxContainer<"wires", [
    ParentOneOf<["ComponentOp", "CombComponentOp"]>,
    SymbolTable /* contains GroupInterface names. */,
    RegionKindInterface
  ]> {
  let summary = "Calyx Wires";
  let description = [{
    The "calyx.wires" operation represents a set of
    guarded connections between component instances,
    which may be placed within groups.

    ```mlir
      calyx.wires {
        calyx.group @A { ... }
        calyx.assign %1 = %2 : i16
      }
    ```
  }];
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins), [{
      Region* region = $_state.addRegion();
      region->push_back(new Block());
    }]>
  ];

  let extraClassDeclaration = [{
    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the body of a Calyx component.
    Block *getBodyBlock() {
      Region* region = &getOperation()->getRegion(0);
      assert(region->hasOneBlock() && "The body should have one Block.");
      return &region->front();
    }

  }];

  let hasVerifier = 1;
}

def InstanceOp : CalyxCell<"instance", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Calyx Component Instance";
  let description = [{
    Represents an instance of a Calyx component, which may include state.

    ```mlir
      %c.in, %c.out = calyx.instance @c of @MyComponent : i64, i16
    ```
  }];

  let extraClassDeclaration = [{
    /// Lookup the component for the symbol. This returns null on
    /// invalid IR.
    ComponentInterface getReferencedComponent();
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$componentName
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $sym_name `of` $componentName attr-dict (`:` qualified(type($results))^)?
  }];
}

def PrimitiveOp : CalyxCell<"primitive", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Calyx Primitive Instance";
  let description = [{
    Represents an instance of a Calyx primitive.

    ```mlir
      %c.in, %c.out = calyx.primitive @c of @MyExternalPrimitive : i64, i16
    ```
  }];

  let extraClassDeclaration = [{
    /// Lookup the component for the symbol. This returns null on
    /// invalid IR.
    hw::HWModuleExternOp getReferencedPrimitive();
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<ParamDeclArrayAttr>:$parameters,
    FlatSymbolRefAttr:$primitiveName
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $sym_name `of` $primitiveName `` custom<ParameterList>($parameters) attr-dict (`:` qualified(type($results))^)?
  }];
}


def GroupOp : CalyxOp<"group", [
    HasParent<"WiresOp">,
    DeclareOpInterfaceMethods<GroupOpInterface>,
    NoRegionArguments,
    RegionKindInterface,
    SingleBlock,
    Symbol
  ]> {
  let summary = "Calyx Group";
  let description = [{
    Represents a Calyx group, which is a collection
    of assignments that are only active when the group
    is run from the control execution schedule. A group
    signifies its termination with a special port named
    a "done" port.

    ```mlir
      calyx.group @MyGroup {
        calyx.assign %1 = %2 : i32
        calyx.group_done %3 : i1
      }
    ```
  }];

  let arguments = (ins SymbolNameAttr: $sym_name);

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the GroupGoOp for this group.
    GroupGoOp getGoOp();

    /// Returns the GroupDoneOp for this group.
    GroupDoneOp getDoneOp();

    /// Returns the body of a Calyx component.
    Block *getBodyBlock() {
      Region* region = &getOperation()->getRegion(0);
      assert(region->hasOneBlock() && "The body should have one Block.");
      return &region->front();
    }

  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $body attr-dict";
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name), [{
      $_state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(),
        StringAttr::get($_state.getContext(), sym_name)
      );
      Region* region = $_state.addRegion();
      region->push_back(new Block());
    }]>,
    OpBuilder<(ins "StringAttr":$sym_name), [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), sym_name);
      Region* region = $_state.addRegion();
      region->push_back(new Block());
    }]>
  ];
}

def CombGroupOp : CalyxOp<"comb_group", [
    HasParent<"WiresOp">,
    DeclareOpInterfaceMethods<GroupOpInterface>,
    NoRegionArguments,
    RegionKindInterface,
    SingleBlock,
    Symbol,
    NoTerminator
  ]> {
  let summary = "Calyx Combinational Group";
  let description = [{
    Represents a Calyx combinational group, which is a collection
    of combinational assignments that are only active when the group
    is run from the control execution schedule.
    A combinational group does not have group_go or group_done operators.

    ```mlir
      calyx.comb_group @MyCombGroup {
        calyx.assign %1 = %2 : i32
      }
    ```
  }];
  let arguments = (ins SymbolNameAttr: $sym_name);
  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the body of a Calyx component.
    Block *getBodyBlock() {
      Region* region = &getOperation()->getRegion(0);
      assert(region->hasOneBlock() && "The body should have one Block.");
      return &region->front();
    }
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $body attr-dict";
  let hasVerifier = 1;
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name), [{
      $_state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(),
        StringAttr::get($_state.getContext(), sym_name)
      );
      Region* region = $_state.addRegion();
      region->push_back(new Block());
    }]>,
    OpBuilder<(ins "StringAttr":$sym_name), [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), sym_name);
      Region* region = $_state.addRegion();
      region->push_back(new Block());
    }]>
  ];
}

def StaticGroupOp : CalyxOp<"static_group", [
    HasParent<"WiresOp">,
    DeclareOpInterfaceMethods<GroupOpInterface>,
    NoRegionArguments,
    RegionKindInterface,
    SingleBlock,
    Symbol,
    NoTerminator
  ]> {
  let summary = "Calyx Static Group";
  let description = [{
    Represents a Calyx static group, which is a collection
    of assignments that are only active when the group
    is run from the control execution schedule. A group
    signifies its termination with a special port named
    a "done" port.

    ```mlir
      calyx.static_group latency<1> @MyStaticGroup {
        calyx.assign %1 = %2 : i32
      }
    ```
  }];

  let arguments = (ins SymbolNameAttr: $sym_name,
                   I64Attr: $latency);

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the GroupGoOp for this group.
    GroupGoOp getGoOp();

    /// Returns the body of a Calyx component.
    Block *getBodyBlock() {
      Region* region = &getOperation()->getRegion(0);
      assert(region->hasOneBlock() && "The body should have one Block.");
      return &region->front();
    }
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "`latency` `<` $latency `>` $sym_name $body attr-dict";
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name, "uint64_t":$latency), [{
      $_state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(),
        StringAttr::get($_state.getContext(), sym_name)
      );
      $_state.addAttribute(
        getLatencyAttrName($_state.name),
        mlir::IntegerAttr::get(IntegerType::get($_state.getContext(), 64), latency)
      );
      Region* region = $_state.addRegion();
      region->push_back(new Block());
    }]>,
    OpBuilder<(ins "StringAttr":$sym_name, "uint64_t":$latency), [{
      $_state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), sym_name);
      $_state.addAttribute(
        getLatencyAttrName($_state.name),
        mlir::IntegerAttr::get(IntegerType::get($_state.getContext(), 64), latency)
      );
      Region* region = $_state.addRegion();
      region->push_back(new Block());
    }]>
  ];
}

def AssignOp : CalyxOp<"assign", [
    SameTypeConstraint<"dest", "src">,
    ParentOneOf<["GroupOp", "CombGroupOp", "StaticGroupOp","WiresOp"]>
  ]> {
  let summary = "Calyx Assignment";
  let description = [{
    The "calyx.assign" operation represents a non-blocking
    assignment. An assignment may optionally be guarded,
    which controls when the assignment should be active.
    This operation should only be instantiated in the
    "calyx.wires" section or a "calyx.group".

    ```mlir
      calyx.assign %dest = %src : i16
      calyx.assign %dest = %guard ? %src : i16
    ```
  }];
  let arguments = (ins
    AnyType:$dest,
    AnyType:$src,
    Optional<I1>:$guard
  );
  let builders = [
    OpBuilder<(ins "Value":$dest, "Value":$src), [{
      $_state.addOperands({dest, src});
    }]>
  ];
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def GroupDoneOp : CalyxGroupPort<"group_done", [
    Terminator
  ]> {
  let summary = "Calyx Group Done Port";
  let description = [{
    The "calyx.group_done" operation represents a port on a
    Calyx group that signifies when the group is finished.
    A done operation may optionally be guarded, which controls
    when the group's done operation should be active.

    ```mlir
      calyx.group_done %src : i1
      calyx.group_done %guard ? %src : i1
    ```
  }];
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$src), [{
      $_state.addOperands(src);
    }]>
  ];
}

def GroupGoOp : CalyxGroupPort<"group_go", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = "Calyx Group Go Port";
  let description = [{
    The "calyx.group_go" operation represents a port on a
    Calyx group that signifies when the group begins.
    A go operation may optionally be guarded, which
    controls when the group's go operation should be
    active. The go operation should only be inserted
    during the Go Insertion pass. It does not
    receive a source until the Compile Control pass.

    ```mlir
      %group_name1.go = calyx.group_go %src : i1
      %group_name2.go = calyx.group_go %guard ? %src : i1
    ```
  }];
  let results = (outs I1);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$src), [{
      $_state.addTypes($_builder.getI1Type());
      $_state.addOperands(src);
    }]>
  ];
}

def CycleOp : CalyxOp<"cycle", [
  HasParent<"StaticGroupOp">,
]> {
  let summary = "Calyx Static Cycle Op";
  let description = [{
    Returns an I1 signal that is active `start` cycles after the static 
    group starts, and inactive all other times. Optional `end` 
    attribute allows the signal to be active from until `end` cycles
    after the static group starts. Must be used within a static group.

    ```mlir
      calyx.static_group latency<2> @MyStaticGroup {
        %1 = calyx.cycle 1 : i1
      }
    ```
  }];
  let arguments = (ins
    I32Attr:$start,
    OptionalAttr<I32Attr>:$end
  );
  let results = (outs 
    I1:$active
  );
  let extraClassDeclaration = [{
    /// Returns the latency of the parent static group.
    uint32_t getGroupLatency();
  }];
  let builders = [
    OpBuilder<(ins "uint32_t":$start), [{
      $_state.addAttribute(
        getStartAttrName($_state.name),
        mlir::IntegerAttr::get(IntegerType::get($_state.getContext(), 32), start)
      );
    }]>,
    OpBuilder<(ins "uint32_t":$start, "uint32_t":$end), [{
      $_state.addAttribute(
        getStartAttrName($_state.name),
        mlir::IntegerAttr::get(IntegerType::get($_state.getContext(), 32), start)
      );
      $_state.addAttribute(
        getEndAttrName($_state.name),
        mlir::IntegerAttr::get(IntegerType::get($_state.getContext(), 32), end)
      );
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
