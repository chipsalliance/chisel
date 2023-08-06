//===- CalyxControl.td - Calyx Control -------------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the operations used for the Calyx control execution schedule.
//
//===----------------------------------------------------------------------===//

def ControlLike : NativeOpTrait<"ControlLike"> {
  let cppNamespace = "::circt::calyx";
}

def ControlOp : CalyxContainer<"control", [
    HasParent<"ComponentOp">
  ]> {
  let summary = "Calyx Control";
  let description = [{
    The "calyx.control" operation represents the
    execution schedule defined for the given
    component, i.e. when each group executes.

    ```mlir
      calyx.control {
        calyx.seq {
          calyx.enable @GroupA
        }
      }
    ```
  }];
  let hasVerifier = 1;
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins), [{
      Region* region = $_state.addRegion();
      region->push_back(new Block());
    }]>
  ];
  let extraClassDeclaration = commonClassDeclaration # [{
    /// Get the InvokeOp of the ControlOp.
    SmallVector<InvokeOp, 4> getInvokeOps();
  }];
}

def IfOp : CalyxContainer<"if", [
    ControlLike,
    DeclareOpInterfaceMethods<IfOpInterface>
  ]> {
  let summary = "Calyx If";
  let arguments = (ins
    I1:$cond,
    OptionalAttr<FlatSymbolRefAttr>:$groupName
    );
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
  let description = [{
    The "calyx.if" operation represents and if-then-else construct for
    conditionally executing two Calyx groups. The operands to an if operation is
    a 1-bit port and an optional combinational group under which this port is driven.

    Note: The native and CIRCT Calyx IRs may diverge wrt. 'with' execution, see:
    https://github.com/cucapra/calyx/discussions/588

    ```mlir
      calyx.if %1 with @G1 {
        calyx.enable @G2
        ...
      } else {
        calyx.enable @G3
        ...
      }
      calyx.if %1 {
        calyx.enable @G2
        ...
      }
    ```
  }];

  let assemblyFormat = "$cond (`with` $groupName^)? $thenRegion (`else` $elseRegion^)? attr-dict";
  let hasVerifier = 1;
  let hasCanonicalizer = true;
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins
      "Value":$cond,
      CArg<"FlatSymbolRefAttr", "nullptr">:$groupName,
      CArg<"bool", "false">:$initializeElseBody), [{
        $_state.addOperands(cond);
        if (groupName)
          $_state.addAttribute("groupName", groupName);

        Region *thenRegion = $_state.addRegion();
        Region *elseRegion = $_state.addRegion();
        thenRegion->push_back(new Block());
        if (initializeElseBody)
          elseRegion->push_back(new Block());
    }]>
  ];
}

def StaticIfOp : CalyxContainer<"static_if", [
    ControlLike,
    DeclareOpInterfaceMethods<IfOpInterface>
  ]> {
  let summary = "Calyx Static If";
  let arguments = (ins
    I1:$cond
    );
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
  let description = [{
    The "calyx.static_if" operation represents an if-then-else construct for
    conditionally executing two Calyx groups. The operands to an if operation is
    a 1-bit port. 

    ```mlir
      calyx.static_if %1 {
        calyx.enable @G2
        ...
      } else {
        calyx.enable @G3
        ...
      }
      calyx.if %1 {
        calyx.enable @G2
        ...
      }
    ```
  }];

  let assemblyFormat = "$cond $thenRegion (`else` $elseRegion^)? attr-dict";
  let hasVerifier = 1;
  let hasCanonicalizer = true;
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins
      "Value":$cond,
      CArg<"bool", "false">:$initializeElseBody), [{
        $_state.addOperands(cond);

        Region *thenRegion = $_state.addRegion();
        Region *elseRegion = $_state.addRegion();
        thenRegion->push_back(new Block());
        if (initializeElseBody)
          elseRegion->push_back(new Block());
    }]>
  ];
}


def SeqOp : CalyxContainer<"seq", [
    ControlLike
  ]> {
  let summary = "Calyx Sequential";
  let description = [{
    The "calyx.seq" operation executes the
    control within its region sequentially.

    ```mlir
      calyx.seq {
        // G2 will not begin execution until G1 is done.
        calyx.enable @G1
        calyx.enable @G2
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
  let hasCanonicalizer = true;
}


def StaticSeqOp : CalyxContainer<"static_seq", [
    ControlLike
  ]> {
  let summary = "Calyx Static Seq";
  let description = [{
    The "calyx.static_seq" operation executes the
    control within its region sequentially.

    ```mlir
      calyx.static_seq {
        // G2 will not begin execution until G1 is done.
        calyx.enable @G1
        calyx.enable @G2
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
  let hasCanonicalizer = true;
  let hasVerifier = 1;
}

def ParOp : CalyxContainer<"par", [
    ControlLike
  ]> {
  let summary = "Calyx Parallel";
  let description = [{
    The "calyx.par" operation executes the
    control within its region in parallel.

    ```mlir
      calyx.par {
        // G1 and G2 will execute in parallel.
        // The region is complete when both
        // G1 and G2 are done.
        calyx.enable @G1
        calyx.enable @G2
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
  let hasCanonicalizer = true;
  let hasVerifier = 1;
}

def StaticParOp : CalyxContainer<"static_par", [
    ControlLike
  ]> {
  let summary = "Calyx Static Parallel";
  let description = [{
    The "calyx.static_par" operation executes the
    control within its region in parallel.

    ```mlir
      calyx.static_par {
        // G1 and G2 will execute in parallel.
        // The region is complete when both
        // G1 and G2 are done.
        calyx.enable @G1
        calyx.enable @G2
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
  let hasCanonicalizer = true;
  let hasVerifier = 1;
}


def EnableOp : CalyxOp<"enable", [
    ControlLike
  ]> {
  let summary = "Calyx Enable";
  let description = [{
    The "calyx.enable" operation represents the execution of
    a group defined explicitly in the "calyx.wires" section.

    The 'compiledGroups' attribute is used in the Compile
    Control pass to track which groups are compiled within
    the new compilation group.

    ```mlir
      calyx.enable @SomeGroup
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$groupName,
    OptionalAttr<ArrayAttr>:$compiledGroups
  );
  let builders = [
    OpBuilder<(ins "StringRef":$groupName), [{
      $_state.addAttribute(
        "groupName",
        FlatSymbolRefAttr::get($_builder.getContext(), groupName)
      );
    }]>
  ];
  let assemblyFormat = "$groupName attr-dict";
  let hasVerifier = 1;
}

def WhileOp : CalyxContainer<"while", [
    ControlLike
  ]> {
  let summary = "Calyx While";
  let arguments = (ins
    I1:$cond,
    OptionalAttr<FlatSymbolRefAttr>:$groupName
  );
  let description = [{
    The "calyx.while" operation represents a construct for continuously
    executing the inner groups of the 'while' operation while the condition port
    evaluates to true. The operands to a while operation is a 1-bit port and an
    optional combinational group under which this port is driven.

    Note: The native and CIRCT Calyx IRs may diverge wrt. 'with' execution, see:
    https://github.com/cucapra/calyx/discussions/588

    ```mlir
      calyx.while %1 with @G1 {
        calyx.enable @G2
        ...
      }
      calyx.while %1 {
        calyx.enable @G2
        ...
      }
    ```
  }];
  let hasCanonicalizeMethod = true;
  let assemblyFormat = "$cond (`with` $groupName^)? $body attr-dict";
  let hasVerifier = 1;
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins
      "Value":$cond,
      CArg<"FlatSymbolRefAttr", "nullptr">:$groupName), [{
        $_state.addOperands(cond);
        if (groupName)
          $_state.addAttribute("groupName", groupName);

        Region *body = $_state.addRegion();
        body->push_back(new Block());
    }]>
  ];

}

def StaticRepeatOp : CalyxContainer<"static_repeat", [
    ControlLike
  ]> {
  let summary = "Calyx Static Repeat";
  let arguments = (ins
    I32Attr:$count
  );
  let description = [{
    The "calyx.static_repeat" operation represents the repeated execution of
    the control within its region. All control within the region must be static.

    ```mlir
      calyx.static_repeat 10 {
        calyx.enable @G1
        ...
      }
    ```
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = true;
  let assemblyFormat = "$count $body attr-dict";
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins
      "uint32_t":$count), [{
        $_state.addAttribute("count", 
          mlir::IntegerAttr::get(IntegerType::get($_state.getContext(), 32), count)
        );

        Region *body = $_state.addRegion();
        body->push_back(new Block());
    }]>
  ];

}

def RepeatOp : CalyxContainer<"repeat", [
    ControlLike
  ]> {
  let summary = "Calyx Dynamic Repeat";
  let arguments = (ins
    I32Attr:$count
  );
  let description = [{
    The "calyx.repeat" operation represents the repeated execution of
    the control within its region.
    The key difference with static repeat is that the body (unlike with static 
    repeat) can be dynamically timed. 

    ```mlir
      calyx.repeat 10 {
        calyx.enable @G1
        ...
      }
    ```
  }];
  let hasCanonicalizer = true;
  let assemblyFormat = "$count $body attr-dict";
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins
      "uint32_t":$count), [{
        $_state.addAttribute("count", 
          mlir::IntegerAttr::get(IntegerType::get($_state.getContext(), 32), count)
        );

        Region *body = $_state.addRegion();
        body->push_back(new Block());
    }]>
  ];
}

def InvokeOp : CalyxOp<"invoke", [
                        ControlLike,
                        SameVariadicOperandSize
                        ]> {
  let summary = "Calyx Invoke";
  let description = [{
    calyx.invoke is similar to the behavior of a function 
    call, which invokes a given component.

    The 'callee' attribute is the name of the component,
    the 'ports' attribute The ports attribute specifies 
    the input port of the component when it is invoked, 
    and the 'inputs' attribute specifies the assignment on 
    the corresponding port.

    ```mlir
      %id.in, %id.out, ... = calyx.instance @id of @identity : i32, i32, ...
      %r.in, ... = calyx.register @r : i32, ...
      ...
      calyx.control {
        calyx.seq {
          calyx.invoke @id(%id.in = %c1_10, %r.in = %id.out) -> (i32, i32)
        }
      }
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, 
                   Variadic<AnyType>:$ports,
                   Variadic<AnyType>:$inputs,
                   ArrayAttr:$portNames,
                   ArrayAttr:$inputNames);
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    // Get the go port of the invoked component.
    Value getInstGoValue();
    // Get the done port of the invoked component.
    Value getInstDoneValue();
  }];
}
