//===- StructureOps.td - Process and Entity definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the LLHD Process, Entity and control flow MLIR ops.
//
//===----------------------------------------------------------------------===//

def LLHD_EntityOp : LLHD_Op<"entity", [
    Symbol,
    FunctionOpInterface,
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<CallableOpInterface>
  ]> {
  let summary = "Create an entity.";
  let description = [{
    The `llhd.entity` operation defines a new entity unit. An entity
    represents the data-flow description of how a circuit's output values
    change in reaction to changing input values.
    An entity contains one region with a single block and an implicit
    `TerminatorOp` terminator. Both the block name and terminator are
    omitted in the custom syntax. No further blocks and control-flow are
    legal inside an entity.

    Syntax:

    ```
    entity-op ::= `llhd.entity` entity-symbol `(` arg-list `)` `->`
      `(` out-list `)` attr-dict entity-region
    ```

    Example:

    ```mlir
    llhd.entity @Foo () -> () {
      %0 = llhd.const 0 : i1
      %toggle = llhd.sig %0 : i1 -> !llhd.sig<i1>
      %1 = llhd.prb %toggle : !llhd.sig<i1> -> i1
      %2 = llhd.not %1 : i1
      %dt = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
      llhd.drv %toggle, %2, %dt : !llhd.sig<i1>, i1, !llhd.time
    }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    I64Attr: $ins,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region SizedRegion<1>: $body);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();

    // Use FunctionOpInterface traits's getFunctionBody method.
    using mlir::detail::FunctionOpInterfaceTrait<EntityOp>::getFunctionBody;

    /// Return the block corresponding to the region.
    Block *getBodyBlock() { return &getBody().front(); }
  }];
}

def LLHD_ProcOp : LLHD_Op<"proc", [
    Symbol,
    FunctionOpInterface,
    IsolatedFromAbove,
    DeclareOpInterfaceMethods<CallableOpInterface>
  ]> {
  let summary = "Create a process";
  let description = [{
    A `llhd.proc` represents control-flow in a timed fashion. It allows a
    procedural description of how a circuit's output signals change in
    reaction to changing input signals. It has a region with arbitrarily
    many basic blocks. The first block is the entry block and cannot be
    targeted by the terminators. It uses `llhd.wait` as a terminator to add
    timed control-flow. Immediate control-flow with `br` or `cond_br` is
    also possible. Every process must either contain an infinite loop or
    terminate with the `llhd.halt` terminator.

    How does a process compare to functions and entities?

    | Unit     | Paradigm     | Timing    | Models                         |
    |----------|--------------|-----------|--------------------------------|
    | Function | control-flow | immediate | Computation in zero time       |
    | Process  | control-flow | timed     | Behavioral circuit description |
    | Entity   | data-flow    | timed     | Structural circuit description |

    Syntax:

    ```
    proc-op ::= `llhd.proc` proc-symbol `(` ssa-input-list `)` `->`
      `(` ssa-output-list `)` attr-dict `{` proc-region `}`
    ```

    Examples:

    ```mlir
    llhd.proc @example(%in0 : !llhd.sig<i64>, %in1 : !llhd.sig<i1>) ->
        (%out2 : !llhd.sig<i1>) {
      br ^bb1
    ^bb1:
      llhd.halt
    }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    I64Attr:$ins,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion: $body);

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
//=== Process and Entity Structure
//===----------------------------------------------------------------------===//

def LLHD_InstOp : LLHD_Op<"inst", [
    CallOpInterface,
    HasParent<"EntityOp">,
    AttrSizedOperandSegments
  ]> {
  let summary = "Instantiates a process or entity.";
  let description = [{
    Instantiates a process or entity and thus allows to build hierarchies.
    Can only be used within an entity. An instance defines a unique name
    within the entity it resides in.

    Syntax:

    ```
    inst-op ::= `llhd.inst` inst-name symbol-name `(` ssa-input-list `)` `->`
      `(` ssa-output-list `)` attr-dict `:`
      functional-type(ssa-input-list, ssa-output-list)
    ```

    Example:

    ```mlir
    llhd.inst "foo" @empty() -> () : () -> ()
    llhd.inst "bar" @proc_symbol() -> (%out0) : () -> !llhd.sig<i32>
    llhd.inst "baz" @entity_symbol(%in0, %in1) -> (%out0, %out1) :
      (!llhd.sig<i32>, !llhd.sig<i16>) -> (!llhd.sig<i8>, !llhd.sig<i4>)
    ```
  }];

  let arguments = (ins StrAttr:$name,
                       FlatSymbolRefAttr:$callee,
                       Variadic<LLHD_AnySigType>:$inputs,
                       Variadic<LLHD_AnySigType>:$outputs);

  let assemblyFormat = [{
    $name $callee `(` $inputs `)` `->` `(` $outputs `)` attr-dict `:`
    functional-type($inputs, $outputs)
  }];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(), callee.get<mlir::SymbolRefAttr>());
    }
  }];

  let hasVerifier = 1;
}

def LLHD_ConnectOp : LLHD_Op<"con", [
    SameTypeOperands,
    HasParent<"EntityOp">
  ]> {
  let summary = "Connect two signals.";
  let description = [{
    The `con` instruction connects two signals such that they essentially become
    one signal. All driven values on one signal will be reflected on the other.
  }];

  let arguments = (ins LLHD_AnySigType:$lhs,
                       LLHD_AnySigType:$rhs);

  let assemblyFormat = [{
    operands attr-dict `:` qualified(type($lhs))
  }];

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
//=== Control Flow Operations
//===----------------------------------------------------------------------===//

def LLHD_WaitOp : LLHD_Op<"wait", [
    Terminator,
    AttrSizedOperandSegments,
    HasParent<"ProcOp">,
    DeclareOpInterfaceMethods<BranchOpInterface>
  ]> {
  let summary = "Suspends execution of a process.";
  let description = [{
    The `wait` instruction suspends execution of a process until any of the
    observed signals change or a fixed time interval has passed. Execution
    resumes at the specified basic block with the passed arguments.
    * This is a terminator instruction.
    * This instruction is only allowed in processes (`llhd.proc`).

    Example:

    ```mlir
    llhd.wait ^bb1
    llhd.wait for %time, ^bb1(%time : !llhd.time)
    llhd.wait (%0, %1 : !llhd.sig<i64>, !llhd.sig<i1>), ^bb1(%1 : !llhd.sig<i1>)
    llhd.wait for %time, (%0, %1 : !llhd.sig<i64>, !llhd.sig<i1>),
      ^bb1(%1, %0 : !llhd.sig<i1>, !llhd.sig<i64>)
    ```
  }];

  let arguments = (ins Variadic<LLHD_AnySigType>:$obs,
                       Optional<LLHD_TimeType>:$time,
                       Variadic<AnyType>:$destOps);

  let successors = (successor AnySuccessor:$dest);

  let assemblyFormat = [{
    (`for` $time^ `,`)? (`(`$obs^ `:` qualified(type($obs))`)` `,`)?
    $dest (`(` $destOps^ `:` qualified(type($destOps)) `)`)? attr-dict
  }];
}

def LLHD_HaltOp : LLHD_Op<"halt", [Terminator, HasParent<"ProcOp">]> {
  let summary = "Terminates execution of a process.";
  let description = [{
    The `halt` instruction terminates execution of a process. All processes
    must halt eventually or consist of an infinite loop.

    * This is a terminator instruction
    * This instruction is only allowed in processes (`llhd.proc`).

    Syntax:

    ```
    halt-op ::= `llhd.halt`
    ```

    Example:

    ```mlir
    llhd.halt
    ```
  }];

  let assemblyFormat = "attr-dict";
}
