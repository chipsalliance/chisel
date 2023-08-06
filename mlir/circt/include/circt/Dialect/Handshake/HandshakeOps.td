//===- Ops.td - Handshake operation definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//============================================================================//
//
// This file define Handshake ops in tablegen.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// This is almost exactly like a standard FuncOp, except that it has some
// extra verification conditions.  In particular, each Value must
// only have a single use.  Also, it defines a Dominance-Free Scope
def FuncOp : Op<Handshake_Dialect, "func", [
   IsolatedFromAbove,
   FunctionOpInterface,
   Symbol,
   RegionKindInterface,
   OpAsmOpInterface,
   HasClock
]> {
  let summary = "Handshake dialect function.";
  let description = [{
    The func operation represents a handshaked function.
    This is almost exactly like a standard FuncOp, except that it has
    some extra verification conditions. In particular, each Value must
    only have a single use.
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs);
  let regions = (region AnyRegion : $body);

  let skipDefaultBuilders = 1;

  let builders =
       [OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock();

    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the names of the arguments to this function.
    ArrayAttr getArgNames() {
      return (*this)->getAttrOfType<ArrayAttr>("argNames");
    }

    /// Returns the names of the results from this function.
    ArrayAttr getResNames() {
      return (*this)->getAttrOfType<ArrayAttr>("resNames");
    }

    /// Returns the argument name at the given index.
    StringAttr getArgName(unsigned idx) {
      return getArgNames()[idx].cast<StringAttr>();
    }

    /// Returns the result name at the given index.
    StringAttr getResName(unsigned idx) {
      return getResNames()[idx].cast<StringAttr>();
    }

    /// Resolve argument and result names. This can be used during building of
    /// a handshake.func operation to ensure that names provided by an incoming
    /// operation fulfills the name requirements of the handshake function.
    void resolveArgAndResNames();

    /// Hook for FunctionOpInterface, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError(
            "requires '" + getFunctionTypeAttrName().getValue() +
            "' attribute of function type");
      return success();
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static ::llvm::StringRef getDefaultDialect() {
      return "handshake";
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// InstanceOp
def InstanceOp : Handshake_Op<"instance", [
    CallOpInterface,
    HasClock,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "module instantiate operation";
  let description = [{
    The `instance` operation represents the instantiation of a module.  This
    is similar to a function call, except that different instances of the
    same module are guaranteed to have their own distinct state.
    The instantiated module is encoded as a symbol reference attribute named
    "module". An instance operation takes a control input as its last argument
    and returns a control output as its last result.

    Example:
    ```mlir
    %2:2 = handshake.instance @my_add(%0, %1, %ctrl) : (f32, f32, none) -> (f32, none)
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$module, Variadic<AnyType>:$opOperands);
  let results = (outs Variadic<AnyType>);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    (ins "FuncOp":$module, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("module", SymbolRefAttr::get(module));
      $_state.addTypes(module.getResultTypes());
  }]>, OpBuilder<
    (ins "SymbolRefAttr":$module, "TypeRange":$results,
     CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("module", module);
      $_state.addTypes(results);
      $_state.addTypes({$_builder.getType<::mlir::NoneType>()});
  }]>, OpBuilder<
    (ins "StringRef":$module, "TypeRange":$results,
     CArg<"ValueRange", "{}">:$operands), [{
       build($_builder, $_state,
             SymbolRefAttr::get($_builder.getContext(), module), results,
             operands);
  }]>];

  let extraClassDeclaration = [{
    FunctionType getModuleType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the module of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("module");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getModuleAttrName(), callee.get<mlir::SymbolRefAttr>());
    }

    /// Get the control operand of this instance op
    Value getControl() {
      return getOperands().back();
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOpOperandsMutable();
    }

  }];

  let assemblyFormat = [{
    $module `(` $opOperands `)` attr-dict `:` functional-type($opOperands, results)
  }];
}

// This is almost exactly like a standard FuncOp, except that it has some
// extra verification conditions.  In particular, each Value must
// only have a single use.
def ReturnOp : Handshake_Op<"return", [Terminator]> {
  let summary = "Handshake dialect return.";
  let description = [{
    The return operation represents a handshaked
    function.  This is almost exactly like a standard ReturnOp, except
    that it exists in a handshake.func.  It has the same operands as
    standard ReturnOp which it replaces and an additional control -
    only operand(exit point of control - only network).
  }];

  let arguments = (ins Variadic<AnyType> : $opOperands);

  let hasVerifier = 1;
  let assemblyFormat =
    [{  attr-dict ($opOperands^ `:` type($opOperands))? }];
}

// Here use I32EnumAttr to better suit the BufferOp sequential
// attribute for it can only be seq or fifo.
def BufferCaseSeq: I32EnumAttrCase<"seq", 0>;
def BufferCaseFIFO: I32EnumAttrCase<"fifo", 1>;

def BufferTypeEnum: I32EnumAttr<
  "BufferTypeEnum", "BufferOp seq or fifo", [BufferCaseSeq, BufferCaseFIFO]> {
  let genSpecializedAttr = 0;
}
def BufferTypeEnumAttr: EnumAttr<Handshake_Dialect, BufferTypeEnum, "buffer_type_enum">;

def BufferOp : Handshake_Op<"buffer", [Pure, HasClock, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<SOSTInterface, ["getSize"]>,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>]> {
  let summary = "buffer operation";

  let description = [{
    The buffer operation represents a buffer operation. $slots
    must be an unsigned integer larger than 0. $bufferType=BufferTypeEnum::seq indicates a
    nontransparent buffer, while $bufferType=BufferTypeEnum::fifo indicates a transparent
    buffer.

    An 'initValues' attribute containing a list of integer values may be provided.
    The list must be of the same length as the number of slots. This will
    initialize the buffer with the given values upon reset.
    For now, only sequential buffers are allowed to have initial values.
    @todo: How to support different init types? these have to be stored (and
    retrieved) as attributes, hence they must be of a known type.
  }];

  let arguments = (ins
    AnyType:$operand,
    ConfinedAttr<I32Attr, [IntMinValue<1>]>:$slots,
    BufferTypeEnumAttr:$bufferType,
    OptionalAttr<I64ArrayAttr>:$initValues);
  let results = (outs AnyType:$result);

  let builders = [OpBuilder<
      (ins "Value":$operand, "int":$slots, "BufferTypeEnum":$bufferType), [{
        $_state.addOperands(operand);
        $_state.addTypes(operand.getType());
        $_state.addAttribute("slots", $_builder.getI32IntegerAttr(slots));
        $_state.addAttribute("bufferType", BufferTypeEnumAttr::get($_state.getContext(), bufferType));
    }]>];

  let extraClassDeclaration = [{
    bool isSequential() {
      return this->getBufferType() == BufferTypeEnum::seq;
    }
    int getNumSlots() {
      return (*this)->getAttrOfType<IntegerAttr>("slots").getValue().getZExtValue();
    }
    SmallVector<int64_t> getInitValueArray() {
      assert(getInitValues() && "initValues attribute not set");
      SmallVector<int64_t> values;
      for (auto value : (*this)->getAttrOfType<ArrayAttr>("initValues"))
        values.push_back(value.cast<IntegerAttr>().getValue().getSExtValue());
      return values;
    }

  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def ForkOp : Handshake_Op<"fork", [
  Pure, HasClock, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<SOSTInterface, ["getSize"]>, 
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "fork operation";

  let description = [{
    The fork operation represents a fork operation.  A
    single input is replicated to N outputs and distributed to each
    output as soon as the corresponding successor is available.

    Example:
    ```mlir
    %1:2 = fork [2] %0 : i32
    ```
  }];

  let arguments = (ins AnyType : $operand);
  let results = (outs Variadic<AnyType> : $result);
  
  let builders = [OpBuilder<
    (ins "Value":$operand, "unsigned":$size), [{
      $_state.addOperands(operand);
      SmallVector<Type> resultTypes {size, operand.getType()};
      $_state.addTypes(resultTypes);
  }]>];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
}

def LazyForkOp : Handshake_Op<"lazy_fork", [
  Pure, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<SOSTInterface, ["getSize", "sostIsControl"]>
]> {
  let summary = "lazy fork operation";
  let description = [{
    The lazy_fork operation represents a lazy fork operation.
    A single input is replicated to N outputs and distributed to each
    output when all successors are available.

    Example:
    ```mlir
    %1:2 = lazy_fork [2] %0 : i32
    ```
  }];

  let arguments = (ins AnyType : $operand);
  let results = (outs Variadic<AnyType> : $result);

  let builders = [OpBuilder<
    (ins "Value":$operand, "unsigned":$size), [{
      $_state.addOperands(operand);
      SmallVector<Type> resultTypes {size, operand.getType()};
      $_state.addTypes(resultTypes);
  }]>];

  let hasCustomAssemblyFormat = 1;
}

def MergeOp : Handshake_Op<"merge", [
  Pure, MergeLikeOpInterface, SOSTInterface, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<ExecutableOpInterface>
]> {
  let summary = "merge operation";
  let description = [{
    The merge operation represents a (nondeterministic)
    merge operation. Any input is propagated to the single output. The
    number of inputs corresponds to the number of predecessor
    blocks.

    Example:
    ```
    %0 = merge %a, %b, %c : i32
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$dataOperands);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;
}

def MuxOp : Handshake_Op<"mux", [
  Pure, MergeLikeOpInterface,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>
]> {
  let summary = "mux operation";
  let description = [{
    The mux operation represents a(deterministic) merge operation.
    Operands: select, data0, data1, data2, ...

    The 'select' operand is received from ControlMerge of the same
    block and it represents the index of the data operand that the mux
    should propagate to its single output.  The number of data inputs
    corresponds to the number of predecessor blocks.

    The mux operation is intended solely for control+dataflow selection.
    For purely dataflow selection, use the 'select' operation instead.

    Example:
    ```mlir
    %0 = mux %select [%data0, %data1, %data2] {attributes}: index, i32
    ```
  }];
  let arguments = (ins AnyType : $selectOperand, Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType : $result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def ControlMergeOp : Handshake_Op<"control_merge", [
  Pure, MergeLikeOpInterface, HasClock, SOSTInterface,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "control merge operation";
  let description = [{
    The control_merge operation represents a
    (nondeterministic) control merge.  Any input is propagated to the
    first output and the index of the propagated input is sent to the
    second output.  The number of inputs corresponds to the number of
    predecessor blocks.

    Example:
    ```
    %0, %idx = control_merge %a, %b, %c {attributes} : i32, index
    ```
  }];

  let arguments = (ins Variadic<AnyType> : $dataOperands);
  let results = (outs AnyType : $result, AnyType : $index);
  
  let builders = [OpBuilder<
    (ins "ValueRange":$operands), [{
      assert(!operands.empty() && "cmerge needs at least one operand");
      $_state.addOperands(operands);
      // By default, the index result has an Index type
      $_state.addTypes(ArrayRef<Type>{operands[0].getType(),
                                      $_builder.getIndexType()});
  }]>];
  
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def BranchOp : Handshake_Op<"br", [
  Pure, SameOperandsAndResultType,
  DeclareOpInterfaceMethods<SOSTInterface, ["sostIsControl"]>, 
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  AllTypesMatch<["dataOperand", "dataResult"]>
  ]> {
  let summary = "branch operation";
  let description = [{
      The branch operation represents an unconditional
      branch.  The single data input is propagated to the single
      successor.  The input must be triggered by some predecessor to
      avoid continous triggering of a successor block.

      Example:
      ```mlir
      %1 = br %0 : i32
      ```
  }];
  let arguments = (ins AnyType : $dataOperand);
  let results = (outs AnyType : $dataResult);
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
}

def ConditionalBranchOp : Handshake_Op<"cond_br", [
  Pure, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>,
  TypesMatchWith<"data operand type matches true branch result type",
                    "dataOperand", "trueResult", "$_self">,
  TypesMatchWith<"data operand type matches false branch result type",
                    "dataOperand", "falseResult", "$_self">
]> {
  let summary = "conditional branch operation";
  let description = [{
     The cbranch operation represents a conditional
     branch.  The data input is propagated to one of the two outputs
     based on the condition input.

     Example:
     ```mlir
     %true, %false = conditional_branch %cond, %data : i32
     ```
  }];

  let arguments = (ins I1 : $conditionOperand, AnyType : $dataOperand);
  let results = (outs AnyType : $trueResult, AnyType : $falseResult);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };
  }];
}

def SinkOp : Handshake_Op<"sink", [
  SOSTInterface, DeclareOpInterfaceMethods<ExecutableOpInterface>
]> {
  let summary = "sink operation";
  let description = [{
    The sink operation discards any data that arrives at its
         input.The sink has no successors and it can continuously consume data.

    Example:
    ```mlir
    sink %data : i32
    ```
  }];
  
  let arguments = (ins AnyType:$operand);
  let hasCustomAssemblyFormat = 1;
}

def SourceOp : Handshake_Op<"source", [
  Pure, DeclareOpInterfaceMethods<SOSTInterface, ["getDataType", "getSize"]> 
]> {
  let summary = "source operation";
  let description = [{
    The source operation represents continuous token
    source.  The source continously sets a 'valid' signal which the
    successor can consume at any point in time.
  }];

  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def NeverOp : Handshake_Op<"never", [Pure]> {
  let summary = "never operation";
  let description = [{
    The never operation represents disconnected data
    source. The source never sets any 'valid' signal which will
    never trigger the successor at any point in time.
  }];

  let results = (outs AnyType:$result);
  let assemblyFormat = [{ attr-dict `:` qualified(type($result))}];
}

def ConstantOp : Handshake_Op<"constant", [
  Pure, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>
]> {
  let summary = "constant operation";
  let description = [{
    The const has a constant value. When triggered by its
    single `ctrl` input, it sends the constant value to its single
    successor.

    Example:
    ```mlir
    %0 = constant %ctrl {value = 42 : i32} : i32
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value, NoneType:$ctrl);
  let results = (outs AnyType : $result);
  let hasCanonicalizer = 1;
  let assemblyFormat = [{ $ctrl attr-dict `:` qualified(type($result))}];
  let hasVerifier = 1;
}

def MemRefTypeAttr : TypeAttrBase<"MemRefType", "memref type attribute">;
def MemoryOp : Handshake_Op<"memory", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryOpInterface>, HasClock,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "memory";
  let description = [{
    Each MemoryOp represents an independent memory or memory region (BRAM or external memory).
    It receives memory access requests from load and store operations. For every request,
    it returns data (for load) and a data-less token indicating completion.
    The memory op represents a flat, unidimensional memory.
    Operands: all stores (stdata1, staddr1, stdata2, staddr2, ...), then all loads (ldaddr1, ldaddr2,...)
    Outputs: all load outputs, ordered the same as
    load data (lddata1, lddata2, ...), followed by all none outputs,
    ordered as operands (stnone1, stnone2,...ldnone1, ldnone2,...)
  }];
  let arguments = (ins Variadic<AnyType> : $inputs,
                   I32Attr : $ldCount,
                   I32Attr : $stCount,
                   I32Attr : $id,
                   MemRefTypeAttr : $memRefType,
                   I1Attr : $lsq);
  let results = (outs Variadic<AnyType> : $outputs);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "ValueRange":$operands, "int":$outputs, "int":$control_outputs, "bool":$lsq,
      "int":$id, "Value":$memref)>
  ];

  let extraClassDeclaration = [{
    llvm::SmallVector<handshake::MemLoadInterface> getLoadPorts();
    llvm::SmallVector<handshake::MemStoreInterface> getStorePorts();
  }];

  let assemblyFormat = "`[` `ld` `=` $ldCount `,` `st` `=`  $stCount `]` `(` $inputs `)` attr-dict `:` $memRefType `,` functional-type($inputs, $outputs)";
  let hasVerifier = 1;
}

def ExternalMemoryOp : Handshake_Op<"extmemory", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "external memory";
  let description = [{
    An ExternalMemoryOp represents a wrapper around a memref input to a
    handshake function. The semantics of the load/store operands are identical
    to what is decribed for MemoryOp. The only difference is that the first
    operand to this operand is a `memref` value.
    Upon lowering to FIRRTL, a handshake interface will be created in the
    top-level component for each load- and store which connected to this memory.

    Example:

    ```mlir
    handshake.func @main(%i: index, %v: i32, %mem : memref<10xi32>, %ctrl: none) -> none {
      %stCtrl = extmemory[ld = 0, st = 1](%mem : memref<10xi32>)(%vout, %addr) {id = 0 : i32} : (i32, index) -> (none)
      %vout, %addr = store(%v, %i, %ctrl) : (i32, index, none) -> (i32, index)
      ...
    }
    ```
  }];
  let arguments = (ins AnyMemRef:$memref,
                       Variadic<AnyType>:$inputs,
                       I32Attr : $ldCount,
                       I32Attr : $stCount,
                       I32Attr : $id);
  let results = (outs Variadic<AnyType>:$outputs);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(
      ins "Value":$memref, "ValueRange":$inputs, "int":$ldCount, "int":$stCount,
      "int":$id)>
  ];
  let assemblyFormat = "`[` `ld` `=` $ldCount `,` `st` `=`  $stCount `]` `(` $memref `:` qualified(type($memref)) `)` `(` $inputs `)` attr-dict `:` functional-type($inputs, $outputs)";

  let extraClassDeclaration = [{
    mlir::MemRefType getMemRefType() {
      return getMemref().getType().cast<mlir::MemRefType>();
    }

    llvm::SmallVector<handshake::MemLoadInterface> getLoadPorts();
    llvm::SmallVector<handshake::MemStoreInterface> getStorePorts();
  }];
}

def LoadOp : Handshake_Op<"load", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "load operation";
  let description = [{
     Load memory port, sends load requests to MemoryOp. From dataflow
     predecessor, receives address indices and a control-only value
     which signals completion of all previous memory accesses which
     target the same memory.  When all inputs are received, the load
     sends the address indices to MemoryOp. When the MemoryOp returns
     a piece of data, the load sends it to its dataflow successor.

     Operands: address indices (from predecessor), data (from MemoryOp), control-only input.
     Results: data (to successor), address indices (to MemoryOp).

    Example:
    ```mlir
    %dataToSucc, %addr1ToMem, %addr2ToMem = load [%addr1, %addr2] %dataFromMem, %ctrl : i8, i16, index
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$addresses, AnyType:$data, NoneType:$ctrl);
  let results = (outs AnyType:$dataResult, Variadic<AnyType>:$addressResults);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$memref, "ValueRange":$indices)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def StoreOp : Handshake_Op<"store", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getOperandName"]>,
  DeclareOpInterfaceMethods<NamedIOInterface, ["getResultName"]>
]> {
  let summary = "store operation";
  let description = [{
    Store memory port, sends store requests to MemoryOp. From dataflow
    predecessors, receives address indices, data, and a control-only
    value which signals completion of all previous memory accesses
    which target the same memory.  When all inputs are received, the
    store sends the address and data to MemoryOp.

    Operands: address indices, data, control-only input.
    Results: data and address indices (sent to MemoryOp).
    Types: data type followed by address type.

    Example:
    ```mlir
    %dataToMem, %addrToMem = store [%addr1, %addr2] %dataFromPred , %ctrl : i8, i16, index
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$addresses, AnyType:$data, NoneType:$ctrl);
  let results = (outs AnyType:$dataResult, Variadic<AnyType>:$addressResult);

  let builders =
      [OpBuilder<(ins "Value":$valueToStore, "ValueRange":$indices)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def JoinOp : Handshake_Op<"join", [
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<ControlInterface, ["isControl"]>,
  DeclareOpInterfaceMethods<GeneralOpInterface>
]> {
  let summary = "join operation";
  let description = [{
    A control-only synchronizer.  Produces a valid output when all
    inputs become available.

    Example:
    ```mlir
    %0 = join %a, %b, %c : i32, i1, none
    ```
  }];
  let arguments = (ins Variadic<AnyType> : $data);
  let results = (outs NoneType : $result);
  let hasCustomAssemblyFormat = 1;
}

def SyncOp : Handshake_Op<"sync", [
  HasClock, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  AllTypesMatch<["operands", "results"]>
]> {
  let summary = "sync operation";
  let description = [{
    Synchronizes an arbitrary set of inputs. Synchronization implies applying
    join semantics in between all in- and output ports.

    Example:
    ```mlir
    %aSynced, %bSynced, %cSynced = sync %a, %b, %c : i32, i1, none
    ```
  }];
  let arguments = (ins Variadic<AnyType> : $operands);
  let results = (outs Variadic<AnyType> : $results);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    (ins "ValueRange":$operands), [{
      $_state.addOperands(operands);
      $_state.addTypes(operands.getTypes());
  }]>];
  let assemblyFormat = [{ $operands attr-dict `:` type($operands)}];
}

def I4 : I<4>;
def I4Attr : SignlessIntegerAttrBase<I4, "4-bit integer attribute">;

def UnpackOp : Handshake_Op<"unpack", [
  HasClock, DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  TypesMatchWith<"result types match element types of 'tuple'",
                     "input", "results",
                     "$_self.cast<TupleType>().getTypes()">
]> {
  let summary = "unpacks a tuple";
  let description = [{
    The `unpack` operation assigns each value of a tuple to a separate
    value for further processing. The number of results corresponds
    to the number of tuple elements.
    Similar to `fork`, each output is distributed as soon as the corresponding
    successor is ready.

    Example:
    ```mlir
    %a, %b = handshake.unpack %tuple {attributes} : tuple<i32, i64>
    ```
  }];

  let arguments = (ins Builtin_Tuple:$input);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [OpBuilder<(ins "Value":$input), [{
    $_state.addOperands(input);

    TupleType type = input.getType().dyn_cast_or_null<TupleType>();
    assert(type && "expect unpack to have a 'TupleType' operand");
    $_state.addTypes(type.getTypes());
  }]>];

  let hasCustomAssemblyFormat = 1;
}

def PackOp : Handshake_Op<"pack", [
  DeclareOpInterfaceMethods<ExecutableOpInterface>,
  DeclareOpInterfaceMethods<GeneralOpInterface>,
  TypesMatchWith<"input types match element types of 'tuple'",
                   "result", "inputs",
                   "$_self.cast<TupleType>().getTypes()">
]> {

  let summary = "packs a tuple";
  let description = [{
    The `pack` operation constructs a tuple from separate values.
    The number of operands corresponds to the number of tuple elements.
    Similar to `join`, the output is ready when all inputs are ready.

    Example:
    ```mlir
    %tuple = handshake.pack %a, %b {attributes} : tuple<i32, i64>
    ```
    }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Builtin_Tuple:$result);

  let builders = [OpBuilder<
    (ins "ValueRange":$operands), [{
      $_state.addOperands(operands);
      $_state.addTypes(TupleType::get($_builder.getContext(), operands.getTypes()));
  }]>];

  let hasCustomAssemblyFormat = 1;
}
