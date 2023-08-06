//===- ArcOps.td - Arc dialect operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCOPS_TD
#define CIRCT_DIALECT_ARC_ARCOPS_TD

include "circt/Dialect/Arc/ArcDialect.td"
include "circt/Dialect/Arc/ArcTypes.td"
include "circt/Dialect/Arc/ArcInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class ArcOp<string mnemonic, list<Trait> traits = []> :
  Op<ArcDialect, mnemonic, traits>;

def DefineOp : ArcOp<"define", [
  IsolatedFromAbove,
  FunctionOpInterface,
  CallableOpInterface,
  Symbol,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"arc::OutputOp">,
  HasParent<"mlir::ModuleOp">
]> {
  let summary = "State transfer arc definition";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let hasRegionVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$sym_name,
                   "mlir::TypeAttr":$function_type), [{
      build($_builder, $_state, sym_name, function_type, mlir::ArrayAttr(),
            mlir::ArrayAttr());
    }]>,
    OpBuilder<(ins "mlir::StringRef":$sym_name, 
                   "mlir::FunctionType":$function_type), [{
      build($_builder, $_state, sym_name, function_type, mlir::ArrayAttr(),
            mlir::ArrayAttr());
    }]>,
  ];

  let extraClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::SSACFG;
    }

    mlir::Block &getBodyBlock() { return getBody().front(); }

    // Get the arc's symbolic name.
    mlir::StringAttr getNameAttr() {
      return (*this)->getAttrOfType<mlir::StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the arc's symbolic name.
    mlir::StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    mlir::LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<mlir::FunctionType>())
        return emitOpError("requires '") << getFunctionTypeAttrName() <<
                           "' attribute of function type";
      return mlir::success();
    }

    /// Returns true if the arc returns the inputs directly and in the same
    /// order, otherwise false.
    bool isPassthrough();

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    mlir::Region *getCallableRegion() { return &getBody(); }
    ArrayRef<mlir::Type> getCallableResults() { return getResultTypes(); }
    mlir::ArrayAttr getCallableArgAttrs() { return nullptr; }
    mlir::ArrayAttr getCallableResAttrs() { return nullptr; }
  }];
}

def OutputOp : ArcOp<"output", [
  Terminator,
  ParentOneOf<["DefineOp", "LutOp", "ClockDomainOp"]>,
  Pure,
  ReturnLike
]> {
  let summary = "Arc terminator";
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` qualified(type($outputs)))?
  }];
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
  let hasVerifier = 1;
}

def StateOp : ArcOp<"state", [
  MemRefsNormalizable,
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<ClockedOpInterface, ["isClocked"]>,
]> {
  let summary = "State transfer arc";

  let arguments = (ins
    FlatSymbolRefAttr:$arc,
    Optional<I1>:$clock,
    Optional<I1>:$enable,
    Optional<I1>:$reset,
    I32Attr:$latency,
    Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $arc `(` $inputs `)` (`clock` $clock^)? (`enable` $enable^)?
    (`reset` $reset^)? `lat` $latency attr-dict
    `:` functional-type($inputs, results)
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;

  let builders = [
    OpBuilder<(ins "DefineOp":$arc, "mlir::Value":$clock, "mlir::Value":$enable,
      "unsigned":$latency, CArg<"mlir::ValueRange", "{}">:$inputs), [{
      build($_builder, $_state, mlir::SymbolRefAttr::get(arc),
            arc.getFunctionType().getResults(), clock, enable, latency,
            inputs);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$arc, "mlir::TypeRange":$results,
      "mlir::Value":$clock, "mlir::Value":$enable, "unsigned":$latency,
      CArg<"mlir::ValueRange", "{}">:$inputs
    ), [{
      build($_builder, $_state, arc, results, clock, enable, Value(), latency,
      inputs);
    }]>,
    OpBuilder<(ins "mlir::SymbolRefAttr":$arc, "mlir::TypeRange":$results,
      "mlir::Value":$clock, "mlir::Value":$enable, "mlir::Value":$reset,
      "unsigned":$latency, CArg<"mlir::ValueRange", "{}">:$inputs
    ), [{
      if (clock)
        $_state.addOperands(clock);
      if (enable)
        $_state.addOperands(enable);
      if (reset)
        $_state.addOperands(reset);
      $_state.addOperands(inputs);
      $_state.addAttribute("arc", arc);
      $_state.addAttribute("latency", $_builder.getI32IntegerAttr(latency));
      $_state.addAttribute(getOperandSegmentSizeAttr(),
        $_builder.getDenseI32ArrayAttr({
          clock ? 1 : 0,
          enable ? 1 : 0,
          reset ? 1 : 0,
          static_cast<int32_t>(inputs.size())}));
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "mlir::StringAttr":$arc, "mlir::TypeRange":$results,
      "mlir::Value":$clock, "mlir::Value":$enable, "unsigned":$latency,
      CArg<"mlir::ValueRange", "{}">:$inputs
    ), [{
      build($_builder, $_state, mlir::SymbolRefAttr::get(arc), results, clock,
            enable, latency, inputs);
    }]>,
    OpBuilder<(ins "mlir::StringRef":$arc, "mlir::TypeRange":$results,
      "mlir::Value":$clock, "mlir::Value":$enable, "unsigned":$latency,
      CArg<"mlir::ValueRange", "{}">:$inputs
    ), [{
      build($_builder, $_state,
            mlir::StringAttr::get($_builder.getContext(), arc),
            results, clock, enable, latency, inputs);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getInputs();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("arc");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getArcAttrName(), callee.get<mlir::SymbolRefAttr>());
    }
  }];
}

def CallOp : ArcOp<"call", [
  MemRefsNormalizable, Pure,
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "calls an arc";

  let arguments = (ins FlatSymbolRefAttr:$arc, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $arc `(` $inputs `)` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return {operand_begin(), operand_end()};
    }
    MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("arc");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getArcAttrName(), callee.get<mlir::SymbolRefAttr>());
    }
  }];
}

def ClockGateOp : ArcOp<"clock_gate", [Pure]> {
  let summary = "Clock gate";
  let arguments = (ins I1:$input, I1:$enable);
  let results = (outs I1:$output);
  let assemblyFormat = [{
    $input `,` $enable attr-dict
  }];
}

def MemoryOp : ArcOp<"memory", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Memory";
  let results = (outs MemoryType:$memory);
  let assemblyFormat = "type($memory) attr-dict";
}

class MemoryAndDataTypesMatch<string mem, string data> : TypesMatchWith<
  "memory and data types must match", mem, data,
  "$_self.cast<MemoryType>().getWordType()">;
class MemoryAndAddressTypesMatch<string mem, string address> : TypesMatchWith<
  "memory and address types must match", mem, address,
  "$_self.cast<MemoryType>().getAddressType()">;

def MemoryReadPortOp : ArcOp<"memory_read_port", [
  Pure,
  MemoryAndDataTypesMatch<"memory", "data">,
  MemoryAndAddressTypesMatch<"memory", "address">
]> {
  let summary = "Read port from a memory";
  let description = [{
    Represents a combinatorial memory read port. No memory read side-effect
    trait is necessary because at the stage of the Arc lowering where this
    operation is legal to be present, it is guaranteed that all reads from the
    same address produce the same output. This is because all writes are
    reordered to happen at the end of the cycle in LegalizeStateUpdates (or
    alternatively produce the necessary temporaries).
  }];
  let arguments = (ins
    MemoryType:$memory,
    AnyInteger:$address
  );
  let results = (outs AnyInteger:$data);

  let assemblyFormat = [{
    $memory `[` $address `]` attr-dict `:` type($memory)
  }];
}

def MemoryWritePortOp : ArcOp<"memory_write_port", [
  MemoryEffects<[MemWrite]>,
  CallOpInterface,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  AttrSizedOperandSegments,
  ClockedOpInterface
]> {
  let summary = "Write port to a memory";
  let arguments = (ins
    MemoryType:$memory,
    FlatSymbolRefAttr:$arc,
    Variadic<AnyType>:$inputs,
    Optional<I1>:$clock,
    UnitAttr:$enable,
    UnitAttr:$mask,
    DefaultValuedAttr<I32Attr, "1">:$latency
  );

  let assemblyFormat = [{
    $memory `,` $arc  `(` $inputs `)` (`clock` $clock^)?  (`enable` $enable^)?
    (`mask` $mask^)? `lat` $latency attr-dict `:`
    type($memory) `,` type($inputs)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    SmallVector<Type> getArcResultTypes();
    static unsigned getAddressIdx() { return 0; }
    static unsigned getDataIdx() { return 1; }
    static unsigned getEnableIdx() { return 2; }
    static unsigned getMaskIdx(bool hasEnable) { return hasEnable ? 3 : 2; }

    operand_range getArgOperands() {
      return getInputs();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("arc");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getArcAttrName(), callee.get<mlir::SymbolRefAttr>());
    }
  }];
}

def MemoryReadOp : ArcOp<"memory_read", [
  MemoryEffects<[MemRead]>,
  MemoryAndDataTypesMatch<"memory", "data">,
  MemoryAndAddressTypesMatch<"memory", "address">
]> {
  let summary = "Read word from memory";
  let arguments = (ins
    MemoryType:$memory,
    AnyInteger:$address
  );
  let results = (outs AnyInteger:$data);

  let assemblyFormat = [{
    $memory `[` $address `]` attr-dict `:` type($memory)
  }];
}

def MemoryWriteOp : ArcOp<"memory_write", [
  MemoryEffects<[MemWrite]>,
  MemoryAndDataTypesMatch<"memory", "data">,
  MemoryAndAddressTypesMatch<"memory", "address">
]> {
  let summary = "Write word to memory";
  let arguments = (ins
    MemoryType:$memory,
    AnyInteger:$address,
    Optional<I1>:$enable,
    AnyInteger:$data
  );

  let assemblyFormat = [{
    $memory `[` $address `]` `,` $data (`if` $enable^)?
    attr-dict `:` type($memory)
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Trigger Grouping
//===----------------------------------------------------------------------===//

def ClockDomainOp : ArcOp<"clock_domain", [
  IsolatedFromAbove,
  RegionKindInterface,
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"arc::OutputOp">
]> {
  let summary = "a clock domain";

  let arguments = (ins Variadic<AnyType>:$inputs, I1:$clock);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    ` ` `(` $inputs `)` `clock` $clock attr-dict `:`
    functional-type($inputs, results) $body
  }];

  let extraClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }
    mlir::Block &getBodyBlock() { return getBody().front(); }
  }];

  let hasRegionVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def ClockTreeOp : ArcOp<"clock_tree", [NoTerminator, NoRegionArguments]> {
  let summary = "A clock tree";
  let arguments = (ins I1:$clock);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $clock attr-dict-with-keyword $body
  }];
  let extraClassDeclaration = [{
    mlir::Block &getBodyBlock() { return getBody().front(); }
  }];
}

def PassThroughOp : ArcOp<"passthrough", [NoTerminator, NoRegionArguments]> {
  let summary = "Clock-less logic that is on the pass-through path";
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    attr-dict-with-keyword $body
  }];
  let extraClassDeclaration = [{
    mlir::Block &getBodyBlock() { return getBody().front(); }
  }];
}

//===----------------------------------------------------------------------===//
// Storage Allocation
//===----------------------------------------------------------------------===//

def AllocStateOp : ArcOp<"alloc_state", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate internal state";
  let arguments = (ins StorageType:$storage, UnitAttr:$tap);
  let results = (outs StateType:$state);
  let assemblyFormat = [{
    $storage (`tap` $tap^)? attr-dict `:` functional-type($storage, $state)
  }];
}

def AllocMemoryOp : ArcOp<"alloc_memory", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a memory";
  let arguments = (ins StorageType:$storage);
  let results = (outs MemoryType:$memory);
  let assemblyFormat = [{
    $storage attr-dict `:` functional-type($storage, $memory)
  }];
}

def AllocStorageOp : ArcOp<"alloc_storage", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate contiguous storage space from a larger storage space";
  let arguments = (ins StorageType:$input, OptionalAttr<I32Attr>:$offset);
  let results = (outs StorageType:$output);
  let assemblyFormat = [{
    $input (`[` $offset^ `]`)? attr-dict `:` functional-type($input, $output)
  }];
}

def RootInputOp : ArcOp<"root_input", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "A root input";
  let arguments = (ins StrAttr:$name, StorageType:$storage);
  let results = (outs StateType:$state);
  let assemblyFormat = [{
    $name `,` $storage attr-dict `:` functional-type($storage, $state)
  }];
}

def RootOutputOp : ArcOp<"root_output", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "A root output";
  let arguments = (ins StrAttr:$name, StorageType:$storage);
  let results = (outs StateType:$state);
  let assemblyFormat = [{
    $name `,` $storage attr-dict `:` functional-type($storage, $state)
  }];
}

//===----------------------------------------------------------------------===//
// Storage Access
//===----------------------------------------------------------------------===//

def AllocatableType : AnyTypeOf<[StateType, MemoryType, StorageType]>;

def StorageGetOp : ArcOp<"storage.get", [Pure]> {
  let summary = "Access an allocated state, memory, or storage slice";
  let arguments = (ins StorageType:$storage, I32Attr:$offset);
  let results = (outs AllocatableType:$result);
  let assemblyFormat = [{
    $storage `[` $offset `]` attr-dict
    `:` qualified(type($storage)) `->` type($result)
  }];
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// State Read/Write
//===----------------------------------------------------------------------===//

class StateAndValueTypesMatch<string state, string value> : TypesMatchWith<
  "state and value types must match", state, value,
  "$_self.cast<StateType>().getType()">;

def StateReadOp : ArcOp<"state_read", [
  MemoryEffects<[MemRead]>,
  StateAndValueTypesMatch<"state", "value">
]> {
  let summary = "Get a state's current value";
  let arguments = (ins StateType:$state);
  let results = (outs AnyInteger:$value);
  let assemblyFormat = [{
    $state attr-dict `:` type($state)
  }];
}

def StateWriteOp : ArcOp<"state_write", [
  MemoryEffects<[MemWrite]>,
  StateAndValueTypesMatch<"state", "value">
]> {
  let summary = "Update a state's value";
  let description = [{
    Changes the value of a state. This operation is treated as a deferred
    assignment by most transformation passes, which allows them to change the
    order of `arc.state_read` and `arc.state_write` ops on the same state
    without affecting the correctness of the model. The reads are always assumed
    to produce the current value of the state and writes to be deferred until
    all operations in the model have been executed for the current time step.

    The only exceptions to this are the state update legalization pass, which
    inserts the necessary temporary variables such that writes can be performed
    immediately without affecting correctness. This allows later lowering passes
    to treat `arc.state_write` as an immediate assignment (without defering).
  }];
  let arguments = (ins StateType:$state, AnyInteger:$value,
                       Optional<I1>:$condition);
  let assemblyFormat = [{
    $state `=` $value (`if` $condition^)? attr-dict `:` type($state)
  }];
}

//===----------------------------------------------------------------------===//
// Miscellaneous
//===----------------------------------------------------------------------===//

def TapOp : ArcOp<"tap"> {
  let summary = "A tracker op to observe a value under a given name";
  let arguments = (ins AnySignlessInteger:$value, StrAttr:$name);
  let assemblyFormat = [{ $value attr-dict `:` type($value) }];
}

def ModelOp : ArcOp<"model", [RegionKindInterface, IsolatedFromAbove,
                              NoTerminator]> {
  let summary = "A model with stratified clocks";
  let arguments = (ins StrAttr:$name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $name attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }
    mlir::Block &getBodyBlock() { return getBody().front(); }
  }];

  let hasVerifier = 1;
}

def LutOp : ArcOp<"lut", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"arc::OutputOp">,
  Pure
]> {
  let summary = "A lookup-table.";
  let description = [{
    Represents a lookup-table as one operation. The operations that map the
    lookup/input values to the corresponding table-entry are collected inside
    the body of this operation.
    Note that the operation is marked to be isolated from above to guarantee
    that all input values have to be passed as an operand. This allows for
    simpler analyses and canonicalizations of the LUT as well as lowering.
    Only combinational operations are allowed inside the LUT, i.e., no
    side-effects, state, time delays, etc.
  }];

  let arguments = (ins Variadic<AnySignlessInteger>:$inputs);
  let results = (outs AnySignlessInteger:$output);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(` $inputs `)` `:` functional-type($inputs, $output)
    attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    mlir::Block *getBodyBlock() { return &getBody().front(); }
  }];

  let hasVerifier = 1;
}

def ZeroCountPredicateLeading  : I32EnumAttrCase<"leading", 0>;
def ZeroCountPredicateTrailing  : I32EnumAttrCase<"trailing", 1>;
def ZeroCountPredicate : I32EnumAttr<
    "ZeroCountPredicate", "arc.zero_count predicate",
    [ZeroCountPredicateLeading, ZeroCountPredicateTrailing]> {
  let cppNamespace = "circt::arc";
}

def ZeroCountOp : ArcOp<"zero_count", [Pure, SameOperandsAndResultType]> {
  let summary = "leading/trailing zero count operation";
  let arguments = (ins AnySignlessInteger:$input,
                       ZeroCountPredicate:$predicate);
  let results = (outs AnySignlessInteger:$output);
  let assemblyFormat = "$predicate $input attr-dict `:` type($input)";
}

#endif // CIRCT_DIALECT_ARC_ARCOPS_TD
