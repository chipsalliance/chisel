//===- Interop.td - Interop dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Interop dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_INTEROP_INTEROP
#define CIRCT_DIALECT_INTEROP_INTEROP

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Interop Dialect
//===----------------------------------------------------------------------===//

def InteropDialect : Dialect {
  let name = "interop";

  let summary = "Provides interoperability between backends and tools";
  let description = [{
    This dialect defines the `interop` dialect which defines operations and
    interfaces necessary to provide interoperability between backends and
    and external tools without the need of writing custom pairwise interop
    solutions.
  }];
  let cppNamespace = "::circt::interop";
}

//===----------------------------------------------------------------------===//
// Interop Enums
//===----------------------------------------------------------------------===//

// C foreign function interface
def CFFI : I32EnumAttrCase<"CFFI", 0, "cffi">;
// Textual C++
def CPP : I32EnumAttrCase<"CPP", 1, "cpp">;

def InteropMechanismAttr : I32EnumAttr<
  "InteropMechanism",
  "interface through which interoperability is achieved",
  [CFFI, CPP]> {}

//===----------------------------------------------------------------------===//
// Interop Operations
//===----------------------------------------------------------------------===//

// Base class for the operations in this dialect.
class InteropOp<string mnemonic, list<Trait> traits = []> :
  Op<InteropDialect, mnemonic, traits>;

def ProceduralAllocOp : InteropOp<"procedural.alloc", []> {
  let summary = "represents persistent state to be allocated";
  let description = [{
    The `interop.procedural.alloc` operation returns a variadic list of values
    that represent persistent state, i.e., state that has to persist across
    multiple executions of the `interop.procedural.update` operation.
    For example, it can be lowered to C++ class fields that are persistent
    across multiple calls of a member function, or to global simulator state
    that persists over simulation cycles, etc.

    Additionally, it has an attribute that specifies the interop mechanism under
    which the state types are valid. This is necessary to allow bridging patterns
    to map the types to valid types in the other interop mechanism, e.g., to an
    opaque pointer, if it does not support the same types.
  }];

  let arguments = (ins InteropMechanismAttr:$interopMechanism);
  let results = (outs Variadic<AnyType>:$states);

  let assemblyFormat = [{
    $interopMechanism ( `:` qualified(type($states))^ )? attr-dict
  }];
}

def ProceduralInitOp : InteropOp<"procedural.init", [SingleBlock]> {
  let summary = "computes the initial values for the allocated state";
  let description = [{
    The `interop.procedural.init` operation takes the variadic list of states
    from the `interop.procedural.alloc` operation as operands and has a body
    with a `interop.return` operation that has a variadic list of operands that
    matches the types of the states and represent the initial values to be
    assigned to the state values.
    The assignment will be inserted by the container-side lowering of the
    interop operations.
    The operation also has an interop mechanism attribute to allow bridging
    patterns to map the types to valid types in another interop mechanism
    and to wrap the operations in the body in a way to make them executable
    in the other interop mechanism, e.g., wrap them in a `extern "C"` function
    to make it callable from C or LLVM IR.
  }];

  let arguments = (ins InteropMechanismAttr:$interopMechanism,
                       Variadic<AnyType>:$states);
  let regions = (region SizedRegion<1>:$initRegion);

  let assemblyFormat = [{
    $interopMechanism ( $states^ `:` qualified(type($states)) )?
    attr-dict-with-keyword $initRegion
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$states, "InteropMechanism":$interopMechanism)>
  ];

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
}

def ProceduralUpdateOp : InteropOp<"procedural.update", [
  SingleBlock, AttrSizedOperandSegments
]> {
  let summary = "takes some persistent state and inputs to compute some results";
  let description = [{
    The `interop.procedural.update` operation has an interop mechanism attribute
    to allow bridging patterns to map the types to valid types in another
    interop mechanism and to wrap the operations in the body in a way to make
    them executable using the other interop mechanism.

    It takes the state values returned by the `interop.procedural.alloc`as
    operands and passes them on to the body via block arguments using
    pass-by-value semantics. In addition to the state values, it also takes a
    variadic list of inputs and also passes them on to the body.
    The `interop.return` inside the body then returns the result values after
    doing some computation inside the body.

    If the state needs to be mutated, it has to be a pointer type.
  }];

  let arguments = (ins InteropMechanismAttr:$interopMechanism,
                       Variadic<AnyType>:$states,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$updateRegion);

  let assemblyFormat = [{
    $interopMechanism ( ` ` `[` $states^ `]` )? ( `(` $inputs^ `)` )? `:`
    (`[` qualified(type($states))^ `]`)? functional-type($inputs, $outputs)
    attr-dict-with-keyword $updateRegion
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$outputs, "ValueRange":$inputs,
                   "ValueRange":$states,
                   "InteropMechanism":$interopMechanism)>,
  ];

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
}

def ProceduralDeallocOp : InteropOp<"procedural.dealloc", [
  SingleBlock, NoTerminator
]> {
  let summary = "performs some deallocation logic before the state is released";
  let description = [{
    The `interop.procedural.dealloc` operation shall be executed right before
    the state requested by the `interop.procedural.alloc` operation is
    released. This allows the instance to do some cleanup, e.g., when the state
    type was a pointer and the instance performed some `malloc`.

    Structurally the operation is the same as the `interop.procedural.update`
    operation, but without input and output values. The state is also passed
    by value.
  }];

  let arguments = (ins InteropMechanismAttr:$interopMechanism,
                       Variadic<AnyType>:$states);
  let regions = (region SizedRegion<1>:$deallocRegion);

  let assemblyFormat = [{
    $interopMechanism ( $states^ `:` qualified(type($states)) )?
    attr-dict-with-keyword $deallocRegion
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$states,
                   "InteropMechanism":$interopMechanism)>,
  ];

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
}

def ReturnOp : InteropOp<"return", [
  ParentOneOf<["ProceduralInitOp",
               "ProceduralUpdateOp"]>,
  ReturnLike, Terminator
]> {
  let summary = "a return operation";
  let description = [{
    The `interop.return` operation lists the computed initial values when
    inside the `init` operation or the computed results when inside the
    `update` operation.
  }];

  let arguments = (ins Variadic<AnyType>:$returnValues);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($returnValues^ `:` type($returnValues))?";
  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_INTEROP_INTEROP
