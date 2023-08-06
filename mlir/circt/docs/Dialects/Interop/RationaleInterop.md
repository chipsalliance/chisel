# Interoperability Dialect Rationale

This document describes the various design points of the Interop dialect, a
dialect that is used to represent partially lowered interoperability layers and
that provides common interfaces and utilities for automated interop generation.
This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

- [Interoperability Dialect Rationale](#interoperability-dialect-rationale)
  - [Introduction](#introduction)
  - [Procedural Interop](#procedural-interop)
    - [Representing Partial Lowerings](#representing-partial-lowerings)
    - [Interop Mechanisms](#interop-mechanisms)
    - [Instance-side Lowering](#instance-side-lowering)
    - [Container-side Lowering](#container-side-lowering)
    - [Bridging between Interop Mechanisms](#bridging-between-interop-mechanism)
    - [How to use this dialect](#how-to-use-this-dialect)
    - [Design considerations](#design-considerations)

## Introduction

The increasing number of CIRCT-based backends as well as the usage of many
external tools raises questions about interoperability and composition of
those tools. Are we supposed to write interop-layers between them by hand or
can this also be automatically generated? Can we generate interfaces to the
outside in multiple standardized formats/languages automatically?

Let's pick simulation as a concrete example. We have Verilator, CIRCT-based
event-driven (e.g., `llhd-sim`) and potentially non-event-driven simulators
that could be mixed using interop to use the faster, more constraint
one for most of the design and switch to the slower, more general one for
parts of the design that require the generality.
They all provide their own interface for writing testbenches in a way that
suits them best. Verilator has its custom C++ interface as well as SystemC
support, `llhd-sim` expects the testbench to be written in LLHD (e.g., compiled
from SystemVerilog), and nobody knows what the interface of the third would look
like (maybe a custom C-header?). This means, if we want to simulate a design
with all three, we have to write the testbench three times or at least some
layer between the TB and the simulator interfaces. This dialect aims to solve
this issue by auto-generating an interface from a selection of standarized
formats (e.g., a SystemC or SystemVerilog module) such that the testbench has
to be written only once and switching to another simulator just means changing
one CLI argument.
What if we have to use some blackbox Verilog code in the design? Is it compiled
by a (not yet existent) SystemVerilog front-end and simulated by the CIRCT-based
simulators or could it also be useful to simulate the blackbox using Verilator
and everything surrounding it using the CIRCT-based simulator? This dialect can
provide the latter by implementing an interop operation and one lowering per
simulator rather than custom pair-wise interop implementations.

We make a clear distinction between procedural and structural interop because
it is not possible to provide direct interop between a purely structural
language and a purely procedural language. To achieve interop between them,
one has to use at least one bridging language that has (some) support for both
as a translation layer.

## Procedural Interop

The `interop` dialect provides a few operations to represent partially lowered
procedural interop instances presented in the
[next section](#representing-partial-lowerings) as well as an interface to be
implemented by the surrounding module and rewrite patterns to call into those
interface implementations presented in the section about
['Container-side Lowering'](#container-side-lowering).
It also provides common rewrite patterns to insert translation layers between
different interop mechanisms like textural C++, C-foreign-functions, Verilog,
etc. They are discussed in more detail in the sections
['Interop Mechanisms'](#interop-mechanisms) and
['Bridging between Interop Mechanisms'](#bridging-between-interop-mechanisms).

### Representing Partial Lowerings

There are four operations to represent partially lowered procedural interop in
the IR:
* `interop.procedural.alloc`
* `interop.procedural.init`
* `interop.procedural.update`
* `interop.procedural.dealloc`

The `alloc` operation returns a variadic list of values that represent
persistent state, i.e., state that has to persist across multiple executions
of the `update` operation. For example, it can be lowered to C++ class fields
that are persistent across multiple calls of a member function, or to global
simulator state that persists over simulation cycles, etc.

Additionally, it has an attribute that specifies the interop mechanism under
which the state types are valid (the `cpp` in the example below). This is
necessary to allow bridging patterns to map the types to valid types in the
other interop mechanism, e.g., to an opaque pointer, if it does not support
the same types.

```mlir
%state = interop.procedural.alloc cpp : !emitc.ptr<!emitc.opaque<"VBar">>
```

The `init` operation takes the variadic list of states from the `alloc`
operation as operands and has a body with a `return` operation that has a
variadic list of operands that matches the types of the states and
represent the initial values to be assigned to the state values. The assignment
will be inserted by the container-side lowering of the interop operations.
The operation also has an interop mechanism attribute for the same reason as
above and, additionally, to wrap the operations in the body in a way to make
them executable in the other interop mechanism, e.g., wrap them in a
`extern "C"` function to make it callable from C or LLVM IR.

```mlir
interop.procedural.init cpp %state : !emitc.ptr<!emitc.opaque<"VBar">> {
  %0 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"VBar">>
  interop.return %0 : !emitc.ptr<!emitc.opaque<"VBar">>
}
```

The `update` operation is similar to the `alloc` operation in that it has an
interop mechanism attribute for the same reason and takes the state values as
operands, but also passes them on to the body via block arguments using
pass-by-value semantics. In addition to the state values, it also takes a variadic
list of inputs and also passes them on to the body. The `return` inside the body
then returns the result values after doing some computation inside the body.
If the state needs to be mutated, it has to be a pointer type. If we need to be
able to change the actual state value in the future, we could return updated
states via the `return` operation (currently not allowed).

```mlir
%1 = interop.procedural.update cpp [%state](%x, %x) :
  [!emitc.ptr<!emitc.opaque<"VBar">>](i32, i32) -> i32 {
^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>, %arg1: i32, %arg2: i32):
  interop.return %5 : i32
}
```

The `dealloc` operation shall be executed right before the state requested by
the `alloc` operation is released. This allows the instance to do some cleanup,
e.g., when the state type was a pointer and the instance performed some
`malloc`. Structurally the operation is the same as the `update` operation, but
without input and output values. The state is also passed by value.

```mlir
interop.procedural.dealloc cpp %state : !emitc.ptr<!emitc.opaque<"VBar">> {
^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>):
  systemc.cpp.delete %arg0 : !emitc.ptr<!emitc.opaque<"VBar">>
  interop.return
}
```

### Interop Mechanisms

A variety of interop mechanisms can be supported. This allows to perform interop
on different abstraction levels and only fall back to C-foreign-functions when
nothing else is supported.

Currently supported interop mechanisms:
* CFFI: C-foreign-functions, i.e., all interop operations are lowered to C
  functions and function calls.
* CPP: textual C++, i.e., the interop operations are lowered to the `systemc`
  and `emitc` dialects and printed as C++ code. In the future, more dialects
  (such as `scf`) could be supported.

Adding a new interop mechanism requires changes to the `interop` dialect. At
a minimum, the table-gen enum has to be modified and a bridging pattern has to
be added.

### Instance-side Lowering

The instance-side lowering always has to happen before the container-side
lowering since that pass should create the interop operations that will then
be picked up by the container-side lowering pass and properly embedded in the
context of the container operation.

To illustrate how this works, consider a design represented in `hw`, `comb`, and
`seq`. We want to simulate that design using Verilator and provide a SystemC
wrapper (basically what Verilator itself can also do using the SystemC Output
Mode `--sc`). As a first step, the
top-level module has to be cloned without the region and a
`systemc.interop.verilated` operation has to be inserted in the body to
instantiate the previously cloned module (here represented as the extern module
`@Bar`) as a verilated module. The input and output ports get connected 1-1.
The original design is then exported through `ExportVerilog` and verilated,
while our wrapper module is lowered by the instance-side lowering.

```mlir
hw.module.extern @Bar (%a: i32, %b: i32) -> (c: i32)

hw.module @Foo (%x: i32) -> (y: i32) {
  %c = systemc.interop.verilated "inst" @Bar (a: %x: i32, b: %x: i32) -> (c: i32)
  hw.output %c : i32
}
```

As a result, the above code example is lowered to the following code. This is
implemented as a pattern on the `systemc.interop.verilated` operation in the
dialect conversion framework. Note that it is required to provide a rewrite
pattern for this lowering to enable one-shot lowering of all interop operations
in a design during a lowering pipeline. The patterns for all instance-side
interop lowerings of a dialect are provided by a population function
(e.g., `populateInstanceInteropLoweringPatterns(RewritePatternSet&)`) exposed
in their public API. Each dialect should also provide a pass with all its
instance-side lowering patterns populated for partial interop lowering.

```mlir
hw.module @Foo(%x: i32) -> (y: i32) {
  %state = interop.procedural.alloc cpp : !emitc.ptr<!emitc.opaque<"VBar">>
  interop.procedural.init cpp %state : !emitc.ptr<!emitc.opaque<"VBar">> {
    %2 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"VBar">>
    interop.return %2 : !emitc.ptr<!emitc.opaque<"VBar">>
  }
  %1 = interop.procedural.update cpp [%state](%x, %x) :
    [!emitc.ptr<!emitc.opaque<"VBar">>](i32, i32) -> i32 {
  ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>, %arg1: i32, %arg2: i32):
    %2 = systemc.cpp.member_access %arg0 arrow "a" :
      (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
    systemc.cpp.assign %2 = %arg1 : i32
    %3 = systemc.cpp.member_access %arg0 arrow "b" :
      (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
    systemc.cpp.assign %3 = %arg2 : i32
    %4 = systemc.cpp.member_access %arg0 arrow "eval" :
      (!emitc.ptr<!emitc.opaque<"VBar">>) -> (() -> ())
    systemc.cpp.call_indirect %4() : () -> ()
    %5 = systemc.cpp.member_access %arg0 arrow "c" :
      (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
    interop.return %5 : i32
  }
  interop.procedural.dealloc cpp %state : !emitc.ptr<!emitc.opaque<"VBar">> {
  ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>):
    systemc.cpp.delete %arg0 : !emitc.ptr<!emitc.opaque<"VBar">>
    interop.return
  }
  hw.output %1 : i32
}
```

In this example, it is possible to do the `HWToSystemC` conversion before or
after the instance-side interop lowering, but that might not be the case in
all situations.

### Container-side Lowering

The container-side interop lowering is slightly more complex than the
instance-side lowering, because it consists of an interface that needs to be
implemented by the container operations (in our example that's the
`systemc.module` operation) and four rewrite patterns that are provided by the
interop dialect and call those interface implementations. Typically, the
interface is implemented as an external model on the container operation, thus
the external model and the four rewrite patterns have to be registered in the
container-side lowering pass.

Similar to the instance-side lowering, each dialect has to provide a function
to register all external models implementing the interop interface of that
dialect as well as a pass that has all those models, the four rewrite patterns,
and all the bridging patterns, provided by the interop dialect, registered.

The `ProceduralContainerInteropOpInterface` provides four function that have
to be implemented:
```cpp
LogicalResult allocState(PatternRewriter&, ProceduralAllocOp, ProceduralAllocOpAdaptor);
LogicalResult initState(PatternRewriter&, ProceduralInitOp, ProceduralInitOpAdaptor);
LogicalResult updateState(PatternRewriter&, ProceduralUpdateOp, ProceduralUpdateOpAdaptor);
LogicalResult deallocState(PatternRewriter&, ProceduralDeallocOp, ProceduralDeallocOpAdaptor);
```

They are responsible for lowering the respective interop operation in a similar
fashion as regular rewrite patterns.

Let's take a look at how the example from the previous section is further
lowered. After the `convert-hw-to-systemc` pass it looks like the following:

```mlir
systemc.module @Foo(%x: !systemc.in<!systemc.uint<32>>,
                    %y: !systemc.out<!systemc.uint<32>>) {
  systemc.ctor {
    systemc.method %update
  }
  %update = systemc.func {
    %0 = systemc.signal.read %x : !systemc.in<!systemc.uint<32>>
    %state = interop.procedural.alloc cpp : !emitc.ptr<!emitc.opaque<"VBar">>
    interop.procedural.init cpp %state : !emitc.ptr<!emitc.opaque<"VBar">> {
      %1 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"VBar">>
      interop.return %1 : !emitc.ptr<!emitc.opaque<"VBar">>
    }
    %2 = systemc.convert %0 : (!systemc.uint<32>) -> i32
    %3 = interop.procedural.update cpp [%state](%2, %2) :
      [!emitc.ptr<!emitc.opaque<"VBar">>](i32, i32) -> i32 {
    ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>, %arg1: i32, %arg2: i32):
      %4 = systemc.cpp.member_access %arg0 arrow "a" :
        (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
      systemc.cpp.assign %4 = %arg1 : i32
      %5 = systemc.cpp.member_access %arg0 arrow "b" :
        (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
      systemc.cpp.assign %5 = %arg2 : i32
      %6 = systemc.cpp.member_access %arg0 arrow "eval" :
        (!emitc.ptr<!emitc.opaque<"VBar">>) -> (() -> ())
      systemc.cpp.call_indirect %6() : () -> ()
      %7 = systemc.cpp.member_access %arg0 arrow "c" :
        (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
      interop.return %7 : i32
    }
    interop.procedural.dealloc cpp %state : !emitc.ptr<!emitc.opaque<"VBar">> {
    ^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>):
      systemc.cpp.delete %arg0 : !emitc.ptr<!emitc.opaque<"VBar">>
      interop.return
    }
    %8 = systemc.convert %3 : (i32) -> !systemc.uint<32>
    systemc.signal.write %y, %8
  }
}
```

Note that the `alloc`, `init`, and `dealloc` interop operations were not moved
to the final locations yet, although, the pass could be modified to do this.
In the SystemC lowering, the interop interface implementation performs the
movement of these operation to the final location. Doing the lowering leads to
the following:

```mlir
systemc.module @Foo(%x: !systemc.in<!systemc.uint<32>>,
                    %y: !systemc.out<!systemc.uint<32>>) {
  %state = systemc.cpp.variable : !emitc.ptr<!emitc.opaque<"VBar">>
  systemc.ctor {
    %0 = systemc.cpp.new() : () -> !emitc.ptr<!emitc.opaque<"VBar">>
    systemc.cpp.assign %state = %0 : !emitc.ptr<!emitc.opaque<"VBar">>
    systemc.method %update
  }
  %update = systemc.func {
    %1 = systemc.signal.read %x : !systemc.in<!systemc.uint<32>>
    %2 = systemc.convert %1 : (!systemc.uint<32>) -> i32
    %3 = systemc.cpp.member_access %state arrow "a" :
      (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
    systemc.cpp.assign %3 = %2 : i32
    %4 = systemc.cpp.member_access %state arrow "b" :
      (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
    systemc.cpp.assign %4 = %2 : i32
    %5 = systemc.cpp.member_access %state arrow "eval" :
      (!emitc.ptr<!emitc.opaque<"VBar">>) -> (() -> ())
    systemc.cpp.call_indirect %5() : () -> ()
    %6 = systemc.cpp.member_access %state arrow "c" :
      (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
    %7 = systemc.convert %6 : (i32) -> !systemc.uint<32>
    systemc.signal.write %y, %7
  }
  systemc.cpp.destructor {
    systemc.cpp.delete %state : !emitc.ptr<!emitc.opaque<"VBar">>
  }
}
```

### Bridging between Interop Mechanisms

As we have already seen, the instance-side lowering adds an attribute to the
interop operations to indicate the interop mechanism that has to be supported
to use the bodies operations and types directly. But what happens when the
instance annotated the operations to be textual C++, but the interface
implementation of the container operation only supports CFFI?
In that case, the dialect that lowers the instance could also provide a lowering
pattern for that particular interop mechanism (instead of registering the
bridging patterns), or there can be patterns that convert the interop operations
to another interop mechanism. Using the dialect conversion framework, this will
then allow the pass to legalize all operations in the IR by using that extra
pattern instead of failing because of an illegal operation for which no pattern
matches.

Let's take a closer look into the CPP to CFFI bridging pattern. It will convert
the interop operations from earlier to CFFI compatible versions.

The `alloc` operation
```mlir
%state = interop.procedural.alloc cpp : !emitc.ptr<!emitc.opaque<"VBar">>
```
is converted to a version where all types are replaced by opaque pointers:
```mlir
%state = interop.procedural.alloc cffi : !llvm.ptr
```

For the `update` operation

```mlir
%1 = interop.procedural.update cpp [%state](%x, %x) :
  [!emitc.ptr<!emitc.opaque<"VBar">>](i32, i32) -> i32 {
^bb0(%arg0: !emitc.ptr<!emitc.opaque<"VBar">>, %arg1: i32, %arg2: i32):
  %2 = systemc.cpp.member_access %arg0 arrow "a" :
    (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
  systemc.cpp.assign %2 = %arg1 : i32
  %3 = systemc.cpp.member_access %arg0 arrow "b" :
    (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
  systemc.cpp.assign %3 = %arg2 : i32
  %4 = systemc.cpp.member_access %arg0 arrow "eval" :
    (!emitc.ptr<!emitc.opaque<"VBar">>) -> (() -> ())
  systemc.cpp.call_indirect %4() : () -> ()
  %5 = systemc.cpp.member_access %arg0 arrow "c" :
    (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
  interop.return %5 : i32
}
```

the state types are also converted to opaque pointers and the body is replaced
with a function call. The original body is moved to a function in another file
that will be piped through `ExportSystemC` to generate the textual C++ that
exports it as a `extern "C"` function.

```mlir
systemc.cpp.func externC @update (%arg0: !emitc.ptr<!emitc.opaque<"void">>,
                                  %arg1: i32, %arg2: i32) -> i32 {
  %0 = emitc.cast %arg0 : !emitc.ptr<!emitc.opaque<"void">> to
    !emitc.ptr<!emitc.opaque<"VBar">>
  %2 = systemc.cpp.member_access %0 arrow "a" :
    (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
  systemc.cpp.assign %2 = %arg1 : i32
  %3 = systemc.cpp.member_access %0 arrow "b" :
    (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
  systemc.cpp.assign %3 = %arg2 : i32
  %4 = systemc.cpp.member_access %0 arrow "eval" :
    (!emitc.ptr<!emitc.opaque<"VBar">>) -> (() -> ())
  systemc.cpp.call_indirect %4() : () -> ()
  %5 = systemc.cpp.member_access %0 arrow "c" :
    (!emitc.ptr<!emitc.opaque<"VBar">>) -> i32
  interop.return %5 : i32
}
// ----- other file
func.func @update (%arg0: !llvm.ptr, %arg1: i32, %arg2: i32) -> i32
%1 = interop.procedural.update cffi [%state](%x, %x) :
  [!llvm.ptr](i32, i32) -> i32 {
^bb0(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32):
  %1 = func.call @update(%arg0, %arg1, %arg2) : (!llvm.ptr, i32, i32) -> i32
  interop.return %1 : i32
}
```

The `init` and `dealloc` operations follow the same pattern as the `update`
operation.

### How to use this dialect

First, we need to distinguish between the different ways this dialect might want
to be used in another dialect or downstream repository:
* Use existing dialects that implement interop lowerings in a specialized
  pipeline: this requires implementing a new pass in the downstream tool that
  registers all the rewrite patterns and external models from the upstream
  dialects required to make the pass succeed in that specialized environment.
  Having a specialized pass allows to reduce the number of dependent dialect as
  all dialects, external models, and patterns that could potentially be needed
  have to be registered upfront in MLIR. It's also more convenient to have a
  one-shot pass instead of a series of partial lowerings.
* Add a new tool or IR that can be used as an instance: implement the
  instance-side lowering pass for your dialect and expose a function to
  populate the rewrite pattern in a provided pattern set.
* Add a new IR that can be used as a container for other interop instances:
  implement the container-side interop interface and a container-side lowering
  pass that registers the implemented interface, the four container-side
  lowering patterns and bridging patterns provided by the `interop` dialect.
* Add a new interop mechanism: this is currently not supported and requires
  modifications to the `interop` dialect. Adding a new mechanism requires
  extending the `InteropMechanism` table-gen enum and adding a bridging pattern
  to CFFI at a minimum.


### Design considerations

This section aims to provide explanations for certain design decisions. It will
take the form of a Q&A.


**Why does the container-side lowering use interfaces that are called from a**
**rewrite pattern?**

Generally speaking, switching between interfaces and rewrite patterns allows us
to control the dependency direction between the container interop operations
and the interop lowering pass.
When using external models to describe the lowering, the pass does not need to
know about that interop operation, it just needs to know about the interface
that describes the external model. However, the interop operation needs to know
about the interop lowering pass.
When using rewrite patterns (without any interfaces), the pattern needs to
depend on the interop operation and the interop lowering pass needs to depend on
the patterns.
While this is also a design consideration, it is not the main concern here.
Note that the lowering patterns should be provided by the operation that is
the closest surrounding op (in terms of regions) to the unrealized interop
operation and that supports interop.
To illustrate why this approach was taken, we can take a look at some of the
alternatives considered:
* The dialect exports a function to populate a rewrite pattern that matches on
  the container operation of that dialect. This has the disadvantage that all
  interop casts inside that container have to be handled in that single pattern
  and the pattern thus needs to walk the region. Another problem is that we need
  the closest operation to perform the interop lowering and the rewrite pattern
  driver walks top-down, thus the pattern would need to stop walking at operations
  that also support interop, but it can only know that if it gets a list of such
  operations or if all of the interop supporting operations implement an interface.
* The dialect exports a function to populate a rewrite pattern that matches on
  the interop cast operation. The disadvantage there is that we then potentially
  have a lot of rewrite patterns matching on the same operation and we thus need
  to specify the matching condition to only succeed when the closest interop
  supporting operation is the one that provided the rewrite pattern. Therefore,
  we would still need to know which operations support interop. Alternatively,
  the pattern returns the operation it would match on and the infra tries all
  of them and selectes the one with the closest operation returned, however,
  that doesn't fit the dialect conversion framework very well and would be
  rather bad performance-wise.
* The dialect adds a trait to the operations that support interop. The interop
  rewrite patterns could then be provided by the downstream dialect, match on
  the unrealized interop operations, and add as a matching condition that no
  other operation having the interop trait is closer to the unrealized interop
  operation. This still has the disadvantage that the conversion framework has
  to cycle through potentially many patterns matching on the same operations
  and check quite expensive matching conditions.
* Current implementation: The dialect implements an external model for the
  interop operation. The lowering pass infra then provides the rewrite patterns
  that match on the unrealized interop operations (one per op) and checks for
  the closest parent operation that implements the interface and then calls the
  lowering provided by the interface implementation.

**Why is instance-side lowering implemented using rewrite patterns?**

We also have the option to implement the lowering patterns in an interface or
directly provide a rewrite pattern with the same trade-offs as mentioned in the
beginning of the above question. The only advantage of the interface approach
is the inversion of the dependencies which is not important for us since we
implement partial lowering passes for every dialect anyways. The pattern based
approach is more straight-forward to implement though.

**Why are those interfaces implemented as external models?**

Because the interface implementations need to implement a lowering pattern that
might have to create operations from a few different dialects, it needs to
include all of them and link against them. This can add a big burden to some
other user that just wants to use the dialect but is not interested in the
interop functionality. It also helps to avoid dependency cycles.

**How are the dialects registered that a pass depends on (for which ops may be**
**created)?**

We have partial lowering passes for each dialect. They have a complete picture
of what operations they create in the instance- and container-side lowerings
and can thus register exactly those. The container-side lowerings also need
to register the dialects needed for the interop mechanism conversion patterns
which are fixed by the `interop` dialect.
When implementing a specialized tool one can set up a specialized one-shot
pass for interop which registers everything that can occur at that stage of
the pipeline.

**Why have unrealized interp operations instead of lowering directly?**

They allow us to get rid of the tight coupling between the instance interop
operations and the surrounding container interop operations. It enables
partial lowering at various places throughout a lowering pipeline and helps
keeping the number of dependences on dialects in the lowering pass low.

**Is there a possibility to support multiple interop mechanisms per instance**
**operation?**

We have to distinguish the case where we know the interop mechanism statically
at the point in time when we create the lowering pass and the case where the
mechanism is selected dynamically in the lowering pass.
The first case is easy to support because one can implement multiple patterns
that lower the instance operation in different ways and register the one that
fits the best.
In a dynamic setting, this is still possible by registering multiple patterns
for the same op, but the set of operations they match on needs to be disjoint
and must not depend on any parent operations. As a result, it might be tricky
to add a specialized lowering without copy-pasting the original lowering
pattern and restricting its match condition.

**How are the bridges between interop mechanisms provided?**

They are provided by the interop dialect as rewrite patterns similar to how the
rewrite patterns for the instance interop to interop cast conversion is provided
by the other dialects. This has the advantage that these conversions, which
usually follow a common pattern, have to be implemented only once.

**Why are the interop operations four separate operations instead of one with**
**three regions?**

This allows to move the code fragments for allocation and deallocation to their
final location not only during the container-side lowering, but also during a
regular transformation pass that, e.g., collects all the state that needs to be
persistent over multiple cycles in a simulator and inserts allocation code at
the top-level.

**Why is there a separate `interop.alloc` operation? Why not make it part of**
**`interop.init`?**

We need to have an operation returning SSA values for the state that which can
be used in the other unrealized interop operations to link them together (we
always need to know what alloc, update, and dealloc operations belong together).
Returning those values from the `init` operation would lead to less flexibility,
a more complicated `init` operation, and might be misleading as the body returns
the values that need to be assigned to the state variables by the container-side
pass, but the returned values would be the state variables.

**Why is this a separate dialect?**

This dialect provides infrastructure and utilities for all dialects that want
to implement some kind of interop. Those dialects need to depend on this one.
The only dialect where it could make sense to add this functionality to is HW,
but we don't want to bloat HW since it is the core dialect that everyone depends
on (also the dialects that don't need interop).
