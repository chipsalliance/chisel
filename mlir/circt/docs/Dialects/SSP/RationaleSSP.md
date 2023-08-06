# SSP Dialect Rationale

This document describes various design points of the SSP dialect, why they are
the way they are, and current status.  This follows in the spirit of other [MLIR
Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

CIRCT's [scheduling infrastructure](https://circt.llvm.org/docs/Scheduling/) is
lightweight and dialect-agnostic, in order to fit into any lowering flow with a
need for static scheduling. However, it lacks an import/export format for
storing and exchanging problem instances. The SSP ("**S**tatic **S**cheduling
**P**roblems") dialect fills that role by defining an IR that captures problem
instances 
- in full fidelity,
- in a concise syntax,
- and independent of any other "host" dialect.

The SSP dialect's main use-cases are [testing](#testing),
[benchmarking](#benchmarking) and [rapid-prototyping](#rapid-prototyping). It is
strictly a companion to the existing scheduling infrastructure, and clients (HLS
flows etc.) are advised to use the C++ APIs directly.

### Testing

In order to test the scheduling infrastructure's problem definitions (in
particular, their input checkers/solution verifiers) and algorithm
implementations, a "host" IR to store problem instances is needed. To that end,
the test-cases started out with a mix of standard and unregistered operations,
and heavy use of generic attributes, as shown in the following example (note
especially the index-based specification of auxiliary dependences):

```mlir
func.func @canis14_fig2() attributes {
  problemInitiationInterval = 3,
  auxdeps = [ [3,0,1], [3,4] ],
  operatortypes = [
    { name = "mem_port", latency = 1, limit = 1 },
    { name = "add", latency = 1 }
  ] } {
  %0 = "dummy.load_A"() { opr = "mem_port", problemStartTime = 2 } : () -> i32
  %1 = "dummy.load_B"() { opr = "mem_port", problemStartTime = 0 } : () -> i32
  %2 = arith.addi %0, %1 { opr = "add", problemStartTime = 3 } : i32
  "dummy.store_A"(%2) { opr = "mem_port", problemStartTime = 4 } : (i32) -> ()
  return { problemStartTime = 5 }
}
```

Here is the same test-case encoded in the SSP dialect:

```mlir
ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
  library {
    operator_type @MemPort [latency<1>, limit<1>]
    operator_type @Add [latency<1>]
    operator_type @Implicit [latency<0>]
  }
  graph {
    %0 = operation<@MemPort>(@store_A [dist<1>]) [t<2>]
    %1 = operation<@MemPort>() [t<0>]
    %2 = operation<@Add>(%0, %1) [t<3>]
    operation<@MemPort> @store_A(%2) [t<4>]
    operation<@Implicit>(@store_A) [t<5>]
  }
}
```

Emitting an SSP dump is also useful to test that an conversion pass correctly
constructs the scheduling problem, e.g. checking that it contains a memory
dependence to another operation:

```mlir
// CHECK: operation<@{{.*}}>(%0, @[[store_1]])
%5 = affine.load %0[0] : memref<1xi32>
...
// CHECK: operation<@{{.*}}> @[[store_1:.*]](%7, %0)
affine.store %7, %0[0] : memref<1xi32>
```

### Benchmarking

Scheduling is a hard combinatorial optimization problem that can be solved by a
variety of approaches, ranging from fast heuristics to exact formulations in
mathematical frameworks such as integer linear programs capable of computing
provably optimal solutions. It is therefore important to evaluate scheduler
implementations beyond just functional correctness testing, i.e. to assess the
scheduler's runtime and scalability, as well as the solution quality, on sets of
representative benchmark instances.

With the SSP dialect, such instances can be saved directly from synthesis flows
using CIRCT's scheduling infrastructure, or emitted in the textual MLIR format
by third-party tools. As the SSP IR is self-contained, it would even be viable
to store problem instances originating from out-of-tree or proprietary flows, as
their source and target IRs would not be required to load and schedule a problem
instance in a benchmark harness.

### Rapid prototyping

The SSP dialect also provides a path towards automatically generated Python
bindings for the scheduling infrastructure, which will ease the prototyping of
new scheduling clients and problem definitions.

### Q&A
- **Q:** Do I have to do a dialect conversion to and from this dialect to
schedule something?

  No, use the C++ API, i.e. the problem classes and scheduler entrypoints in
  `circt::scheduling`, directly! This dialect is a one-way street in terms of
  dialect conversion, and only intended to load and store problem instances for
  the use-cases listed above.

- **Q:** Why don't you use something like Cap'nProto to (de)serialize the
problem instances?

  Textual MLIR is reasonably easy to write by hand, which is important for
  test-cases, and we need MLIR operations anyways, because the scheduling
  infrastructure builds on top of the MLIR def-use graph to represent its
  dependence graphs.

- **Q:** `OperationOp` doesn't seem like a great name.

  No, you're right. However, the SSP dialect uses the same terminology as the
  scheduling infrastructure, so any changes would have to originate there.

## Rationale for selected design points

### Use of container-like operations instead of regions in `InstanceOp`

This dialect defines the `OperatorLibraryOp` and `DependenceGraphOp` to serve as
the first and second operation in an `InstanceOp`'s region. The alternative of
using two regions on the `InstanceOp` is not applicable, because the
`InstanceOp` then needs to provide a symbol table, but the upstream
`SymbolTable` trait enforces single-region ops. Lastly, we also considered using
a single graph region to hold both `OperatorTypeOp`s and `OperationOp`s, but
discarded that design because it cannot be safely roundtripped via a
`circt::scheduling::Problem` (internally, registered operator types and
operations are separate lists).

### Stand-alone use of the `OperatorLibraryOp`

The `OperatorLibraryOp` can be named and used outside of an `InstanceOp`. This
is useful to share operator type definitions across multiple instances. In
addition, until CIRCT gains better infrastructure to manage predefined hardware
modules and their properties, such a stand-alone `OperatorLibraryOp` can also
act as an interim solution to represent operator libraries for scheduling
clients.

### Use of SSA operands _and_ symbol references to encode dependences

This is required to faithfully reproduce the internal modeling in the scheduling
infrastructure, which distinguishes def-use (result to operand, tied to MLIR SSA
graph) and auxiliary (op to op, stored explicitly) dependences
([example](https://circt.llvm.org/docs/Scheduling/#constructing-a-problem-instance)).
To represent the former, the `OperationOp` produces an arbitrary number of
`NoneType`-typed results, and accepts an arbitrary number of operands, thus
spanning a def-use graph. Auxiliary dependences are encoded as symbol uses,
which reference the name of the dependence's source `OperationOp`. Modeling
these dependences with symbols rather than SSA operands is a necessity because
the scheduling infrastructure implicitly considers *all* def-use edges between
registered operations. Hence, auxiliary dependences, hypothetically encoded as
SSA operands, would be counted twice.

### No attribute interface for scheduling properties

Properties are represented by dialect attributes inheriting from the base
classes in `PropertyBase.td`, which include `extraClassDeclaration`s for
`setInProblem(...)` and `getFromProblem(...)` methods that directly interact
with the C++ problem class. In order to get/set a certain property, a reference
to the concrete class is required, e.g.: a `CyclicProblem &` if we want to set a
dependence's `distance` property.

A more obvious design would be to make these methods part of an attribute
interface. However, then the methods could only accept a `Problem &`, which
cannot be statically downcasted to the concrete class due to the use of virtual
multiple inheritance in the problem class hierarchy. If the inheritance model
were to change in the scheduling infrastructure, the use of attribute interfaces
should be reconsidered.

## Import/export

The `circt/Dialect/SSP/Utilities.h` header defines methods to convert between
`ssp.InstanceOp`s and `circt::scheduling::Problem` instances. These utilities
use template parameters for the problem class and the property attribute
classes, allowing client code to load/save an instance of a certain problem
class with the given properties (but ignoring others). Incompatible properties
(e.g. `distance` on a base `Problem`, or `initiationInterval` on an operation)
will be caught at compile time as errors in the template instantiation. Note
that convenience versions that simply load/save all properties known in the
given problem class are provided as well.

## Extensibility

A key feature of the scheduling infrastructure is its extensibility. New problem
definitions in out-of-tree projects have to define attributes inheriting from
the property base classes in one of their own dialects. Due to the heavy use of
templates in the import/export utilities, these additional attributes are
supported uniformly alongside the built-in property attributes. The only
difference is that the SSP dialect provides short-form pretty printing for its
own properties, whereas externally-defined properties fall back to the generic
dialect attribute syntax.
