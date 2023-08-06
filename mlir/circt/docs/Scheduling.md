# Static scheduling infrastructure

Scheduling is a common concern in hardware design, for example in high-level
synthesis flows targeting an FSM+Datapath execution model ("static HLS"). This
document gives an overview of, and provides rationale for, the infrastructure in
the `circt::scheduling` namespace. At its core, it defines an **extensible
problem model** that acts as an interface between **clients** (i.e. passes that
have a need to schedule a graph-like IR) and reusable **algorithm**
implementations.

This infrastructure aims to provide:
- a library of ready-to-use problem definitions and schedulers for clients to
  hook into.
- an API to make algorithm implementations comparable and reusable.
- a mechanism to extend problem definitions to model additional concerns and
  constraints.

## Getting started

Let's walk through a simple example. Assume we want to *schedule* the
computation in the entry block of a function such as `@foo(...)` in the listing
below. This means we want to assign integer *start times* to each of the
*operations* in this untimed IR.

```mlir
func @foo(%a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32) -> i32 {
  %0 = arith.addi %a1, %a2 : i32
  %1 = arith.addi %0, %a3 : i32
  %2:3 = "more.results"(%0, %1) : (i32, i32) -> (i32, i32, i32)
  %3 = arith.addi %a4, %2#1 : i32
  %4 = arith.addi %2#0, %2#2 : i32
  %5 = arith.addi %3, %3 : i32
  %6 = "more.operands"(%3, %4, %5) : (i32, i32, i32) -> i32
  return %6 : i32
}
```

Our only constraint is that an operation can start *after* its operands have
been computed. The operations in our source IR are unaware of time, so we need
to associate them with a suitable *operator type*. Operator types are an
abstraction of the target architecture onto which we want to schedule the source
IR. Here, the only *property* we need to model is their *latency*. Let's assume
that additions take 1 time step, the operations in the dummy `more.` dialect
take 3 time steps. As the return operation just passes control back to the
caller, we assume a latency of 0 time steps for it.

### Boilerplate

The scheduling infrastructure currently has three toplevel header files.

```c++
//...
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"
//...
using namespace circt::scheduling;
```

### Constructing a problem instance

Our stated goal requires solving an acyclic scheduling problem without resource
constraints, represented by the `Problem` class in the scheduling
infrastructure. We need to construct an *instance* of the problem, which serves
as a container for the problem *components* as well as their properties. The
MLIR operation passed as an argument to the `get(...)` method is used to emit
diagnostics.

```c++
auto prob = Problem::get(func);
```

Then, we set up the operator types with the latencies as discussed in the
introduction. Operator types are identified by string handles.

```c++
auto retOpr = prob.getOrInsertOperatorType("return");
prob.setLatency(retOpr, 0);
auto addOpr = prob.getOrInsertOperatorType("add");
prob.setLatency(addOpr, 1);
auto mcOpr = prob.getOrInsertOperatorType("multicycle");
prob.setLatency(mcOpr, 3);
```

Next, we register all operations that we want to consider in the problem
instance, and link them to one of the operator types.

```c++
auto &block = func.getBlocks().front();
for (auto &op : block) {
  prob.insertOperation(&op);
  if (isa<func::ReturnOp>(op))
    prob.setLinkedOperatorType(&op, retOpr);
  else if (isa<arith::AddIOp>(op))
    prob.setLinkedOperatorType(&op, addOpr);
  else
    prob.setLinkedOperatorType(&op, mcOpr);
}
```

Note that we do not have to tell the instance about the *dependences* between
the operations in this simple example because the problem model automatically
includes the SSA def-use-edges maintained by MLIR. However, we often have to
consider additional dependences that are not represented by value flow, such as
memory dependences. For these situations, so-called [auxiliary](#components)
dependences between operations are inserted explicitly into the problem:
`prob.insertDependence(srcOp, destOp)`.

### Scheduling

Before we attempt to schedule, we invoke the `check()` method, which ensures
that the constructed instance is complete and valid. For example, the check
would capture if we had forgot to set an operator type's latency. We dump the
instance to visualize the dependence graph.

```c++
auto checkRes = prob.check();
assert(succeeded(checkRes));
dumpAsDOT(prob, "sched-problem.dot");
```

![Dump of example instance](https://circt.llvm.org/includes/img/sched-instance.svg)

We use a simple list scheduler, available via the `Algorithms.h` header, to
compute a solution for the instance.

```c++
auto schedRes = scheduleASAP(prob);
assert(succeeded(schedRes));
```

### Working with the solution

The solution is now stored in the instance, and we invoke the problem's
`verify()` method to ensure that the computed start times adhere to the
precedence constraint we stated earlier, i.e. operations start after their
operands have computed their results. We can also convince ourselves of that by
dumping the instance and inspecting the solution.

```c++
auto verifRes = prob.verify();
assert(succeeded(verifRes));
dumpAsDOT(prob, "sched-solution.dot");
```

![Dump of example instance, including solution](https://circt.llvm.org/includes/img/sched-solution.svg)

To inspect the solution programmatically, we can query the instance in the
following way. Note that by convention, all getters in the problem classes
return `Optional<T>` values, but as we have already verified that the start
times for registered operations are set, we can directly dereference the values.

```c++
for (auto &op : prob.getOperations())
  llvm::dbgs() << *prob.getStartTime(&op) << "\n";
```

And that's it! For a more practical example, have a look at the 
[`AffineToPipeline`](https://github.com/llvm/circt/blob/main/lib/Conversion/AffineToPipeline/AffineToPipeline.cpp)
pass.

## Extensible problem model

### Theory and terminology

Scheduling problems come in many flavors and variants in the context of hardware
design. In order to make the scheduling infrastructure as modular and flexible
as CIRCT itself, it is build on the following idea of an *extensible problem
model*:

An *instance* is comprised of *components* called *operations*, *dependences*
and *operator types*. Operations and dependences form a graph structure and
correspond to the source IR to be scheduled. Operator types encode the
characteristics of the target IR. The components as well as the instance can be
annotated with *properties*. Properties are either *input* or *solution*
properties, based on whether they are supplied by the client, or computed by the
algorithm. The values of these properties are subject to the *input constraints*
and *solution constraints*, which are a first-class concern in the model and are
intended to be strictly enforced before respectively after scheduling.

Concrete problem definitions derived from this model share the same
representation of the components, but differ in their sets of properties (and
potentially distinction of input and solution properties) and input and solution
constraints. Hence, we tie together properties and constraints to model a
specific scheduling problem. Extending one (or more!) parent problems means
inheriting or adding properties, and redefining the constraints (as these don't
always compose automatically).

A key benefit of this approach is that these problem definitions provide a
reliable contract between the clients and algorithms, making it clear which
information needs to be provided, and what kind of solution is to be expected.
Clients can therefore choose a problem definition that fits their needs, and
algorithms can *opt-in* to accepting a specific subset of problems, which they
can solve efficiently. Extensibility is ensured because new problem definitions
can be added to the infrastructure (or inside a specific lowering flow, or even
out-of-tree) without adapting any existing users.

### Implementation

See
[Problems.h](https://github.com/llvm/circt/blob/main/include/circt/Scheduling/Problems.h) /
[Problems.cpp](https://github.com/llvm/circt/blob/main/lib/Scheduling/Problems.cpp).

#### Problem definitions

The `Problem` class is currently the base of the problem hierarchy. Several
extended problems are [currently defined](#available-problem-definitions) via
virtual multiple inheritance. Upon construction, a `containingOp` is passed to
instances. This MLIR operation is currently only used to emit diagnostics, and
has no semantic meaning beyond that.

#### Components

The infrastructure uses the following representation of the problem components.

Operations are just `mlir::Operation *`s.

We distinguish two kinds of dependences, *def-use* and *auxiliary*. Def-use
dependences are part of the SSA graph maintained by MLIR, and can distinguish
specific result and operand numbers. As we expect any relevant graph-like input
IR to use this MLIR facility, instances automatically consider these edges
between registered operations. Auxiliary dependences, in contrast, only specify
a source and destination operation, and have to be explicitly added to the
instance by the client, e.g. for control or memory dependences. The
`detail::Dependence` class abstracts the differences between both kinds, in
order to offer a uniform API to iterate over dependences and query their
properties.

Lastly, operator types are identified by `mlir::StringAttr`s, in order to give
clients maximum flexibility in modeling their operator library. This may change
in the future, when a CIRCT-wide concept to model physical properties of
hardware emerges.

#### Properties

Properties can involve arbitrary data types, as long as these can be stored in
maps. Problem classes offer public getter and setter methods to access a given
components properties. Getters return optional values, in order to indicate if a
property is unset. For example, the signature of the method the queries the
computed start time is `Optional<unsigned> getStartTime(Operation *op)`.

#### Constraints

Clients call the virtual `Problem::check()` method to test any input
constraints, and `Problem::verify()` to test the solution constraints. Problem
classes are expected to override them as needed. There are no further
restrictions of how these methods are implemented, but it is recommended to
introduce helper methods that test a specific aspect and can be reused in
extended problems. In addition, it makes sense to check/verify the properties in
an order that avoids redundant tests for the presence of a particular property
as well as redundant iteration over the problem components.

## Available problem definitions

*See the linked Doxygen docs for more details.*

- [Problem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1Problem.html):
  A basic, acyclic problem at the root of the problem hierarchy. Operations are
  linked to operator types, which have integer latencies. The solution comprises
  integer start times adhering to the precedence constraints implied by the
  dependences.
- [CyclicProblem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1CyclicProblem.html):
  Cyclic extension of `Problem`. Its solution solution can be used to construct
  a pipelined datapath with a fixed, integer initiation interval, in which the
  execution of multiple iterations/samples/etc. may overlap. Operator types are
  assumed to be fully pipelined.
- [SharedOperatorsProblem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1SharedOperatorsProblem.html):
  A resource-constrained scheduling problem that corresponds to multiplexing
  multiple operations onto a pre-allocated number of fully pipelined operator
  instances.
- [ModuloProblem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1ModuloProblem.html):
  Models an HLS classic: Pipeline scheduling with limited resources.
- [ChainingProblem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1ChainingProblem.html):
  Extends `Problem` to consider the accumulation of physical propagation delays
  on combinational paths along SSA dependences.

NB: The classes listed above each model a *trait*-like aspect of scheduling.
These can be used as-is, but are also intended for mixing and matching, even
though we currently do not provide definitions for all possible combinations in
order not to pollute the infrastructure. For example, the `ChainingProblem` may
be of limited use standalone, but can serve as a parent class for a future
chaining-enabled modulo scheduling problem.

## Available schedulers

- ASAP list scheduler
  ([`ASAPScheduler.cpp`](https://github.com/llvm/circt/blob/main/lib/Scheduling/ASAPScheduler.cpp)):
  Solves the basic `Problem` with a worklist algorithm. This is mostly a
  problem-API demo from the viewpoint of an algorithm implementation.
- Linear programming-based schedulers
  ([`SimplexSchedulers.cpp`](https://github.com/llvm/circt/blob/main/lib/Scheduling/SimplexSchedulers.cpp)):
  Solves `Problem`, `CyclicProblem` and `ChainingProblem` optimally, and
  `SharedOperatorsProblem` / `ModuloProblem` with simple (not state-of-the-art!)
  heuristics. This family of schedulers shares a tailored implementation of the
  simplex algorithm, as proposed by de Dinechin. See the sources for more
  details and literature references.
- Integer linear programming-based scheduler
  ([`LPSchedulers.cpp`](https://github.com/llvm/circt/blob/main/lib/Scheduling/LPSchedulers.cpp)):
  Demo implementation for using an ILP solver via the OR-Tools integration.

## Utilities

See 
[`Utilities.h`](https://github.com/llvm/circt/blob/main/include/circt/Scheduling/Utilities.h):
- Topological graph traversal
- DFA to compute combinational path delays
- DOT dump

## Adding a new problem

*See e.g. [#2233](https://github.com/llvm/circt/pull/2233), which added the
`ChainingProblem`.*

- Decide where to add it. Guideline: If it is trait-like and similar to the
  existing problem mentioned above, add it to `Problems.h`. If the model is
  specific to your use-case, it is best to start out in locally in your
  dialect/pass.
- Declare the new problem class and inherit *virtually* from the relevant
  superclasses (at least `Problem`).
- Define additional properties (private), and the corresponding public
  getters/setters. Getters return `Optional<T>` values, to indicate an unset
  state.
   - Note that dependence properties are somewhat expensive to store, making it
     desirable that clients and algorithms expect and handle the unset state.
     This should be clearly documented. Example: `distance` property in
     `CyclicProblem`.
- Redefine the `getProperties(*)` methods to get dumping support. These should
  consider any properties the new class adds, plus properties defined in the
  superclass(es).
- Redefine `check()` (input constraints) and `verify()` (solution constraints).
  If possible, follow the
  [design used in the existing problem classes](#constraints).

### Testing

Please extend the [SSP](https://circt.llvm.org/docs/Dialects/SSP/) dialect to
enable testing of the new problem definition.
- If the problem defines any new properties, add them to
  [`SSPAttributes.td`](https://github.com/llvm/circt/blob/main/include/circt/Dialect/SSP/SSPAttributes.td).
- Instantiate the
  [`Default<ProblemT>`](https://github.com/llvm/circt/blob/main/include/circt/Dialect/SSP/Utilities.h#L457-L459)
  template for the new problem.
- Handle the problem class in the
  [`-ssp-roundtrip`](https://github.com/llvm/circt/blob/main/lib/Dialect/SSP/Transforms/Roundtrip.cpp)
  pass.
- Write a couple of "positive" testcases, as well as at least one error test for
  each input/solution constraint, as validated by `check()` / `verify()`. See
  the [existing test cases](https://github.com/llvm/circt/tree/main/test/Scheduling)
  for inspiration.

## Adding a new scheduler

*See e.g. [#2650](https://github.com/llvm/circt/pull/2650), which added a
scheduler for the `CyclicProblem`.*

- Schedulers should opt-in to specific problems by providing entry points for
  the problem subclasses they support. Example:
  ```c++
  LogicalResult awesomeScheduler(Problem &prob);
  LogicalResult awesomeScheduler(CyclicProblem &prob);
  ```
- Schedulers can expect that the input invariants were enforced by a
  `check()`-call in the client, and must compute a solution that complies with
  the solution constraints when the client calls the problem's `verify()`
  method.
- Schedulers can live anywhere. If a new algorithm is not entirely
  dialect/pass-specific and supports problems defined in `Problems.h`, it should
  offer entry points in `Algorithms.h`.
- Objectives are not part of the problem signature. Therefore, if an algorithm
  supports optimizing for different objectives, clients should be able to select
  one via the entry point(s).

### Testing

- To enable testing, add the new scheduler to the
  [`-ssp-schedule`](https://github.com/llvm/circt/blob/main/lib/Dialect/SSP/Transforms/Schedule.cpp)
  pass, and invoke it from the test cases for the supported problems
  ([example](https://github.com/llvm/circt/blob/main/test/Scheduling/problems.mlir#L2-L4)).
- If the algorithm may fail in certain situations (e.g., "linear program is
  infeasible"), add suitable error tests as well.
