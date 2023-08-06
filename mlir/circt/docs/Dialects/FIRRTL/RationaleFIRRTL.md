# FIRRTL Dialect Rationale

This document describes various design points of the FIRRTL dialect, why it is
the way it is, and current status and progress.  This follows in the spirit of
other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

[The FIRRTL project](https://github.com/chipsalliance/firrtl) is an existing
open source compiler infrastructure used by the Chisel framework to lower ".fir"
files to Verilog.  It provides a number of useful compiler passes and
infrastructure that allows the development of domain specific passes.  The
FIRRTL project includes a [well documented IR
specification](https://github.com/chipsalliance/firrtl/blob/master/spec/spec.pdf)
that explains the semantics of its IR, an [ANTLR
grammar](https://github.com/chipsalliance/firrtl/blob/master/src/main/antlr4/FIRRTL.g4)
includes some extensions beyond it, and a compiler implemented in Scala which we
refer to as the _Scala FIRRTL Compiler_ (SFC).

_The FIRRTL dialect in CIRCT is designed to provide a drop-in replacement for
the SFC for the subset of FIRRTL IR that is produced by Chisel and in common
use._  The FIRRTL dialect also provides robust support for SFC _Annotations_.

To achieve these goals, the FIRRTL dialect follows the FIRRTL IR specification
and the SFC implementation almost exactly.  Where the FIRRTL specification
allows for undefined behavior, FIRRTL dialect and its passes will choose the SFC
interpretation of specific undefined behavior.  The small deviations we do make
are discussed below.  Early versions of the FIRRTL dialect made _heavy
deviations_ from FIRRTL IR and the SFC (see the Type Canonicalization section
below).  These deviations, while elegant, led to difficult to resolve mismatches
with the SFC and the inability to verify FIRRTL IR.  The remaining small
deviations introduced in the FIRRTL dialect are done to simplify the CIRCT
implementation of a FIRRTL compiler and to take advantage of MLIR's various
features.

This document generally assumes that you've read and have a basic grasp of the
FIRRTL IR spec, and it can be occasionally helpful to refer to the ANTLR
grammar.

## Status

The FIRRTL dialect and FIR parser is a generally complete implementation of the
FIRRTL specification and is actively maintained, tracking new enhancements. The
FIRRTL dialect supports some undocumented features and the "CHIRRTL" flavor of
FIRRTL IR that is produced from Chisel.  The FIRRTL dialect has support for
parsing an SFC Annotation file consisting of only local annotations and
converting this to operation or argument attributes.  Non-local annotations are
also supported.

There are some exceptions to the above:

1) We don't support the `Fixed` types for fixed point numbers, and some
   primitives associated with them.
2) We don't support `Interval` types

Some of these may be research efforts that didn't gain broad adoption, in which
case we don't want to support them.  However, if there is a good reason and a
community that would benefit from adding support for these, we can do so.

## Naming

Names in Verilog form part of the public API of a design and are used for many
purposes and flows.  Many things in verilog may have names, and those names
specify points of interaction with the design.  For example, a wire has a name,
and one can monitor the value on the wire from a testbench by knowing this name.
Instances have names and form the core of hierarchical references through
designs.  Even always blocks and loops can have names, which are required and
used.

It is therefore critical that Chisel, and by extension FIRRTL, have
language-level semantics about how entities are named and how named entities are
used and transformed.  This must specify which entities with names in Chisel
generate predictable output.  Since names serve multiple purposes in a design,
for example, debugging, test-bench attachment, hooks for physical layout, etc,
we must balance multiple needs.  This section describes the base semantics,
which are conservative and aimed at enabling debugging.  The CIRCT
implementation of a FIRRTL compiler provides options to change the name
preservation behavior to produce more debuggable or more optimized output.

Modules shall use the name given in Chisel, unless they conflict with a Verilog
reserved word, not withstanding de-duplication or relevant annotations on the
module.

Instances shall use the name given in Chisel, unless they conflict with a
Verilog reserved word.  Instances have preferential use of the name in the
output in case of a conflict, after ports.

Chisel provides a "Don't Touch" annotation to protect entities from
transformation.  A "Don't Touch" on a wire or node produces a wire in Verilog
and preserves the data-flow through that wire.  Even a wire driven by a constant
shall not have the constant forwarded around the wire.  This is because a "Don't
Touch" annotation signals the possible public use of a wire and one common use
is to provide a place to drive a new value into the logic from an external
test-bench.  If the node or wire is named (and it always should be for Chisel
"Don't Touch"), this name is used, unless it conflicts with a Verilog reserved
word.  This wire has preferential use of the name in the output in case of a
conflict, after ports and instances.

Named wires and nodes in FIRRTL shall appear as a wire in the output verilog.
There is no requirement that data-flow through a wire be maintained, only that
the data-flow into a wire be maintained.  This allows bypassing and forwarding
around wires who exist solely because of their name.  An implementation may
choose to not bypass trivial wires to reduce unused wire lint warnings, but
shouldn't cause other lint warnings to avoid unused wire warnings.  A named wire
without a symbol is thus equivalent to a named read-probe in the circuit.

Any name of an entity inside a module which starts with `_` may be discarded.
This name pattern indicates the name is for convenience in the Chisel code
(often temporaries are required) and there is no expectation it exist in the
output.

### Mandatory Renaming

We want the naming of Verilog objects to match the names used in the original
Chisel, but in several passes, there is mandatory renaming.  It is important
that this be a predictable transformation.  For example, after bundles are
replaced with scalars in the lower-types pass, each field should be prefixed
with the bundle name:

```scala
circuit Example
  module Example
    reg myreg: { a :UInt<1>, b: UInt<1> }, clock
; firrtl-lower-types =>
circuit Example
  module Example
    reg myreg_a: UInt<1>, clock
    reg myreg_b: UInt<1>, clock
```

The name transformations applied by the SFC have become part of the documented
API, and people rely on the final names to take a certain form.

### Temporaries

There are names for temporaries generated by the Chisel and FIRRTL tooling which
are not important to maintain. These names are discarded when parsing, which
saves memory during compilation. New names are generated at Verilog export time,
which has the effect of renumbering intermediate value names.  Names generated
by Chisel typically look like `_T_12`, and names generated by the SFC look like
`_GEN_12`. The FIRRTL compiler will not discard these names if the object has an
array attribute `annotations` containing the attribute `{class =
"firrtl.transforms.DontTouchAnnotation}`.

Chisel-generated temporaries will not be discarded in compilation modes which
preserve all names.

### Name Preservation Modes

Name preservation modes, compiler options that produce different name
preservation behavior, was implemented as a compromise between two divergent and
seemingly irreconcilable goals:

1. A FIRRTL to Verilog compiler should apply heavy optimizations to improve its
   own performance (early optimizations produce smaller IR which means later
   passes need to do less work) and to improve the performance of tools
   consuming output Verilog, e.g., Verilog simulator compilation and run time.

2. Chisel users (design and verification engineers) want to see a one-to-one
   correspondence between what they write in Chisel and the Verilog that a
   FIRRTL compiler produces to enable debuggability.

These two goals are viewed as irreconcilable because certain increases to
optimizations (1) necessarily detract from debuggability (2).

Currently CIRCT's FIRRTL compiler, `firtool`, provides two optimization modes,
debug and release, as well as finer grained options with lower-level flags:

1. `-O=release` (or `-preserve-values=none`) may delete any component as part of
   an optimization.
2. `-O=debug` (or `-preserve-values=named`) keeps components with names that do
   not begin with a leading underscore.
3. `-preserve-values=all`, which has no exposed `-O` option, keeps all
   components.

As an example of these modes consider the following FIRRTL circuit:

``` firrtl
circuit Foo:
  module Foo:
    input a: UInt<1>
    output b: UInt<1>

    node named = a
    node _unnamed = named

    b <= _unnamed
```

When compiled with `-O=release` (or `--preserve-values=none`), no intermediary
nodes/wires are preserved because CIRCT inlines the usages of `a` into the
assignment to `b`:

``` verilog
module Foo(
  input  a,
  output b);

  assign b = a;
endmodule
```

When compiled with `-O=debug` (or `-preserve-values=named`), the `_unnamed` node
is removed, but the `named` node is preserved:

``` verilog
module Foo(
  input  a,
  output b);

  wire named = a;
  assign b = named;
endmodule
```

When compiled with `-preserve-values=all` this produces the following Verilog
that preserves all nodes, regardless of name:

``` verilog
module Foo(
  input  a,
  output b);

  wire named = a;
  wire _unnamed = named;
  assign b = _unnamed;
endmodule
```

Design teams are expected to use `-O=debug` debuggabilty.  Verification teams
and downstream tools are expected to use/consume `-O=release`.

This split of two different Verilog outputs initially created reproducibility
problems that CIRCT has attempted to solve with a guarantee of stable
randomization.  Consider a situation where a verification team trips an
assertion failure using a release build with a particular seed.  Because release
Verilog is highly optimized and difficult to debug, they want to switch to a
debug build.  If the release build seed does not reproduce the failure in debug
mode, the verification team needs to search for a failure.  This proved to be a
drag on Chisel users.  Towards alleviating this, CIRCT will now guarantee that
registers in debug or release mode will be randomized to the same value for the
same seed.

It is important to note that the debug/release split was born out of our
inability to reconcile the goals at the top of this section.  Discussion in the
subsequent section involves approaches to unify these two approaches.

#### Alternative Approaches to Name Preservation Modes and Historical Background

The following alternatives were implemented or considered instead of the
debug/release solution.

First, we created dead wire taps _with symbols_ for all "named" things in the
original FIRRTL design.  We would then try to use these dead wire taps in place
of unnamed things when possible.  This simple solution produced much more
readable Verilog.  However, this also had a number of problems.  Namely, leaving
in dead wire taps would result in situations where ports that downstream users
were expecting to be removed were not.  E.g., a module with dead wire taps would
result in more ports at a physical design boundary.  Additionally, leaving in
dead wire taps may introduce coverage holes for verification teams.  We
attempted to remove dead wire taps when possible.  However, this was problematic
as we had given them symbols which indicates that they may have external readers
(e.g., from a manually written testbench) and was intended to indicate that
later passes could never remove these.  We considered using an alternative to a
symbol, but this was rejected due to its highly special-cased nature---it was
forcing us to communicate a Chisel expectation/semantic all the way to HW/SV
dialects.

These drawbacks are unfortunate because they stem from learned expectations of
how the Scala-based FIRRTL Compiler worked.  A negative view of this is that
some level of optimization was required for a learned definition of correctness.
If CIRCT was the first FIRRTL compiler, we may have been able to circumvent
these problems with alternative means that included modifications to Chisel.

Second, we considered having CIRCT create "debug modules" that included all
named signals in the design.  An instance of this debug module would then be
instantiated, via a SystemVerilog `bind` statement, inside the original module.
This was an early suggestion.  However, a concern of users of any "debug module"
is that the debug module would not show usages of the named signals.  E.g., the
example circuit shown above would compile to something like:


``` verilog
module Foo_debug(
  input _0;

  wire named = _0;
endmodule

bind Foo Foo_debug Foo_debug (
  ._0(a)
);

module Foo(
  input  a,
  output b);

  assign b = a;
endmodule
```

The main concern is that while users can see the value of `named` in a waveform,
they cannot trace back its usage in the computation of port `b` in module `Foo`.
This approach also suffers from the issues of the first approach of leaving in
ports and dead logic (that is only used when a debug instance is bound in).

This approach may be revisited in the future as it provides benefits of unifying
debug and release builds into a single release build with run-time debugging
information that can be bound in.  Additionally, use of FIRRTL `RefType`s that
lower to Verilog cross-module references (XMRs) may alleviate some of the issues
above.

Third, a single build that always preserved names was considered.  At the time,
this introduced long Verilog compilation and simulation times.  We were not able
to discern an optimization design point which balanced the needs of
debuggability with compilation and simulation performance.  This does not mean
that such a point does _not_ exist, only that we were not able to find it.  Such
a design point may exist and should be investigated.

Since all these efforts happened, other work has occurred which may make
reviving these efforts a fruitful endeavor.  FIRRTL now has `RefType`s which are
operations which lower to Verilog cross-module references (XMRs).  This may
provide a mechanism to implement the "bound debug instance" approach above
without perturbing port optimizations.  Reliance on symbols to encode
optimization blocking behavior has been largely rolled back.  A
`DontTouchAnnotation` is now encoded as an annotation as opposed to a symbol.  A
new inter-module dead code elimination (IMDCE) pass was implemented which
handles port removal.  The approaches above, or new approaches, may be able to
build a better name preservation approach, but with certain optimizations
enabled.

## Symbols and Inner Symbols

Symbols and Inner Symbols are documented in [Symbol
Rationale](https://circt.llvm.org/docs/RationaleSymbols/).  This documents how symbols are used,
their interaction with "Don't Touch", and the semantics imposed by them.

Public Symbols indicate there are uses of an entity outside the analysis scope
of the compiler.  This requires the entity be preserved in such a way as the
operations possible in the target language have the expected effect.  For
example, a wire or port with a public symbol may be used by name in a test bench
to read or write new values into the circuit.  Therefore, these wires cannot be
detached form their original dataflow as this would break the remote write case,
nor can their input dataflow be changed, as this would break the remote read
case.  They cannot be renamed, as this would break all remote access.

Private Symbols indicate there are symbolic references to the entity, but they
are all within the scope of the compiler's IR and analysis.  An entity with a
private symbol may be arbitrarily transformed, so long as the transformation is
semantic preserving with respect to all uses of the private symbol.  If it can
be proved a wire with a private symbol is only read from via its symbol and not
written to, for example, the input can forwarded to the output (bypassing the
wire) safely.  If a private symbol is unused, it may be removed.  Private
symbols impose no restriction on output; they only exist to enable non-local
effects in the IR.

"Don't Touch" is implemented as a public symbol on an entity.  A conservative
interpretation of "Don't Touch", and a common use case, is that the entity is
referred to by a testbench in unknown ways.  This implies no transformation
which would change observed behavior if the entity was arbitrarily read or
written to remotely.  This further implies the existence of the entity in the
output.

Importantly, the existence of a symbol doesn't specify whether something is
read-only, write-only, or read-write.  Without analysis, a pass must assume the
most conservative case, and in the case of public symbols, must always assume
the most conservative case.  To do better, all uses must be analyzed and
understood (e.g. a symbol used by a verbatim has unknown use).

### Hierarchical Path

In the FIRRTL dialect, it might be necessary to identify specific instances of
operations in the instance hierarchy. The FIRRTL `HierPathOp` operation
(`firrtl.hierpath`) can be used to describe the path through an instance
hierarchy to a declaration, which can be used by other operations or non-local
annotations.  Non-local anchors can refer to most declarations, such as modules,
instances, wires, registers, and memories.

The `firrtl.hierpath` operations defines a symbol and contains a namepath, which
is a list of `InnerRefAttr` and `FlatSymbolRefAttr` attributes. A
`FlatSymbolRefAttr` is used to identify modules, and is printed as `@Module`.
`InnerRefAttr` identifies a declaration inside a module, and is printed as
`@Module::@wire`. Each element along the Paths's namepath carries an annotation
with class `circt. nonlocal`, which has a matching `circt. nonlocal` field
pointing to the global op. Thus instances participating in nonlocal paths are
readily apparent.

In the following example, `@nla` specifies instance `@bar` in module `@Foo`,
followed by instance `@baz` in module `@Bar`, followed by the wire named `@w` in
module `@Baz`.

``` mlir
firrtl.circuit "Foo" {
  firrtl.hierpath @nla [@Foo::@bar, @Bar::@baz, @Baz::@w]
  firrtl.module @Baz() {
    %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla, class = "ExampleAnno"}]} : !firrtl.uint
  }
  firrtl.module @Bar() {
    firrtl.instance baz sym @baz {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]} @Baz()
  }
  firrtl.module @Foo() {
    firrtl.instance bar sym @bar {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]} @Bar()
  }
}
```

## Type system

### Not using standard types

At one point we tried to use the integer types in the standard dialect, like
`si42` instead of `!firrtl.sint<42>`, but we backed away from this.  While it
originally seemed appealing to use those types, FIRRTL operations generally need
to work with "unknown width" integer types (i.e.  `!firrtl.sint`).

Having the known width and unknown width types implemented with two different
C++ classes was awkward, led to casting bugs, and prevented having a
`FIRRTLType` class that unified all the FIRRTL dialect types.

### Not Canonicalizing Flip Types

An initial version of the FIRRTL dialect relied on canonicalization of flip
types according to the following rules:

1) `flip(flip(x))` == `x`.
2) `flip(analog(x))` == `analog(x)` since analog types are implicitly
    bidirectional.
3) `flip(bundle(a,b,c,d))` == `bundle(flip(a), flip(b), flip(c), flip(d))` when
   the bundle has non-passive type or contains an analog type.  This forces the
   flip into the subelements, where it recursively merges with the non-passive
   subelements and analogs.
4) `flip(vector(a, n))` == `vector(flip(a), n)` when the vector has non-passive
   type or analogs.  This forces the flip into the element type, generally
   canceling it out.
5) `bundle(flip(a), flip(b), flip(c), flip(d))` == `flip(bundle(a, b, c, d)`.
   Due to the other rules, the operand to a flip must be a passive type, so the
   entire bundle will be passive, and rule #3 won't be recursively reinvoked.

While elegant in a number of ways (e.g., FIRRTL types are guaranteed to have a
canonical representation and can be compared using pointer equality, flips
partially subsume port directionality and "flow", and analog inputs and outputs
are canonicalized to the same representation), this resulted in information loss
during canonicalization because the number of flip types can change.  Namely,
three problems were identified:

1) Type canonicalization may make illegal operations legal.
2) The flow of connections could not be verified because flow is a function of
   the number of flip types.
3) The directionality of leaves in an aggregate could not be determined.

As an example of the first problem, consider the following circuit:

```firrtl
module Foo:
  output a: { flip a: UInt<1> }
  output b: { a: UInt<1> }

  b <= a
```

The connection `b <= a` _is illegal_ FIRRTL due to a type mismatch where `{ flip
a: UInt<1> }` is not equal to `{ a: UInt<1> }`.  However, type canonicalization
would transform this circuit into the following circuit:

```firrtl
module Foo:
  input a: { a: UInt<1> }
  output b: { a: UInt<1> }

  b <= a
```

Here, the connection `b <= a` _is legal_ FIRRTL.  This then makes it impossible
for a type canonical form to be type checked.

As an example of the second problem, consider the following circuit:

```firrtl
module Bar:
  output a: { flip a: UInt<1> }
  input b: { flip a: UInt<1> }

  b <= a
```

Here, the connection `b <= a` _is illegal_ FIRRTL because `b` is a source and
`a` is a sink.  However, type canonicalization converts this to the following
circuit:

```firrtl
module Bar:
  input a: { a: UInt<1> }
  output b: { a: UInt<1> }

  b <= a
```

Here, the connect `b <= a` _is legal_ FIRRTL because `b` is now a sink and `a`
is now a source.  This then makes it impossible for a type canonical form to be
flow checked.

As an example of the third problem, consider the following circuit:

```firrtl
module Baz:
  wire a: {flip a: {flip a: UInt<1>}}
  wire b: {flip a: {flip a: UInt<1>}}

  b.a <= a.a
```

The connection `b.a <= a.a`, when lowered, results in the _reverse_ connect
`a.a.a <= b.a.a`.  However, type canonicalization will remove the flips from the
circuit to produce:

```firrtl
module Baz:
  wire a: {a: {a: UInt<1>}}
  wire b: {a: {a: UInt<1>}}

  b.a <= a.a
```

Here, the connect `b.a <= a.a`, when lowered, results in the normal connect
`b.a.a <= a.a.a`.  Type canonicalization has thereby changed the semantics of
connect.

Due to the elegance of type canonicalization, we initially decided that we would
use type canonicalization and CIRCT would accept more circuits than the SFC.
The third problem (identified much later than the first two) convinced us to
remove type canonicalization.

For a historical discussion of type canonicalization see:

- [`llvm/circt#380`](https://github.com/llvm/circt/issues/380)
- [`llvm/circt#919`](https://github.com/llvm/circt/issues/919)
- [`llvm/circt#944`](https://github.com/llvm/circt/pull/944)

### Flow

The FIRRTL specification describes the concept of "flow".  Flow encodes
additional information that determines the legality of operations.  FIRRTL
defines three different flows: `sink`, `source`, and `duplex`.  Module inputs,
instance outputs, and nodes are `source`, module outputs and instance inputs are
`sink`, and wires and registers are `duplex`.  A value with `sink` flow may only
be written to, but not read from (with the exception of module outputs and
instance inputs which may be also read from).  A value with `source` flow may be
read from, but not written to.  A value with `duplex` flow may be read from or
written to.

For FIRRTL connect statements, it follows that the left-hand-side must be `sink`
or `duplex` and the right-hand-side must be `source`, `duplex`, or a
port/instance `sink`.

Flow is _not_ represented as a first-class type in CIRCT.  We instead provide
utilities for computing flow when needed, e.g., for connect statement
verification.

### Non-FIRRTL Types

The FIRRTL dialect has limited support for foreign types, i.e., types that are defined outside the FIRRTL dialect. Almost all operations expect to be dealing with FIRRTL types, especially those that are sensitive to the type they operate on, like `firrtl.add` or `firrtl.connect`. However, a restricted set of operations allows for simple pass-through semantics of foreign types. These include the following:

- Ports on a `firrtl.module`, where the foreign types are treated as opaque values moving in and out of the module
- Ports on a `firrtl.instance`
- `firrtl.wire` to allow for def-after-use cases; the wire must have a single strict connect that uniquely defines the wire's value
- `firrtl.strictconnect` to module outputs, instance inputs, and wires

The expected lowering for strict connects is for the connect to be eliminated and the right-hand-side source value of the connect being instead materialized in all places where the left hand side is used. Basically we want wires and connects to disappear, and all places where the wire is "read" should instead read the value that was driven onto the wire.

The reason we provide this foreign type support is to allow for partial lowering of FIRRTL to HW and other dialects. Passes might lower a subset of types and operations to the target dialect and we need a mechanism to have the lowered values be passed around the FIRRTL module hierarchy untouched alongside the FIRRTL ops that are yet to be lowered.

### Const Types

FIRRTL hardware types can be specified as `const`, meaning they can only be assigned compile-time constant values or values of other `const` types.

## Operations

### Multiple result `firrtl.instance` operation

The FIRRTL spec describes instances as returning a bundle type, where each
element of the bundle corresponds to one of the ports of the module being
instanced.  This makes sense in the Scala FIRRTL implementation, given that it
does not support multiple ports.

The MLIR FIRRTL dialect takes a different approach, having each element of the
bundle result turn into its own distinct result on the `firrtl.instance`
operation.  This is made possible by MLIR's robust support for multiple value
operands, and makes the IR much easier to analyze and work with.

### Module bodies require def-before-use dominance instead of allowing graphs

MLIR allows regions with arbitrary graphs in their bodies, and this is used by
the HW dialect to allow direct expression of cyclic graphs etc.  While this
makes sense for hardware in general, the FIRRTL dialect is intended to be a
pragmatic infrastructure focused on lowering of Chisel code to the HW dialect,
it isn't intended to be a "generally useful IR for hardware".

We recommend that non-Chisel frontends target the HW dialect, or a higher level
dialect of their own creation that lowers to HW as appropriate.

### `input` and `output` Module Ports

The FIRRTL specification describes two kinds of ports: `input` and `output`.  In
the `firrtl.module` declaration we track this via an arbitrary precision integer
attribute (`IntegerAttr`) where each bit encodes the directionality of the port
at that index.

Originally, we encoded direction as the absence of an outer flip type (input) or
presence of an outer flip type (output).  This was done as part of the original
type canonicalization effort which combined input/output with the type system.
However, once type canonicalization was removed flip type only became used in
three places: on the types of bundle fields, on the variadic return types of
instances or memories, and on ports.  The first is the same as the FIRRTL
specification.  The second is a deviation from the FIRRTL specification, but
allowable as it takes advantage of the MLIR's variadic capabilities to simplify
the IR.  The third was an inelegant abuse of an unrelated concept that added
bloat to the type system.  Many operations would have to check for an outer flip
on ports and immediately discard it.

For this reason, the `IntegerAttr` encoding implementation was chosen.

For a historical discussion of this issue and its development see:

- [`llvm/circt#989`](https://github.com/llvm/circt/issues/989)
- [`llvm/circt#992`](https://github.com/llvm/circt/pull/992)

### `firrtl.bitcast`

The bitcast operation represents a bitwise reinterpretation (cast) of a value.
It can be used to cast a vector or bundle type to an int type or vice-versa.
The bit width of input and result types must be known.  For an aggregate type,
the bit width of every field must be known.  This always synthesizes away in
hardware, and follows the same endianness policy as `hw.bitcast`.

### `firrtl.mem`

Unlike the SFC, the FIRRTL dialect represents each memory port as a distinct
result value of the `firrtl.mem` operation.  Also, the `firrtl.mem` node does
not allow zero port memories for simplicity.  Zero port memories are dropped by
the .fir file parser.

In the FIRRTL pipeline, the `firrtl.mem` op can be lowered into either a
external module for macro replacement or a register of vector type. The
conditions for macro replacement are as follows:

1. `–replSeqMem` option is passed and
2. `readLatency == 1`  and
3. `writeLatency == 1` and
4. `width(data) > 0`

Any `MemOp` not satisfying the above conditions is lowered to Register vector.

#### MemToRegOfVec transformation outline:

The `MemToRegOfVec` pass runs early in the pipeline, after the `LowerCHIRRTL`
pass and right before the `InferResets` pass.

1. Select all MemOps that are not candidates for macro replacement,
2. Create a reg
3. Read ports return the value at the address when the enable signal is high.

```c++
if (enable) {
  readOut = register[address]
}
```

4. Write ports store the value at the address when the mask signal is high.

```c++
if (enable) {
  if (mask[0])
    register[0] = dataIn[0]
  if (mask[1])
    register[1] = dataIn[1]
}
```

#### Handling of MemTaps

The `sifive.enterprise.grandcentral.MemTapAnnotation` annotation is attached to
the `MemOp` and the corresponding Memtap module ports. After lowering the memory
to registers, this annotation must be properly scattered such that
GrandCentralTaps can generate the appropriate code.

The memtap module has memtap annotations, where the number of ports with the
annotation is equal to the memory depth. In the `MemToRegOfVec` transformation,
after lowering the memory to the register vector, a subannotation is created for
each sub-field of the data and the
`sifive.enterprise.grandcentral.MemTapAnnotation` annotation is copied from the
original `MemOp`. The `LowerTypes` pass will handle the subannotations
appropriately.

#### Interaction with AsyncReset Inference

The `AsyncReset` pass runs right after the `MemToRegOfVec`.  It will transform
the memory registers to async registers if the corresponding annotations are
present.  Only if a `MemOp` had
`sifive.enterprise.firrtl.ExcludeMemFromMemToRegOfVec`, annotation, then it is
not converted to an async reset register.

#### `firrtl.mem` Attributes

A `firrtl.mem` has the following properties:

1. Data type
2. Mask bitwidth
3. Depth
4. Name
5. Number of read ports, write ports, read-write ports
6. Read under write behavior
7. Read latency
8. Write latency

##### Mask bitwidth

Any aggregate memory data type is lowered to ground type by the `LowerTypes`
pass. After lowering the data type, the data bitwidth must be divisible by mask
bitwidth. And we define the property granularity as: `mask granularity = (Data
bitwidth)/(Mask bitwidth)`.

Each mask bit can guard the write to `mask granularity` number of data bits.
For a single-bit mask, one-bit guards write to the data, hence `mask granularity
= data bitwidth`.

#### Macro replacement

Memories that satisfy the conditions above are candidates for macro replacement.

A memory generator defines the external module definition corresponding to the
memory for macro replacement. Memory generators need metadata to generate the
memory definition. SFC uses some metadata files to communicate with the memory
generators.

`<design-name>.conf` is a file, that contains the metadata for the memories
which are under the "design-under-test" module hierarchy. Following is a sample
content of the file:

```
name dir_ext depth 512 width 248 ports mrw mask_gran 31
name banks_0_ext depth 2048 width 72 ports rw
name banks_1_ext depth 2048 width 72 ports rw
```

1. `name` followed by the memory name.
2. `depth` followed by the memory depth.
3. `width` followed by the data bitwidth.
4. `ports` followed by the `mrw` for read-write port, `mwrite` for a write port
   and `read` for a read port.
5. `mask_gran` followed by the mask granularity.

### CHIRRTL Memories

FIRRTL has two different representations of memories: Chisel `cmemory`
operations, `smem` and `cmem`, and the standard FIRRTL `mem` operation.  Chisel
memory operations exist to make it easy to produce FIRRTL code from Chisel, and
closely match the Chisel API for memories. Chisel memories are intended to be
replaced with standard FIRRTL memories early in the pipeline.  The set of
operations related to Chisel memories are often referred to as CHIRRTL.

The main difference between Chisel and FIRRTL memories is that Chisel memories
have an operation to add a memory port to a memory, while FIRRTL memories
require all ports to be defined up front. Another difference is that Chisel
memories have "enable inferrence", and are usually inferred to be enabled where
they are declared. The following example shows a CHIRRTL memory declaration, and
the standard FIRRTL memory equivalent.

```firrtl
smem mymemory : UInt<4>[8]
when p:
  read mport port0 = mymemory[address], clock
```

```firrtl
mem mymemory:
    data-type => UInt<4>
    depth => 8
    read-latency => 0
    write-latency => 1
    reader => port0
    read-under-write => undefined

mymemory.port0.en <= p
mymemory.port0.clk <= clock
mymemory.port0.addr <= address
```

FIRRTL memory operations were created because it was thought that a concrete
memory primitive, that looks like an instance, is a better design for a compiler
IR.  It was originally intended that Chisel would be modified to emit FIRRTL
memory operations directly, and the CHIRRTL operations would be retired.  The
lowering from Chisel memories to FIRRTL memories proved far more complicated
than originally envisioned, specifically surrounding the type of ports,
inference of enable signals, and inference of clocks.

CHIRRTL operations have since stuck around, but their strange behavior has lead
to discussions to remove, improve, or totally redesign them.  For some current
discussion about this see [^0], [^1]. Since CIRCT is attempting to be a drop in
replacement FIRRTL compiler, we are not attempting to implement these new ideas
for Chisel memories. Instead, we are trying to implement what exists today.

There is, however, a major compatibility issue with the existing implementation
of Chisel memories which made them difficult to support in CIRCT.  The FIRRTL
specification disallows using any declaration outside of the scope where it is
created.  This means that a Chisel memory port declared inside of a `when` block
can only be used inside the scope of the `when` block.  Unfortunately, this
invariant is not enforced for memory ports, and this leniency has been abused by
the Chisel standard library. Due to the way clock and enable inference works, we
couldn't just hoist the declaration into the outer scope.

To support escaping memory port definitions, we decided to split the memory port
operation into two operations.  We created a `chirrtl.memoryport` operation to
declare the memory port, and a `chirrtl.memoryport.access` operation to enable
the memory port. The following is an example of how FIRRTL translates into the
CIRCT dialect:

```firrtl
smem mymem : UInt<1>[8]
when cond:
  infer mport myport = mymem[addr], clock
out <= myport
```

```mlir
%mymem = chirrtl.seqmem Undefined  : !chirrtl.cmemory<uint<1>, 8>
%myport_data, %myport_port = chirrtl.memoryport Infer %mymem {name = "myport"}  : (!chirrtl.cmemory<uint<1>, 8>) -> (!firrtl.uint<1>, !chirrtl.cmemoryport)
firrtl.when %cond : !firrtl.uint<1> {
  chirrtl.memoryport.access %myport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<3>, !firrtl.clock
}
firrtl.connect %out, %myport_data : !firrtl.uint<1>, !firrtl.uint<1
```

The CHIRRTL operations and types are contained in the CHIRRTL dialect.  The is
primary reason to move them into their own dialect was to keep the CHIRRTL types
out of the FIRRTL dialect type hierarchy. We tried to have the CHIRRTL dialect
depend on the FIRRTL dialect, but the flow checking in FIRRTL had to know about
CHIRRTL operations, which created a circular dependency.  To simplify how this
is handled, both dialects are contained in the same library.

For a historical discussion of this issue and its development see
[`llvm/circt#1561`](https://github.com/llvm/circt/issues/1561).

[^0]: https://github.com/chipsalliance/firrtl/issues/727
[^1]: https://github.com/chipsalliance/firrtl/pull/1821

### More things are represented as primitives

We describe the `mux` expression as "primitive", whereas the IR spec and grammar
implement it as a special kind of expression.

We do this to simplify the implementation: These expressions have the same
structure as primitives, and modeling them as such allows reuse of the parsing
logic instead of duplication of grammar rules.

### `invalid` Invalidate Operation is an expression

The FIRRTL spec describes an `is invalid` statement that logically computes an
invalid value and connects it to `x` according to flow semantics.  This behavior
makes analysis and transformation a bit more complicated, because there are now
two things that perform connections: `firrtl.connect` and the `is invalid`
operation.

To make things easier to reason about, we split the `is invalid` operation into
two different ops: an `firrtl.invalidvalue` op that takes no operands and
returns an invalid value, and a standard `firrtl.connect` operation that
connects the invalid value to the destination (or a `firrtl.attach` for analog
values).  This has the same expressive power as the standard FIRRTL
representation but is easier to work with.

During parsing, we break up an `x is invalid` statement into leaf connections.
As an example, consider the following FIRRTL module where a bi-directional
aggregate, `a` is invalidated:

```firrtl
module Foo:
  output a: { a: UInt<1>, flip b: UInt<1> }

  a is invalid
```

This is parsed into the following MLIR.  Here, only `a.a` is invalidated:

``` mlir
firrtl.module @Foo(out %a: !firrtl.bundle<a: uint<1>, b: flip<uint<1>>>) {
  %0 = firrtl.subfield %a[a] : !firrtl.bundle<a: uint<1>, b: flip<uint<1>>>
  %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
  firrtl.connect %0, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
}
```

### Inline SystemVerilog through `verbatim.expr` operation

The FIRRTL dialect offers a `firrtl.verbatim.expr` operation that allows for
SystemVerilog expressions to be embedded verbatim in the IR. It is lowered to
the corresponding `sv.verbatim.expr` operation of the underlying SystemVerilog
dialect, which embeds it in the emitted output. The operation has a FIRRTL
result type, and a variadic number of operands can be accessed from within the
inline SystemVerilog source text through string interpolation of `{{0}}`-style
placeholders.

The rationale behind this verbatim operation is to offer an escape hatch
analogous to `asm ("...")` in C/C++ and other languages, giving the user or
compiler passes full control of what exactly gets embedded in the
output. Usually, though, you would rather add a new operation to the IR to
properly represent additional constructs.

As an example, a verbatim expression could be used to interact with
yet-unsupported SystemVerilog constructs such as parametrized class typedef
members:

```mlir
firrtl.module @Magic (out %n : !firrtl.uint<32>) {
  %0 = firrtl.verbatim.expr "$bits(SomeClass #(.Param(1))::SomeTypedef)" : !firrtl.uint<32>
  firrtl.connect %n, %0 : !firrtl.uint<32>, !firrtl.uint<32>
}
```

This would lower through the other dialects to SystemVerilog as you would
expect:

```systemverilog
module Magic (output [31:0] n);
  assign n = $bits(SomeClass #(.Param(1))::SomeTypedef);
endmodule
```

## Interpretation of Undefined Behavior

The [FIRRTL
Specification](https://github.com/chipsalliance/firrtl/blob/master/spec/spec.pdf)
has undefined behavior for certain features.  For compatibility reasons, FIRRTL
dialect _always_ chooses to implement undefined behavior in the same manner as
the SFC.

### Invalid

The SFC has multiple context-sensitive interpretations of invalid.  Failure to
implement all of these can result in formal equivalence failures when comparing
CIRCT-generated Verilog with SFC-generated Verilog.  A list of these
interpretations is enumerated below and then described in more detail.

1. An invalid value driving the initialization value of a register (looking
   through wires and connections within module scope) removes the reset from the
   register.
1. An invalid value used in a `when`-encoded multiplexer tree results in a
   direct connection to the non-invalid leg of the multiplexer.
1. Any other use of an invalid value is treated as constant zero.

Interpretation (1) is a mechanism to remove unnecessary reset connections in a
circuit as fewer resets can enable a higher performance design.  The SFC
implementation of this works as a dedicated pass that does a module-local
analysis looking for registers with resets whose initialization values come from
invalidated signals.  This analysis only looks through wires and connections.
It is legal to use an invalidated output port or instance input port.

As an example, the following module should have register `r` converted to a
reset-less register:

```firrtl
wire inv: UInt<8>
inv is invalid

wire tmp: UInt<8>
tmp <= inv

reg r: UInt<8>, clock with : (reset => (reset, tmp))
```

Notably, if `tmp` is a `node`, this optimization should not be performed.

Interpretation (2) means that the following circuit should be optimized to a
direct connection from `bar` to `foo`:

```firrtl
foo is invalid
when cond:
  foo <= bar
```

Note that the SFC implementation of this optimization is handled via two passes.
An `ExpandWhens` (later refactored as `ExpandWhensAndCheck`) pass converts all
`when` blocks to multiplexer trees.  Any invalid values that arise from this
conversion produce `validif` expressions.  (This is the "conditionally valid"
expression which is an internal detail of the SFC which was removed from the
FIRRTL specification.)  A later pass, `RemoveValidIfs` optimizes/removes
`validif` by replacing it with a direct connection.

It is important to note that the above formulations using `when` or the
SFC-internal representation using `validif` _are not equivalent to a mux
formulation_ like the following.  The code below should be optimized using
Interpretation (3) of invalid as constant zero:

```firrtl
wire inv: UInt<8>
inv is invalid

foo <= mux(cond, bar, inv)
```

A legal lowering of this is only to:

```firrtl
foo <= mux(cond, bar, UInt<8>(0))
```

Interpretation (3) is used in all other situations involving an invalid value.

**Critically, the nature of an invalid value has context-sensitive information
that relies on the exact structural nature of the circuit.**  It follows that
any seemingly mundane optimization can result in an end-to-end miscompilations
where the SFC is treated as ground truth.

As an example, consider a reformulation of the `when` example above, but using a
temporary, single-use, invalidated wire:

```firrtl
wire inv: UInt<8>
inv is invalid

b <= inv
when cond:
  b <= a
```

This should _not_ produce a direction connection to `b` and should instead lower
to:

```firrtl
b <= mux(cond, a, inv)
```

It follows that interpretation (3) will then convert the false leg of the `mux`
to a constant zero.

## Intrinsics

Intrinsics are implementation-defined constructs.  Intrinsics provide a way to
extend the system with funcitonality without changing the langauge.  They form
an implementation-specific built-in library.  Unlike traditional libraries,
implementations of intrinsics have access to internals of the compiler, allowing
them to implement features not possible in the language.
