# HW Dialect Rationale

This document describes various design points of the `hw` dialect as well as
global perspective on the `hw`, `comb`, and `sv` dialects, why
they are the way they are, and current status.  This follows in the spirit of
other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).  For more
information about the other dialects, please see the
[Comb Dialect Rationale](../Comb/RationaleComb.md) and [SV Dialect
Rationale](../SV/RationaleSV.md).

- [HW Dialect Rationale](#hw-dialect-rationale)
  - [General Introduction](#general-introduction)
  - [Introduction to the `hw` Dialect](#introduction-to-the-hw-dialect)
  - [`hw` Type System](#hw-type-system)
  - [`hw.module` and `hw.instance`](#hwmodule-and-hwinstance)
    - [Instance paths](#instance-paths)
  - [Parameterized Modules](#parameterized-modules)
    - [Valid Parameter Expression Attributes](#valid-parameter-expression-attributes)
    - [Parameter Expression Canonicalization](#parameter-expression-canonicalization)
    - [Using parameters in the body of a module](#using-parameters-in-the-body-of-a-module)
    - [Parameterized Types](#parameterized-types)
    - [Answers to other common questions](#answers-to-other-common-questions)
  - [Type declarations](#type-declarations)
    - [Type declaration IR](#type-declaration-ir])
    - [Type declaration System Verilog output](#type-declaration-system-verilog-output)
    - [Type canonicalization](#type-canonicalization)
    - [Open Issues](#open-issues)
  - [Symbols and Visibility](#symbols-and-visibility)
  - [Future Directions](#future-directions)

## General Introduction

[SystemVerilog](https://en.wikipedia.org/wiki/SystemVerilog) is an industry
standard language for hardware design and verification, is known by a large
number of engineers who write it manually, and is
an important interchange format between EDA tools.  However, while it is
ubiquitous, SystemVerilog is not easy to generate or transform.  Furthermore, it
is non-trivial for compiler tools to generate high-quality human-readable
SystemVerilog.

The `hw`, `comb` and `sv` dialects attempt to address these problems with
several major contributions: 

 1) The `hw` dialect provides unifying structure and
    abstractions that are useful for a wide range of hardware modeling problems.
    It allows other dialects to "mix in" with it to provide higher level
    functionality. `hw` is roughly akin to the "std" dialect in MLIR (but better
    curated).
 2) The `comb` dialect provides a common set of operations for combinational
    logic.  This dialect is designed to allow easy analysis and transformation.
 3) The `sv` dialect provides direct access to a wide variety
    of SystemVerilog constructs, including behavioral constructs, syntactic
    sugar constructs, and even idioms like `ifdef` blocks.
 4) The `circt` project provides a high quality implementation and a number of useful compiler
    passes for analyzing and transforming these dialects, and a SystemVerilog
    emitter that generates pretty output.

The combination of these capabilities provides a useful suite of functionality
for compiler tools that want to generate high quality SystemVerilog.

## Introduction to the `hw` Dialect

The `hw` dialect defines a set of common functionality, such as `hw.module` and 
`hw.instance` for representing hardware modules, as well as common types (e.g. 
`hw.array<xx>`) and attributes.   It is *not* designed to model SystemVerilog or
any other hardware
design language directly, and doesn't contain combinational or sequential
operations and does not have "connect" semantics.  Instead, it is designed to be
a flexible and extensible substrate that may be extended with higher level
dialects mixed into it (like `sv`, `comb`, and `seq` in the future, etc).

## `hw` Type System

TODO: Describe inout types.  Analogy to lvalues vs rvalues.  Array indices for
both forms.  Arrays, structs,
moving [UnpackedArray](https://github.com/llvm/circt/issues/389) to SV someday.

CLEANUP: InOut types is defined in the `hw` dialect, but logically lives at the
`sv` dialect level.  `sv` provides connects, wires and other syntactic
constructs that work with the inout type.  These aren't necessary
for combinational logic, but are nonetheless pretty useful when generating
Verilog.

### `enum` Type

Enum types have the property that the bit width of the type is the minimum 
necessary to hold the tag values.  Tag values are either explicit or 
sequentially numbered in tag order from 0.  Enum tags are unsigned values.

### `union` Type

Union types contain a single data element (which may be an aggregate).  They
optionally have an offset per varient which allows non-SV layouts.

## `hw.module` and `hw.instance`

The basic structure of a hardware design is made up an "instance tree" of
"modules" and "instances" that refer to them.  There are loose analogies to
software programs which have corresponding "functions" and "calls" (but there
are also major differences, see "[Instance paths](#instance-paths)" below).
Modules can have a
definition `hw.module`, they can be a definition of an external module whose
signature is known but whose body is provided separately `hw.module.extern`,
and can be a definition of an external module with a known signature that
can/will be generated in the future on demand (`hw.module.generated`).

A simple example module looks like this (many more can be found in the
testsuite):

```mlir
hw.module @two_and_three(%in: i4) -> (twoX: i4, threeX: i4) {
  %0 = comb.add %in, %in : i4
  %1 = comb.add %a, %0 : i4
  hw.output %0, %1 : i4, i4
}
```

The signature of a modules have these major components:

1) A symbol `name` which specifies the MLIR name for the module
   (`@two_and_three` in the example above).  This is what connects instances to
   modules in a stable way.
2) A list of input ports, each of which has a type (`%in: i4` in the example
   above).  Each input port is available as an SSA value through a block
   argument in the entry block of an `hw.module`, allowing them to be used
   within its body.  "inout" ports are modeled as inputs with an `!hw.inout<T>`
   type.  Input port names are prefixed with a `%` because they are available
   as SSA values in the body.
3) A list of result port names and types (`twoX: i4` and `threeX: i4` in the
   example above).  In a `hw.module` definition, the values for the
   results are provided by the operands to the `hw.output` terminator in the
   body block.  The names of result ports are not prefixed with `%` because
   they are not MLIR SSA values.
4) A list of module "parameters", which provide parametric polymorphism
   capabilities (somewhat similar to C++ templates) for modules.  These are
   described in more detail in the "[Parameterized 
   Modules](#parameterized-modules) section below.
5) The `verilogName` attribute can be used to override the name for an external
   module.  TODO: we should eliminate this in the future and just use the symbol.
6) Other ad-hoc attributes.  The `hw` dialect is intended to allow open
   extensibility by other dialects.  Ad-hoc attributes put on `hw` dialect 
   modules should be namespace qualified according to the dialect they come
   from to avoid conflicts.

This definition is fairly close to the Verilog family, but there are some
notable differences: for example:

 - We split output ports from input ports, don't use `hw.output` instead of 
   connects to specify the results.  This allows better SSA dataflow
   analysis from the `hw.output` which is useful for inter-module analyses.
 - We allow arbitrary types for module ports.  The `hw` dialect is generally
   designed to be extensible by other dialects, and thus being permissive here
   is useful.  That said, the [Verilog exporter](../../VerilogGeneration.md) does not
   support arbitrary user-defined types.
 - The `comb` dialect in particular does not use signed integer types, but do
   support zero-width integer types.  Modules in the `hw` dialect,
   support both of these.  Zero width ports and wires are omitted (printed as
   comments) when generating Verilog.

### GlobalRefOp
The GlobalRefOp operation (`hw.globalRef`) can be used to identify the unique
 instance path of an operation globally.
`hw.globalRef` can be used to attach nonlocal annotations in FIRRTL dialect
and also for metadata emission.
`hw.globalRef` defines a symbol and contains a list of module local
`hw.innerNameRef` symbols to define the instance path.
 For example, in the following example, `@glbl_B_M1` specifies instance
 "h1" in module `@A`, followed by instance
 "M1" in module `@B`.

`hw.globalRef` can define a unique instance path, and each element along the way
 carries an attribute `circt.globalRef`, pointing to the global op. 
 Thus instances participating in nonlocal paths are readily apparent.

``` mlir
  hw.globalRef @glbl_B_M1 [#hw.innerNameRef<@A::@inst_1>, #hw.innerNameRef<@B::@memInst>]
  hw.globalRef @glbl_D_M1 [#hw.innerNameRef<@A::@inst_0>, #hw.innerNameRef<@C::@inst>, #hw.innerNameRef<@D::@memInst>]
  hw.globalRef @glbl_D_M2 [#hw.innerNameRef<@A::@SF>, #hw.innerNameRef<@F::@symA>]
  hw.globalRef @glbl_D_M3 [#hw.innerNameRef<@A::@SF>, #hw.innerNameRef<@F::@symB>]
  hw.module @D() -> () {
    hw.instance "M1" sym @memInst @FIRRTLMem() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_D_M1>]}
  }
  hw.module @B() -> () {
     hw.instance "M1" sym @memInst @FIRRTLMem() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_B_M1>]}
  }
  hw.module @C() -> () {
    hw.instance "m" sym @inst @D() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_D_M1>]}
  }
  hw.module @A() -> () {
    hw.instance "h1" sym @inst_1 @B() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_B_M1>]}
    hw.instance "h2" sym @inst_0 @C() -> () {circt.globalRef = [#hw.globalNameRef<@glbl_D_M1>]}
    %c0 = hw.constant 0 : i1
    %2 = hw.instance "ab" sym @SF  @F (a1: %c0: i1) -> (a2 : i1) {circt.globalRef = [#hw.globalNameRef<@glbl_D_M2>, #hw.globalNameRef<@glbl_D_M3>]}
  }
  hw.module.extern  @F(%a1: i1 {hw.exportPort = @symA, circt.globalRef = [#hw.globalNameRef<@glbl_D_M2>]}) -> (a2: i1 {hw.exportPort = @symB, circt.globalRef = [#hw.globalNameRef<@glbl_D_M3>]}) attributes {}

sv.verbatim "{{0}}" { symbols = [@glbl_D_M1] }
sv.verbatim "{{0}}" { symbols = [@glbl_B_M1] }
```

### Instance paths

An IR for Hardware is different than an IR for Software in a very important way:
while each function in a software program usually compiles into one blob of
binary code no matter how many times it is called, each instance in a hardware
design is typically fully instantiated, because different instances turn into
different gates.  The consequence of this is that the instance tree is really a
compression mechanism that is eventually elaborated away.

This compression approach has major advantages: it is much better for memory
and compile time to represent a single definition of a hardware block than the 
(possibly thousands or millions) of concrete instances that will eventually be
required.  However, hardware engineers often do need to reason about and control
the different instances in some cases (e.g. providing physical layout
constraints for one instance but not the rest).

TODO: Bake out a design for instance path references, an equivalent to the
FIRRTL dialect `InstanceGraph` type, etc.

## Parameterized Modules

The `hw` dialect supports parametric "compile-time" polymorphism for modules.
This allows for metaprogramming along the instance tree, guaranteed
"instantiation time" optimizations and code generation, further enables
the "IR compression" benefits of using instances in the first place, and enables
the generation of parameters in generated Verilog (which can increase the
perceived readability of the generated code).

Parameters are declared on modules (including generated and external ones)
with angle brackets: each parameter has a name and type, and can optionally
have a default value.  Instances of a parameterized module provide a value for
each parameter (even defaulted ones) in the same order:

```mlir
// This module has two parameters "p1" and "p2".
hw.module.extern @parameterized<p1: i42 = 17, p2: i1>(%in: i8) -> (out: i8)

hw.module @UseParameterized(%a: i8) -> (ww: i8) {
  %r0 = hw.instance "inst" @parameters<p1: i42 = 17, p2: i1 = 1>(in: %a: i8) -> (out: i8)
  hw.output %r0 : i8
}
```

This approach makes analysis and transformation of the IR simple, predictable,
and efficient: because the parameter list on instances and on modules always
line up, they are indexable by integers (instead of strings), intermodule
analysis is straight-forward (no filling in of default values etc), and
Verilog generation is always predictable: the default value for a parameter
is used when the instance and the module default are the same (e.g. in the
example above, `p1` is not printed at the instance site because it is the same
as the default.

The `sv` dialect provides the `sv.localparam` operation, which is used for
naming constants.  These may be derived from module parameters or may just be
nicely named constants intended to improve readability.  This is part of the
`sv` dialect (not the `hw` dialect) because it only makes sense as a concept
when generating Verilog.

**Compatibility with classic Verilog**

Note that typed parameters are a SystemVerilog extension, many Verilog-only
tools do not support them.  If you need compatibility with Verilog-only tools,
please use `i32` integer, `f64`, and string parameters only.  All module
parameter definitions
[should have default values](https://github.com/llvm/circt/issues/1848#issuecomment-933558192) for maximum tool compatibility as well, even if
they are always overridden.

**TODO:** CIRCT could do parameter legalization to convert parameters smaller or
larger than 32-bits to 32-bits when generating Verilog-only output.

### Valid Parameter Expression Attributes

The following attributes may be used as expressions involving parameters at
an instance site or in the default value for a parameter declaration on a
module:

- IntegerAttr/FloatAttr/StringAttr constants may be used as simple leaf values.
- The `#hw.param.decl.ref` attribute is used to refer to the value of a
  parameter in the current module.  This is valid in most positions where a
  parameter attribute is used - except in the default value for a module.
- The `#hw.param.expr` operator allows combining other parameter expressions
  into an expression tree.  Expression trees have important canonicalization
  rules to ensure important cases are canonicalized to uniquable
  representations.
- `#hw.param.verbatim<"some string">` may be used to provide an opaque blob of
  textual Verilog that is uniqued by its string contents.  This is intended
  as a general "escape hatch" that allows frontend authors to express anything
  Verilog cannot, even if first-class IR support doesn't exist yet.  CIRCT does not
  provide any checking to ensure that this is correct or safe, and assumes it
  is single expression - parenthesize the string contents if not to be safe.
  This [should eventually support
  substitutions](https://github.com/llvm/circt/issues/1881) like
  `sv.verbatim.expr`.

Because parameter expressions are MLIR attributes, they are immortal values
that are uniqued based on their structure.  This has several important
implications, including:

- A parameter reference (`#hw.param.decl.ref`) to a parameter `x` doesn't know
  what module it is in.  The verifier checks that parameter expressions are
  valid within the body of a module, and that the types line up between the
  parameter reference and the declaration (after all, two different modules can
  have two different parameters named `x` with different types).
- We want to depend on MLIR canonicalizing and uniquing the pointer address of
  attributes in a predictable way to ensure that further derived uniqued objects
  (e.g. a parameterized integer type) is also uniqued correctly.  For example,
  we do not want the types `hw.int<x+1>` and `hw.int<1+x>` to turn into
  different types.  See the [Parameter Expression
  Canonicalization](#parameter-expression-canonicalization) section below for
  more details.
- Whereas the rest of the `hw` dialect is generally open for extension, the
  current grammar of attribute expressions is closed: you have to hack the
  HW dialect verifier and VerilogEmitter to add new kinds of valid expressions.
  This is considered a limitation, we'd like to move to an attribute interface
  at some point that would allow dialect-defined attributes.  For example, this
  would allow  moving `hw.param.verbatim` attribute down to the `sv` dialect.

Note that there is no parameter expression equivalent for `comb.sub`:
`(sub x, y)` is represented with `(add x, (mul y, -1))` which makes maintaining
canonical form simpler and more consistent.

### Parameter Expression Canonicalization

As mentioned above, it is important to canonicalize parameter expressions.  This
slightly reduces memory usage, but more importantly ensures that equivalent
parameter expressions are pointer equivalent: we don't want `x+1` and `1+x` to
be different, because that would cause everything derived from them to be as
well.

On the other hand, we expect to support a lot of weird expressions over time (at
least the full complement that Verilog supports) and canonicalizing arbitrary
expressions in a predictable way is untenable.  As such, we support
canonicalizing a fixed set of expressions predictably: more may be added in
the future.

This set includes:

 - Constant folding: parameter expressions with all integer constant operands
   are folded to their corresponding result.
 - Constant identities are simplified, e.g. `p1 & 0` into `0`, `p1 * 1` into
   `p1`, `p1 << 0` into `p1`, `p1 >> 0` into `p1`, `p1 / 1` into `p1` and 
   `p1 % 1` into `0`.
 - Constant operand merging: any constant operands in associative operations are
   merged into a single operand and moved to the right, e.g. `(add 4, x, 2)` 
   into `(add x, 6)`.
 - Fully associative operators flatten subexpressions, e.g.
   `(add x, (add y, z))` into `(add x, y, z)`.
 - We simplify affine expressions into a sum of products representation, pulling
   additions out of products, e.g. `(a+b)*c*d` into `(a*c*d + b*c*d)`
 - Operands of fully-associative expressions are put into a stable order, at
   least for the case of affine expressions involving constant integers and
   named parameters.  For example `p2+p1` turns into `p1+p2` reliably.  The
   actual ordering moves subexpressions to the start of the list (more complex
   ones first) followed by verbatims, followed by parameter references, followed
   by constants, each group sorted w.r.t. each other.
 - Common operand factoring for adds, e.g. `(a+b+a)` into `(2*a + b)` and
   `(a*4 + a)` into `(a*5)`
 - Shift left by constant is canonicalized into multiply to compose correctly
   with affine expression canonicalization, e.g. `(shl x, 1)` into
   `(mul x, 2)`.
 - As mentioned above, `sub` is not supported as a ParamExprAttr - it is
   represented as multiply by `-1`.  This allows it to trivially compose
   with affine expression canonicalizations.

### clog2 Parameter Expression
A `clog2` parameter expression opcode is provided, which calculates the ceiling
of log base 2 of its argument. This is useful for calculating things like the
minimum bitwidth needed to address memory of a parameterized size.

Note that `clog2(0)` is `0`, which follows the Verilog spec.

### Using parameters in the body of a module

Parameters are not [SSA values](https://en.wikipedia.org/wiki/Static_single_assignment_form), so they cannot directly be used within the body
of the module.  Just like you use `hw.constant` to project a constant integer
value into the SSA domain, you can use the `hw.param.value` to project a
parameter expression, like so:

```mlir
hw.module @M1<param1: i1>(%clock : i1, ...) {
  ...
  %param1 = hw.param.value i1 = #hw.param.decl.ref<"param1">
  ...
    sv.if %param1 {  // Compile-time conditional on parameter.
      sv.fwrite "Only happens when the parameter is set\n"
    }
  ...
}
```

Alternately, you can project them with a specific name, you can use the
`sv.localparam` declaration like so:

```mlir
hw.module @M1<param1: i1>(%clock : i1, ...) {
  ...
  %param1 = sv.localparam : i1 { value = #hw.param.decl.ref<"param1">: i1 }
  ...
    sv.if %param1 {  // Compile-time conditional on parameter.
      sv.fwrite "Only happens when the parameter is set\n"
    }
  ...
}
```

Using `sv.localparam` is helpful when you're looking to produce specifically
pretty Verilog for human consumption.  The optimizer won't fold aggressively
around these names.

### Parameterized Types

TODO: Not done yet.

### Answers to other common questions

During the design work on parameterized modules, we had several proposals for
alternative designs a lot of discussion on this.  See in particular, these
discussions at the open design meetings:

 - [September 15, 2021](https://docs.google.com/document/d/1fOSRdyZR2w75D87yU2Ma9h2-_lEPL4NxvhJGJd-s5pk/edit#heading=h.gdy95njn5105):
   discussion about using SSA values vs attributes for expressions, whether
   parameters should just be a "special kind of port" etc.
 - [September 22, 2021](https://docs.google.com/document/d/1fOSRdyZR2w75D87yU2Ma9h2-_lEPL4NxvhJGJd-s5pk/edit#heading=h.tcwfqa9fi7u2):
   discussions on expression canonicalization, parameterized type casting and
   other topics.

This section tries to condense some of those discussions into key points:

**Why do instances repeat default parameters from modules?**

As described above, the full set of module parameters are specified on an
instance, even if some have default values.  The reason for this is that we want
the IR to be simple and efficient to analyze by the compiler: keeping (and
verifying that) instance parameters are in canonical form means that we can
index them with integers instead of names (just like module input and result
ports), and intermodule analysis/optimization doesn't have to handle default
values as a special case.  Instead they are just a matter for frontends and
the Verilog exporter to care about.

**Why model parameters with attributes instead of SSA values?**

It seems unfortunate to replicate some parts of the `comb` dialect (e.g.
`comb.add`) as attributes rather than just reusing the existing attributes.
Such a design has historical familiarity (e.g. LLVM's `ConstantExpr` class)
which led to a bunch of complexity in LLVM that would have been better avoided
(and yes - there are much better designs for LLVM's purposes than what it has
now).

All that said, using attributes is the right thing for a number of reasons:

1) This arithmetic happens at metaprogramming time, these ops do not turn into
   hardware.  It use important and useful to be able to know that structurally.
2) We need to verify parameter expressions are valid for the module they are
   defined in - it isn't generally ok for the verifier of the `hw.instance` op
   to walk an arbitrary amount of IR to check that an SSA value is valid as a
   parameter.
3) We need to support parameterized types like `!hw.int<n>`: because MLIR types
   are immortal and uniqued, they can refer to attributes but cannot refer to
   [SSA values](https://en.wikipedia.org/wiki/Static_single_assignment_form)
   (which may be destroyed).
4) Operations need to be able to compute their own type without creating other
   operations.  For example, we need to compute that the result type of
   `comb.concat %a, %b : (i1, !hw.int<n>)` is `!hw.int<n+1>` without introducing
   a new `comb.add` node to "add one to n".
5) In practice, comb ops and the canonicalizations that apply to them have very
   different goals than the canonicalizations we apply to parameter expressions.

## Type declarations

**Type declaration IR**

*Type Scope Operation*

A [Type Scope](Dialects/HW/#hwtype_scope-circthwtypescopeop) declares a single
region with a single block that contains type declarations. Type scopes provide
a means to group type declarations. Type scopes have a symbol and are themselves
symbol tables, so they may be looked up, and type declarations may be looked up
within them.

*Type Declaration Operation*

A [Type Declaration](https://circt.llvm.org/docs/Dialects/HW/#hwtypedecl-circthwtypedeclop)
declares a symbolic name for a type. It consists of:

* A symbolic name, which can be referred to in the IR by a Type Alias Type.
* A type attribute, which contains the underlying type the name refers to
* An optional string attribute, which specifies a name to give the type in the
  output. If not specified, the symbolic name is used in the output.

*Type Alias Type*

A [Type Alias](https://circt.llvm.org/docs/Dialects/HW/#an-symbolic-reference-to-a-type-declaration)
refers to a type declaration symbolically. It consists of:

* A symbolic reference to a Type Scope and Type Declaration within the scope
* A copy of the underlying type, cached in type storage
* A copy of the canonical type, cached in type storage

**Type declaration System Verilog output**

In ExportVerilog, Type Scopes may be split into their own output file, included
at the top of multiple split output files, or included via a header file.

Each Type Declaration will be emitted according to the System Verilog spec,
section 6.18, User-defined types. For example:

```
typedef logic mytype;
```

**Type canonicalization**

We take an approach to type canonicalization similar to [Clang](https://clang.llvm.org/docs/InternalsManual.html#canonical-types).

To implement this, CIRCT has an extra layer of indirection in the ODS
declarations of types that may be targeted by Type Declarations. This layer
makes it possible to generically let operations declare their operands and
results to be of a specific type, and also accept type declarations that
canonicalize to that type.

The HW dialect types are declared in [HWTypesImpl.td](https://github.com/llvm/circt/blob/main/include/circt/Dialect/HW/HWTypesImpl.td),
which declares the actual type and is used to generate the C++ wrappers. To make
these types usable in ODS operations in a generic way that support type
declarations, wrapper types are declared in [HWTypes.td](https://github.com/llvm/circt/blob/main/include/circt/Dialect/HW/HWTypes.td),
which uses helpers to define dialect types that may be either the declared type,
or a type declaration whose canonical type is the declared type.

In order to "see through" type declarations and get at the canonical type, a set
of helper functions are added to the hardware dialect: `hw::type_isa`,
`hw::type_cast`, and `hw::type_dyn_cast`. These should generally be used instead
of the MLIR `Type::isa`, `Type::cast`, and `Type::dyn_cast`, because they
transparently support type declarations.

As the Clang docs state: "The only hard part here is remembering not to use the
isa/cast/dyn_cast operations."

**Open Issues**

*Duplicated type in Type Alias Type IR*

In order to support Type canonicalization, the Type Alias Type keeps a copy of
the underlying type, as well as what it canonicalizes to, cached in Type
storage. This allows the helpers like `hw::type_isa` to efficiently query the
underlying type or canonical type in constant time.

Ideally, the Type Alias Type would only contain a symbolic reference to the
declaration, and at construction time (during parsing or when built
programmatically) look up and cache the underlying type and its canonical type.

Unfortunately, this is not currently possible with the MLIR parser API. There is
some discussion about this on [Discourse](https://discourse.llvm.org/t/contextual-type-verification/4525),
and it seems like a good enhancement. This is tracked in issue [#1642](https://github.com/llvm/circt/issues/1642).
Until then, we must duplicate the type in the IR.

## Symbols and Visibility

Verilog has a broad notion of what can be named outside the context of its
declaration.  This is compounded by the many tools which have additional source
files which refer to Verilog names (e.g. tcl files).  However, we do not want to
require that every wire, register, instance, localparam, port, etc which can be
named not be touched by passes.  We want only entities marked as public facing
to impede transformation.

For this reason, wires, registers, and instances may optionally define a symbol.
When the symbol is defined, the entity is considered part of the visible
interface and should be preserved in transformation.  Entities without a symbol
defined are considered private and may be changed by transformation.

**Implementation constraints**

Currently, MLIR restricts symbol resolution to looking in and downward through
any nested symbol tables when resolving symbols.  This assumption has
implications for verification, the pass manager, and threading.  Until symbol
references are more general, SV and HW dialects do not define symbol tables for
modules.  Therefore, wires, registers, and interfaces exist in the same
namespace as modules.  It is encouraged that one prefaces the names to avoid
conflict with modules.  The symbol names on these entities has no bearing on the
output Verilog, each of these entities has a defined way to assign its name (SSA
value name for wires and regs, a non-optional string for instances).

As MLIR symbol support improves, it is desired to move to per-module symbol
tables and to unify names with symbol names.

**Ports**

Module ports are remotely nameable entities in Verilog, but are not easily named
with symbols.  A suggested workaround is to attach a wire to a port and use its
symbol for remote references.  Instance ports have a similar problem.

## Future Directions

There are many possible future directions that we anticipate tackling, when and
if the need arises:

**More support for IR**

Many in the CIRCT community are interested in adding first-class support for
parametric modules -- similar but more general than SystemVerilog module
parameters.  It isn't clear yet whether this should be part of the HW dialect
or something higher level.

Separate from a "good" representation of parametric modules, the SV dialect
could grow direct support for representing the SystemVerilog functionality
in this space, including even things like "generate" blocks.

**EDA Tool-specific Subdialects**

The EDA tool ecosystem is filled with a wide range of tools with different
capabilities -- for example [see this
table](https://symbiflow.github.io/sv-tests-results/) for one compilation of
different systems and their capabilities.  As such, we expect that the day will
come where a frontend wants to generate fancy features for some modern systems,
but cannot afford to break compatibility with other ecosystem tools.

Given the design of the HW/SV dialects, there is no need to resort to "lowest
common denominator" approach here: we can allow frontends to generate "fancy"
features, then use progressive lowering when dealing with tools that can't
handle them.  This can also allow IP providers to decide what flavor
of features they want to provide to their customers (or provide multiple
different choices).

**SystemVerilog Parser**

As the SV dialect grows out, it becomes natural to think about building a high
quality parser that reads SystemVerilog source code and parses it into the SV
dialect.  Such functionality could be very useful to help build tooling for the
SystemVerilog ecosystem.

Such a parser should follow clang-style principles of producing high quality
diagnostics, preserving source location information, being built as a library,
etc.
