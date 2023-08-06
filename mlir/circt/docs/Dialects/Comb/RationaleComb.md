# `comb` Dialect Rationale

This document describes various design points of the Comb dialect, a common
dialect that is typically used in conjunction with the `hw` and `sv` dialects.
Please see the [`hw` Dialect Rationale](../HW/RationaleHW.md) for high level insight
on how these work together.  This follows in the spirit of
other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

- [`comb` Dialect Rationale](#comb-dialect-rationale)
  - [Introduction to the `comb` Dialect](#introduction-to-the-comb-dialect)
  - [Type System for `comb` Dialect](#type-system-for-comb-dialect)
    - [Zero-bit integer width is not supported](#zero-bit-integer-width-is-not-supported)
  - [Comb Operations](#comb-operations)
    - [Fully associative operations are variadic](#fully-associative-operations-are-variadic)
    - [Operators carry signs instead of types](#operators-carry-signs-instead-of-types)
    - [No implicit extensions of operands](#no-implicit-extensions-of-operands)
    - [No "Complement", "Negate", "ZExt", "SExt", Operators](#no-complement-negate-zext-sext-operators)
    - [No multibit mux operations](#no-multibit-mux-operations)
  - [Endianness: operand ordering and internal representation](#endianness-operand-ordering-and-internal-representation)
  - [Bitcasts](#bitcasts)
  - [Cost Model](#cost-model)

## Introduction to the `comb` Dialect

The `comb` dialect provides a collection of operations that define a mid-level
compiler IR for combinational logic.   It is *not* designed to model
SystemVerilog or any other hardware design language directly.  Instead, it is
designed to be easy to analyze and transform, and be a flexible and extensible
substrate that may be extended with higher level dialects mixed into it.

## Type System for `comb` Dialect

TODO: Simple integer types, eventually parametrically wide integer type
`hw.int<width>`.  Supports type aliases.  See HW rationale for more info.

### Zero-bit integer width is not supported

Combinational operations like add and multiply work on values of signless
standard integer types, e.g. `i42`, but they do not allow zero bit inputs.  This
design point is motivated by a couple of reasons:

1) The semantics of some operations (e.g. `comb.sext`) do not have an obvious
   definition with a zero bit input.

1) Zero bit operations are useless for operations that are definable, and their
   presence makes the compiler more complicated.

On the second point, consider an example like `comb.mux` which could allow zero
bit inputs and therefore produce zero bit results.  Allowing that as a design
point would require us to special case this in our cost models, and we would
have that optimizes it away.

By rejecting zero bit operations, we choose to put the complexity into the
lowering passes that generate the HW dialect (e.g. LowerToHW from FIRRTL).

Note that this decision only affects the core operations in the `comb` dialect
itself - it is perfectly reasonable to define your operations and mix them into
other `comb` constructs. 

## Comb Operations

This section contains notes about design decisions relating to
operations in the `comb` dialect.

### Fully associative operations are variadic

TODO: describe why add/xor/or are variadic

### Operators carry signs instead of types

TODO: describe why we have divu/divs but not addu/adds, and not sint vs uint.

### Selectable truth-table

To keep the interpretation of comb operators local to the dialect, each
operation where it matters has an optional flag to indicate what semantics it needs
to preserve.  All operations are defined in the expected way for 2-state (binary) logic.  However, comb is used for operations which have extended truth table for non-2-state logic for various target languages.  To accommodate this, operations can opt into known extended truth tables so that any transformation will preserve semantics with respect to the extended truth table.

Initially, operations support 2-state or the union of 4-state (verilog) and 9-state (VHDL) behavior.  2-state is specified with the "bin" flag on operations.  In the future, explicit flags for "4state" and "9state" might be added.

This is done so as to not make the operations in comb type-dependent.
This is a tradeoff in that comb operations are either 2-state or the union of 
common backend language weirdness.  This could be refined in the future.

### No implicit extensions of operands

Verilog and many other HDL's allow operators like `+` to work with
mixed size operands, and some have complicated contextual rules about how wide
the result is (e.g. adding two 12 bit integers gives you a 13 bit result).

While this is convenient for source programmers, this makes the job of compiler
analysis and optimization extremely challenging: peephole optimizations and
dataflow transformations need to reason about these pervasively.  Because the
`comb` dialect is designed as a "mid-level" dialect focused on optimization,
it doesn't allow implicit extensions: for example, `comb.add` takes the same
width inputs and returns the same width result.

There is room in the future for other points in the design space: for example,
it might be useful to add an `sv.add` operation that allows mixed operands to
get better separation of concerns in the Verilog printer if we wanted really
fancy extension elision. So far, very simple techniques have been enough to get
reasonable output.

### No "Complement", "Negate", "ZExt", "SExt", Operators

We choose to omit several operators that you might expect, in order to make the
IR more regular, easy to transform, and have fewer canonical forms.

 * No `~x` complement or `-x` negation operator: instead use `comb.xor(x, -1)`.
   or `comb.sub(0, x)` respectively.  These avoid having to duplicate many folds
   between `xor` and `sub`.

 * No zero extension operator to add high zero bits.  This is strictly redundant
   with `concat(zero, value)`.
   
 * No sign extension operator to add high sign bits.  `sext(x)` is strictly
   redundant with `concat(replicate(extract(x, highbit)), x)`.

The absence of these operations doesn't affect the expressive ability of the IR,
and ExportVerilog will notice these and generate the compact Verilog syntax
e.g. a complement or negate when needed.

### No multibit mux operations

The comb dialect in CIRCT doesn't have a first-class multibit mux.  Instead we
prefer to use two array operations to represent this.  For example, consider
a 3-bit condition:

```
 hw.module @multibit_mux(%a: i32, %b: i32, %c: i32, %idx: i3) -> (%out: i32) {
   %x_i32 = sv.constantX : i32
   %tmpArray = hw.array_create %a, %b, %x_i32, %b, %c, %x_i32 : i32
   %result   = hw.array_get %tmpArray[%idx] : !hw.array<6xi32>
   hw.output %result: i32
 }
```

This gets lowered into (something like) this Verilog:

```
module multibit_mux(
  input  [31:0] a, b, c,
  input  [2:0]  idx,
  output [31:0] out);

  wire [5:0][31:0] _T = {{a}, {b}, {32'bx}, {b}, {c}, {32'bx}};
  assign out = _T[idx];
endmodule
```

In this example, the last X element could be dropped and generate
equivalent code.

We believe that synthesis tools handle the correctly and generate efficient
netlists.  For those that don't (e.g. Yosys), we have a `disallowPackedArrays`
LoweringOption that legalizes away multi-dimensional arrays as part of lowering.

While we could use the same approach for single-bit muxes, we choose to have a
single bit `comb.mux` operation for a few reasons:

 * This is extremely common in hardware, and using 2x the memory to represent
   the IR would be wasteful.
 * This are many peephole and other optimizations that apply to it.

We discussed these design points at length in an [August 11, 2021 design
meeting](https://docs.google.com/document/d/1fOSRdyZR2w75D87yU2Ma9h2-_lEPL4NxvhJGJd-s5pk/edit#heading=h.ygmlwiic5e1y), and
discussed the tradeoffs of adding support for a single-operation mux.  Such a
move has some advantages and disadvantages:

1) It is another operation that many transformations would need to be aware of,
   e.g. Verilog emission would have to handle it, and peephole optimizations
   would have to be aware of `array_get` and `comb.mux`.
2) We don't have any known analyses or optimizations that are difficult to
   implement with the current representation.

We agreed that we'd revisit in the future if there were a specific reason to
add it.  Until then we represent the `array_create`/`array_get` pattern for
frontends that want to generate this.

## Endianness: operand ordering and internal representation

Certain operations require ordering to be defined (i.e. `comb.concat`,
`hw.array_concat`, and `hw.array_create`). There are two places where this
is relevant: in the MLIR assembly and in the MLIR C++ model.

In MLIR assembly, operands are always listed MSB to LSB (big endian style):

```mlir
%msb = comb.constant 0xEF : i8
%mid = comb.constant 0x7 : i4
%lsb = comb.constant 0xA018 : i16
%result = comb.concat %msb, %mid, %lsb : i8, i4, i16
// %result is 0xEF7A018
```

**Note**: Integers are always written in left-to-right lexical order. Operand
ordering for `concat.concat` was chosen to be consistent with simply abutting
them in lexical order.

```mlir
%1 = comb.constant 0x1 : i4
%2 = comb.constant 0x2 : i4
%3 = comb.constant 0x3 : i4
%arr123 = hw.array_create %1, %2, %3 : i4
// %arr123[0] = 0x3
// %arr123[1] = 0x2
// %arr123[2] = 0x1

%arr456 = ... // {0x4, 0x5, 0x6}
%arr78  = ... // {0x7, 0x8}
%arr = comb.array_concat %arr123, %arr456, %arr78 : !hw.array<3 x i4>, !hw.array<3 x i4>, !hw.array<2 x i4>
// %arr[0] = 0x8
// %arr[1] = 0x7
// %arr[2] = 0x6
// %arr[3] = 0x5
// %arr[4] = 0x4
// %arr[5] = 0x3
// %arr[6] = 0x2
// %arr[7] = 0x1
```

**Note**: This ordering scheme is unintuitive for anyone expecting C
array-like ordering. In C, arrays are laid out with index 0 as the least
significant value and the first element (lexically) in the array literal. In
the CIRCT _model_ (assembly and C++ of the operation creating the array), it
is the opposite -- the most significant value is on the left (e.g. the first
operand is the most significant). The indexing semantics at runtime, however,
differ in that the element zero is the least significant (which is lexically
on the right).

In the CIRCT C++ model, lists of values are in lexical order. That is, index
zero of a list is the leftmost operand in assembly, which is the most
significant value.

```cpp
ConcatOp result = builder.create<ConcatOp>(..., {msb, lsb});
// Is equivalent to the above integer concatenation example.
ArrayConcatOp arr = builder.create<ArrayConcatOp>(..., {arr123, arr456});
// Is equivalent to the above array example.
```

**Array slicing and indexing** (`array_get`) operations both have indexes as
operands. These indexes are the _runtime_ index, **not** the index in the
operand list which created the array upon which the op is running.

## Bitcasts

The bitcast operation represents a bitwise reinterpretation (cast) of a value.
This always synthesizes away in hardware, though it may or may not be
syntactically represented in lowering or export language. Since bitcasting
requires information on the bitwise layout of the types on which it operates,
we discuss that here. All of the types are _packed_, meaning there is never
padding or alignment.

- **Integer bit vectors**: MLIR's `IntegerType` with `Signless` semantics are
used to represent bit vectors. They are never padded or aligned.
- **Arrays**: The HW dialect defines a custom `ArrayType`. The in-hardware
layout matches C -- the high index of array starts at the MSB. Array's 0th
element's LSB located at array LSB.
- **Structs**: The HW dialect defines a custom `StructType`. The in-hardware
layout matches C -- the first listed member's MSB corresponds to the struct's
MSB. The last member in the list shares its LSB with the struct.
- **Unions**: The HW dialect's `UnionType` could contain the data of any of the
member types so its layout is defined to be equivalent to the union of members
type bitcast layout. In cases where the member types have different bit widths,
all members start at the 0th bit and are padded up to the width of the widest
member. The value with which they are padded is undefined.

**Example figure**

```
15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0 
-------------------------------------------------
| MSB                                       LSB | 16 bit integer vector
-------------------------------------------------
                         | MSB              LSB | 8 bit integer vector
-------------------------------------------------
| MSB      [1]       LSB | MSB     [0]      LSB | 2 element array of 8 bit integer vectors
-------------------------------------------------

      13 12 11 10  9  8  7  6  5  4  3  2  1  0 
                            ---------------------
                            | MSB           LSB | 7 bit integer vector
      -------------------------------------------
      | MSB     [1]     LSB | MSB    [0]    LSB | 2 element array of 7 bit integer vectors
      -------------------------------------------
      | MSB a LSB | MSB b[1] LSB | MSB b[0] LSB | struct
      -------------------------------------------  a: 4 bit integral
                                                   b: 2 element array of 5 bit integer vectors
```

## Cost Model

As a very general mid-level IR, it is important to define the principles that
canonicalizations and other general purpose transformations should optimize for.
There are often many different ways to represent a piece of logic in the IR, and
things will work better together if we keep the compiler consistent.

First, unlike something like LLVM IR, keep in mind that the HW dialect is a
model of hardware -- each operation generally corresponds to an instance of
hardware, it is not an "instruction" that is executed by an imperative CPU.
As such, the primary concerns are area and latency (and size of generated
Verilog), not "number of operations executed".  As such, here are important
concerns that general purpose
transformations should consider, ordered from most important to least important.

**Simple transformations are always profitable**

Many simple transformations are always a good thing, this includes:

1) Constant folding.
2) Simple strength reduction (e.g. divide to shift).
3) Common subexpression elimination.

These generally reduce the size of the IR in memory, can reduce the area of a
synthesized design, and often unblock secondary transformations.

**Reducing widths of non-trivial operations is always profitable**

It is always a good idea to reduce the width of non-trivial operands like add,
multiply, shift, divide, `and`, `or` (etc) since it produces less hardware and
enables other simplifications.

That said, it is a bad idea to *duplicate* operations to reduce widths: for
example, it is better to have one large multiply with many users than to clone
it because one user only needs some of the output bits.

It is also beneficial to reduce widths, even if it adds truncations or
extensions in the IR (because they are "just wires"). However, there are limits:
any and-by-constant could be lowered to a concat of each bit principle,
e.g. it is legal to turn `and(x, 9)` into `concat(x[3], 00, x[0])`.  Doing so is
considered unprofitable though, because it bloats the IR (and generated
Verilog).

**Don't get overly tricky with divide and remainder**

Divide operations (particularly those with non-constant divisors) generate a lot
of hardware, and can have long latencies.  As such, it is a generally bad idea
to do anything to an individual instance of a divide that can increase its
latency (e.g. merging a narrow divide with a wider divide and using a subset of
the result bits).

**Constants and moving bits around is free**

The following are considered "free" for area and latency concerns:

1) `hw.constant`
2) concatenation (including zero/sign extension idioms) and truncation
3) `comb.and` and `comb.or` with a constant.
4) Other similar operations that do not synthesize into hardware. 

All things being equal it is good to reduce the number of instances of these (to
reduce IR size and increase canonical form) but it is ok to introduce more of
these to improve on other metrics above.

**Ordering Concat and Extract**

The`concat(extract(..))` form is preferred over the `extract(concat(..))` form,
because

- `extract` gets "closer" to underlying `add/sub/xor/op` operations, giving way
  optimizations like narrowing.
- the form gives a more accurate view of the values that are being depended on.
- redundant extract operations can be removed from the concat argument lists,
  e.g.:
  `cat(extract(a), b, c, extract(d))`

Both forms perform similarly on hardware, since they are simply bit-copies.

