# SV Dialect Rationale

This document describes various design points of the `sv` dialect, a common
dialect that is typically used in conjunction with the `hw` and `comb` dialects.
Please see the [HW Dialect Rationale](../HW/RationaleHW.md) for high level insight
on how these work together.  This follows in the spirit of
other [MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

- [SV Dialect Rationale](#sv-dialect-rationale)
  - [Introduction to the `sv` dialect](#introduction-to-the-sv-dialect)
  - [`sv` Type System](#sv-type-system)
  - [Overview of `sv` dialect operations](#overview-of-sv-dialect-operations)
  - [Statements](#statements)
  - [Declarations](#declarations)
  - [Expressions](#expressions)
    - [Verbatim op](#verbatim-op)
  - [Cost Model](#cost-model)
  - [SV Dialect Attributes](#sv-dialect-attributes)

## Introduction to the `sv` dialect

The `sv` dialect is one of the dialects that can be mixed into the HW dialect,
providing access to a range of syntactic and behavioral constructs in
SystemVerilog.  The driving focus of this dialect is to provide simple and
predictable access to these features: it is not focused primarily on being easy
to analyze and transform.

The `sv` dialect is designed to build on top of the `hw` dialect and is often
used in conjunction with the `comb` or other dialects, so it does not have its
own operations for combinational logic, modules, or other common functionality.

## `sv` Type System

Like the HW dialect, the SV dialect is designed to tolerate unknown types where
possible, allowing other dialects to mix in with it.  In addition to these
external types, and the types used by the HW dialect, the SV dialect defines
types for SystemVerilog interfaces.

TODO: Describe interface types, modports, etc.

## Overview of `sv` dialect operations

Because the SV dialect aims to align with the textual nature of SystemVerilog,
many of the constructs in the SV dialect have an "AST" style of representation.
The major classes of operations you'll find are:

1) Statements like `sv.if`, `sv.ifdef`, `sv.always` and `sv.initial` that
   expose primary task-like operations and the behavioral model.
1) Procedural assignment operators, including the `sv.bpassign` and `sv.passign`
   operators that expose the blocking (`x = y`) and non-blocking (`x <= y`)
   procedural operators.
1) Directives like `sv.finish` and `sv.alias` and behavioral functions like
   `sv.fwrite`.
1) Access to verification constructs with `sv.assert`, `sv.assume`, and
   `sv.cover`.
1) Escape hatches that allow direct integration of textual expressions
   (`sv.verbatim.expr`) and full statements (`sv.verbatim`).

These operations are designed to directly model the syntax of the SystemVerilog
language and to be easily printable by the ExportVerilog pass.  While there are
still many things in SystemVerilog that we cannot currently express in the SV
dialect, this design makes it easy to incrementally build out new capabilities
over time.

## Statements

TODO.

## Declarations

TODO: Describe `sv.wire`, `sv.reg`, 

## Expressions

TODO: Describe `sv.read_inout` and `sv.array_index_inout`.
### Indexed Part Select
Unlike Bit-selects which extract a particular bit from integer types,
 part-select can extract several contiguous bits in a vector net, vector reg,
 integer variable, or time variables. 
 
 `SystemVerilog` supports two types of part-selects, a `constant part-select`
 and an `indexed part-select`. 
SV dialect has two ops named `sv.part_select` and `sv.part_select_inout`,
that is lowered to the `indexed part-select` operation.
 The `sv.part_select` is defined on `Integer` type input and 
 `sv.part_select_inout` is defined on `inout` type.

 Part select consists of 3 arguments, the input value,
 a `width` and a `base` and an optional boolean attribute `decrement`.
 The `width` shall be a compile-time constant expression. 
 The `base` can be a runtime integer expression. 
 
 The operation selects bits starting at the `base` and ascending 
 or descending the bit range. The number of bits selected is equal to the
 `width` expression. The bit addressing is always ascending starting from the
 `base`, unless the `decrement` attribute is specified.

Part-selects that address a range of bits that are completely out of the
 address bounds of the net, reg, integer, or time, or when the part-select
 is x or z, shall yield the value x when read, and shall have no effect on
 the data stored when written.

Part-selects that are partially out of range shall when read return x for
 the bits that are out of range, and when written shall only affect the bits
 that are in range.
 
 In this example, bits starting from `%c2` and of width `1` are addressed.
 Hence `%0` is of width `1`.
  ```
  %0 = sv.part_select_inout %combWire[%c2 : 1] : !hw.inout<i10>, i3, !hw.inout<i1>
  ```
  
  
### Verbatim op

The verbatim operation produces a typed value expressed by a string of
SystemVerilog.  This can be used to access macros and other values that are
only sensible as Verilog text. There are three kinds of verbatim operations:

 1. VerbatimOp(`sv.verbatim`, the statement form
 2. VerbatimExprOp(`sv.verbatim.expr`), the expression form.
 3. VerbatimExprSEOp(`sv.verbatim.expr.se`), the effectful expression form.

For the verbatim expression form, the text string is assumed to have the
highest precedence - include parentheses in the text if it isn't a single token.
`sv.verbatim.expr` is assumed to not have side effects (is `NoSideEffect` in
MLIR terminology), whereas `sv.verbatim.expr.se` may have side effects.

Verbatim allows operand substitutions with '{{0}}' syntax.
For macro substitution, optional operands and symbols can be added after the 
string. Verbatim ops may also include an array of symbol references.
The indexing begins at 0, and if the index is greater than the
number of operands, then it is used to index into the symbols array.
It is invalid to have macro indices greater than the total number 
of operands and symbols.
Example, 

```
sv.verbatim "MACRO({{0}}, {{1}} reg={{4}}, {{3}})" 
            (%add, %xor) : i8, i8
            {symRefs = [@reg1, @Module1, @instance1]}
```

## Cost Model

The SV dialect is primarily designed for human consumption, not machines.  As
such, transformations should aim to reduce redundancy, eliminate useless
constructs (e.g. eliminate empty ifdef and if blocks), etc.

## SV Dialect attributes

### `sv.namehint`

TODO.

### `sv.attribute` and `sv.attributes`
`sv.attribute` is used to encode Verilog _attribute_ that annotates metadata
to verilog constructs. See more detail in the `sv.attribute` definition.
We encode SV attributes into attr-dict with the key `sv.attributes`.
The item of `sv.attributes` must be an ArrayAttr whose elements are `sv.attribute`.
Currently, SV attributes don't block most optimizations; therefore, users
should not expect that sv attributes always appear in the output verilog.
However, in the future, we might have to re-consider blocking every optimization
for operations with SV attributes.

Example,

```mlir
%0 = sv.wire { sv.attributes = [#sv.attribute<"foo">,
                                   #sv.attribute<"bar"="baz">]}

==>
(* foo, bar = baz *)
wire GEN;
```
