---
layout: docs
title:  "Operators"
section: "chisel3"
---

# Chisel Operators

Chisel defines a set of hardware operators:

| Operation        | Explanation |
| ---------        | ---------           |
| **Bitwise operators**                       | **Valid on:** SInt, UInt, Bool    |
| `val invertedX = ~x`                        | Bitwise NOT |
| `val hiBits = x & "h_ffff_0000".U`          | Bitwise AND                     |
| `val flagsOut = flagsIn \| overflow`         | Bitwise OR                      |
| `val flagsOut = flagsIn ^ toggle`           | Bitwise XOR                     |
| **Bitwise reductions.**                     | **Valid on:** SInt and UInt. Returns Bool. |
| `val allSet = x.andR`                       | AND reduction                     |
| `val anySet = x.orR`                        | OR reduction                      |
| `val parity = x.xorR`                       | XOR reduction                     |
| **Equality comparison.**                    | **Valid on:** SInt, UInt, and Bool. Returns Bool. |
| `val equ = x === y`                         | Equality                          |
| `val neq = x =/= y`                         | Inequality                        |
| **Shifts**                                  | **Valid on:** SInt and UInt       |
| `val twoToTheX = 1.S << x`                  | Logical shift left                |
| `val hiBits = x >> 16.U`                    | Right shift (logical on UInt and arithmetic on SInt). |
| **Bitfield manipulation**                   | **Valid on:** SInt, UInt, and Bool. |
| `val xLSB = x(0)`                           | Extract single bit, LSB has index 0.     |
| `val xTopNibble = x(15, 12)`                | Extract bit field from end to start bit position.     |
| `val usDebt = Fill(3, "hA".U)`              | Replicate a bit string multiple times.     |
| `val float = Cat(sign, exponent, mantissa)` | Concatenates bit fields, with first argument on left.     |
| **Logical Operations**                      | **Valid on:** Bool
| `val sleep = !busy`                         | Logical NOT                       |
| `val hit = tagMatch && valid`               | Logical AND                       |
| `val stall = src1busy || src2busy`          | Logical OR                        |
| `val out = Mux(sel, inTrue, inFalse)`       | Two-input mux where sel is a Bool |
| **Arithmetic operations**                   | **Valid on Nums:** SInt and UInt.  |
| `val sum = a + b` *or* `val sum = a +% b`   | Addition (without width expansion) |
| `val sum = a +& b`                          | Addition (with width expansion)    |
| `val diff = a - b` *or* `val diff = a -% b` | Subtraction (without width expansion) |
| `val diff = a -& b`                         | Subtraction (with width expansion) |
| `val prod = a * b`                          | Multiplication                     |
| `val div = a / b`                           | Division                           |
| `val mod = a % b`                           | Modulus                            |
| **Arithmetic comparisons**                  | **Valid on Nums:** SInt and UInt. Returns Bool. |
| `val gt = a > b`                            | Greater than                       |
| `val gte = a >= b`                          | Greater than or equal              |
| `val lt = a < b`                            | Less than                          |
| `val lte = a <= b`                          | Less than or equal                 |

>Our choice of operator names was constrained by the Scala language.
We have to use triple equals```===``` for equality and ```=/=```
for inequality to allow the
native Scala equals operator to remain usable.

The Chisel operator precedence is not directly defined as part of the Chisel language.
Practically, it is determined by the evaluation order of the circuit,
which naturally follows the [Scala operator precedence](https://docs.scala-lang.org/tour/operators.html).
If in doubt of operator precedence, use parentheses.

> The Chisel/Scala operator precedence is similar but
not identical to precedence in Java or C. Verilog has the same operator precedence as C, but VHDL
does not. Verilog has precedence ordering for logic operations, but in VHDL
those operators have the same precedence and are evaluated from left to right.
