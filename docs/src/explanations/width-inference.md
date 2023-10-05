---
layout: docs
title:  "Width Inference"
section: "chisel3"
---

# Width Inference

Chisel provides bit width inference to reduce design effort. Users are encouraged to manually specify widths of ports and registers to prevent any surprises, but otherwise unspecified widths will be inferred by the FIRRTL compiler.

For all circuit components declared with unspecified widths, the FIRRTL compiler will infer the minimum possible width that maintains the legality of all its incoming connections. Implicit here is that inference is done in a right to left fashion in the sense of an assignment statement in chisel, i.e. from the left hand side from the right hand side. If a component has no incoming connections, and the width is unspecified, then an error is thrown to indicate that the width could not be inferred.

For module input ports with unspecified widths, the inferred width is the minimum possible width that maintains the legality of all incoming connections to all instantiations of the module.
The width of a ground-typed multiplexor expression is the maximum of its two corresponding input widths. For multiplexing aggregate-typed expressions, the resulting widths of each leaf subelement is the maximum of its corresponding two input leaf subelement widths.
The width of a conditionally valid expression is the width of its input expression.  For the full formal description see the [Firrtl Spec](https://github.com/freechipsproject/firrtl/blob/master/spec/spec.pdf).

Hardware operators have output widths as defined by the following set of rules:

| operation        | bit width           |
| ---------        | ---------           |
| `z = x + y` *or* `z = x +% y`        | `w(z) = max(w(x), w(y))`   |
| `z = x +& y`       | `w(z) = max(w(x), w(y)) + 1`   |
| `z = x - y` *or* `z = x -% y`       | `w(z) = max(w(x), w(y))`    |
| `z = x -& y`       | `w(z) = max(w(x), w(y)) + 1`   |
| `z = x & y`        | `w(z) = max(w(x), w(y))`    |
| `z = Mux(c, x, y)` | `w(z) = max(w(x), w(y))`    |
| `z = w * y`        | `w(z) = w(x) + w(y)`        |
| `z = x << n`       | `w(z) = w(x) + maxNum(n)` |
| `z = x >> n`       | `w(z) = w(x) - minNum(n)` |
| `z = Cat(x, y)`    | `w(z) = w(x) + w(y)`      |
| `z = Fill(n, x)`   | `w(z) = w(x) * maxNum(n)` |

>where for instance `w(z)` is the bit width of wire `z`, and the `&`
rule applies to all bitwise logical operations.

Given a path of connections that begins with an unspecified width element (most commonly a top-level input), then the compiler will throw an exception indicating a certain width was uninferrable.

A common "gotcha" comes from truncating addition and subtraction with the operators `+` and `-`. Users who want the result to maintain the full, expanded precision of the addition or subtraction should use the expanding operators `+&` and `-&`.

> The default truncating operation comes from Chisel's history as a microprocessor design language.
