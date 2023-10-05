---
layout: docs
title:  "Muxes and Input Selection"
section: "chisel3"
---

# Muxes and Input Selection

Selecting inputs is very useful in hardware description, and therefore Chisel provides several built-in generic input-selection implementations.

### Mux
The first one is `Mux`. This is a 2-input selector. Unlike the `Mux2` example which was presented previously, the built-in `Mux` allows
the inputs (`in0` and `in1`) to be any datatype as long as they are the same subclass of `Data`.

By using the functional module creation feature presented in the previous section, we can create multi-input selector in a simple way:

```scala
Mux(c1, a, Mux(c2, b, Mux(..., default)))
```

### MuxCase

The nested `Mux` is not necessary since Chisel also provides the built-in `MuxCase`, which implements that exact feature.
`MuxCase` is an n-way `Mux`, which can be used as follows:

```scala
MuxCase(default, Array(c1 -> a, c2 -> b, ...))
```

Where each selection dependency is represented as a tuple in a Scala
array [ condition -> selected_input_port ].

### MuxLookup
Chisel also provides `MuxLookup` which is an n-way indexed multiplexer:

```scala
MuxLookup(idx, default)(Seq(0.U -> a, 1.U -> b, ...))
```

This is the same as a `MuxCase`, where the conditions are all index based selection:

```scala
MuxCase(default,
        Array((idx === 0.U) -> a,
              (idx === 1.U) -> b, ...))
```

Note that the conditions/cases/selectors (eg. c1, c2) must be in parentheses.

### Mux1H
Another ```Mux``` utility is the one-hot mux, ```Mux1H```. It takes a sequence of selectors and values and returns the value associated with the one selector that is set. If zero or multiple selectors are set the behavior is undefined.  For example:

```scala
  val hotValue = chisel3.util.Mux1H(Seq(
    io.selector(0) -> 2.U,
    io.selector(1) -> 4.U,
    io.selector(2) -> 8.U,
    io.selector(4) -> 11.U,
  ))
```
```Mux1H``` whenever possible generates *Firrtl* that is readily optimizable as low depth and/or tree.  This optimization is not possible when the values are of type ```FixedPoint``` or an aggregate type that contains ```FixedPoint```s and results instead as a simple ```Mux``` tree.  This behavior could be sub-optimal.  As ```FixedPoint``` is still *experimental* this behavior may change in the future.
