---
layout: docs
title:  "Combinational Circuits"
section: "chisel3"
---

# Combinational Circuits

A circuit is represented as a graph of nodes in Chisel.  Each node is
a hardware operator that has zero or more inputs and that drives one
output.  A literal, introduced above, is a degenerate kind of node
that has no inputs and drives a constant value on its output.  One way
to create and wire together nodes is using textual expressions.  For
example, we can express a simple combinational logic circuit
using the following expression:

```scala
(a & b) | (~c & d)
```

The syntax should look familiar, with `&` and `|`
representing bitwise-AND and -OR respectively, and `~`
representing bitwise-NOT.  The names `a` through `d`
represent named wires of some (unspecified) width.

Any simple expression can be converted directly into a circuit tree,
with named wires at the leaves and operators forming the internal
nodes.  The final circuit output of the expression is taken from the
operator at the root of the tree, in this example, the bitwise-OR.

Simple expressions can build circuits in the shape of trees, but to
construct circuits in the shape of arbitrary directed acyclic graphs
(DAGs), we need to describe fan-out.  In Chisel, we do this by naming
a wire that holds a subexpression that we can then reference multiple
times in subsequent expressions.  We name a wire in Chisel by
declaring a variable.  For example, consider the select expression,
which is used twice in the following multiplexer description:
```scala
val sel = a | b
val out = (sel & in1) | (~sel & in0)
```

The keyword `val` is part of Scala, and is used to name variables
that have values that won't change.  It is used here to name the
Chisel wire, `sel`, holding the output of the first bitwise-OR
operator so that the output can be used multiple times in the second
expression.

### Wires

Chisel also supports wires as hardware nodes to which one can assign values or connect other nodes.

```scala
val myNode = Wire(UInt(8.W))
when (isReady) {
  myNode := 255.U
} .otherwise {
  myNode := 0.U
}
```

```scala
val myNode = Wire(UInt(8.W))
when (input > 128.U) {
  myNode := 255.U
} .elsewhen (input > 64.U) {
  myNode := 1.U
} .otherwise {
  myNode := 0.U
}
```

Note that the last connection to a Wire takes effect. For example, the following two Chisel circuits are equivalent:

```scala
val myNode = Wire(UInt(8.W))
myNode := 10.U
myNode := 0.U
```

```scala
val myNode = Wire(UInt(8.W))
myNode := 0.U
```
