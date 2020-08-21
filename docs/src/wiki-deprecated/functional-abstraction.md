---
layout: docs
title:  "Functional Abstraction"
section: "chisel3"
---
We can define functions to factor out a repeated piece of logic that
we later reuse multiple times in a design.  For example, we can wrap
up our earlier example of a simple combinational logic block as
follows:
```scala
def clb(a: UInt, b: UInt, c: UInt, d: UInt): UInt =
  (a & b) | (~c & d)
```

where ```clb``` is the function which takes ```a```, ```b```,
```c```, ```d``` as arguments and returns a wire to the output of a
boolean circuit.  The ```def``` keyword is part of Scala and
introduces a function definition, with each argument followed by a colon then its type,
and the function return type given after the colon following the
argument list.  The equals (```=})```sign separates the function argument list from the function
definition.

We can then use the block in another circuit as follows:
```scala
val out = clb(a,b,c,d)
```

We will later describe many powerful ways to use functions to
construct hardware using Scala's functional programming support.
