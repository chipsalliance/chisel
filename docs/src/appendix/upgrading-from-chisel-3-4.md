---
layout: docs
title:  "Upgrading From Chisel 3.4 to 3.5"
section: "chisel3"
---

<!-- Prelude -->
```scala mdoc:invisible
import chisel3._
```
<!-- End Prelude -->

## Upgrading From Chisel 3.4 to 3.5

Chisel 3.5 was a major step forward. It added support for Scala 2.13 as well as dropped many long deprecated APIs.
Some users may run into issues while upgrading so this page serves as a central location to describe solutions to common issues.


### General Strategy for Upgrade

Users are encouraged to first upgrade to the latest version of Chisel 3.4 (3.4.4 at the time of writing) and resolve all deprecation warnings. Doing so should enable a smoother transition to Chisel 3.5.

### Common Issues

#### Value io is not a member of chisel3.Module

This issue most often arises when there are two implementations of a given `Module` that may be chosen between by a generator parameter.
For example:

```scala mdoc
class Foo extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  io.out := io.in
}

class Bar extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  io.out := io.in + 1.U
}
```

```scala mdoc:fail
class Example(useBar: Boolean) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })

  val inst = if (useBar) {
    Module(new Bar)
  } else {
    Module(new Foo)
  }

  inst.io.in := io.in
  io.out := inst.io.out
}
```

`Foo` and `Bar` clearly have the same interface, yet we get a type error in Chisel 3.5.
Notably, while this does work in Chisel 3.4, it does throw a deprecation warning.
In short, this code is relying on old behavior of the Scala type inferencer.
In Scala 2.11 and before, the type inferred for `val inst` is: `Module { def io : { def in : UInt; def out : UInt } }`.
And in fact, if we manually ascribe this type to `val inst`, our same code from above works in Chisel 3.5:

```scala mdoc
class Example(useBar: Boolean) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })

  val inst: Module { def io : { def in : UInt; def out : UInt } } = if (useBar) {
    Module(new Bar)
  } else {
    Module(new Foo)
  }

  inst.io.in := io.in
  io.out := inst.io.out
}
```

So what is going on and why is this type so ugly?
This is called a [_structural_ (or _duck_) type](https://en.wikipedia.org/wiki/Structural_type_system).
Basically, code does not provide any unifying type for `Foo` and `Bar` so the compiler does its best to make one up.
One negative consequence of the old Scala behavior is that structural type inference makes it very easy to accidentally
change the public API of your code without meaning to.
Thus, in the bump from Scala 2.11 to 2.12, the behavior of the Scala compiler changed to not do structural type inference by default.

The solution, is to explicitly provide a type to the Scala compiler:

```scala mdoc:invisible:reset
import chisel3._
```

```scala mdoc
trait HasCommonInterface extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
}

class Foo extends Module with HasCommonInterface {
  io.out := io.in
}

class Bar extends Module with HasCommonInterface {
  io.out := io.in + 1.U
}
```

Now our original code works:

```scala mdoc
class Example(useBar: Boolean) extends Module {
  val io = IO(new Bundle {
    val in = IO(Input(UInt(8.W)))
    val out = IO(Output(UInt(8.W)))
  })

  // Now, inst is inferred to be of type "HasCommonInterface"
  val inst = if (useBar) {
    Module(new Bar)
  } else {
    Module(new Foo)
  }

  inst.io.in := io.in
  io.out := inst.io.out
}
```

**Historical Note**

This may sound similar because a very similar error is included in [Common Issues](upgrading-from-scala-2-11#common-issues) in the Appendix for upgrading from Scala 2.11 to 2.12.
The workaround employed in Chisel for Scala 2.12 did not work in Scala 2.13, so we came up with the more robust solution described above.

