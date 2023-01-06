---
layout: docs
title:  "Frequently Asked Questions"
section: "chisel3"
---

# Frequently Asked Questions

* [Where should I start if I want to learn Chisel?](#where-should-i-start-if-i-want-to-learn-chisel)
* [How do I ... in Chisel?](#how-do-i-do--eg-like-that-in-verilog-in-chisel)
* [What versions of the various projects work together?](#what-versions)
* [How can I contribute to Chisel?](#how-can-i-contribute-to-chisel)
* [Why DecoupledIO instead of ReadyValidIO?](#why-decoupledio-instead-of-readyvalidio)
* [Why do I have to wrap module instantiations in `Module(...)`?](#why-do-i-have-to-wrap-module-instantiations-in-module)
* [Why Chisel?](#why-chisel)
* [Does Chisel support X and Z logic values?](#does-chisel-support-x-and-z-logic-values)
* [I just want some Verilog; what do I do?](#get-me-verilog)
* [I just want some FIRRTL; what do I do?](#get-me-firrtl)
* [Why doesn't Chisel tell me which wires aren't connected?](#why-doesnt-chisel-tell-me-which-wires-arent-connected)
* [What does `Reference ... is not fully initialized.` mean?](#what-does-reference--is-not-fully-initialized-mean)
* [Can I specify behavior before and after generated initial blocks?](#can-i-specify-behavior-before-and-after-generated-initial-blocks)

### Where should I start if I want to learn Chisel?

We recommend the [Chisel Bootcamp](https://github.com/freechipsproject/chisel-bootcamp) for getting started with Chisel.

### How do I do ... (e.g. like that in Verilog) in Chisel?

See the [cookbooks](../cookbooks/cookbook).

### What versions of the various projects work together? <a name="what-versions"></a>

See [Chisel Project Versioning](../appendix/versioning).

### How can I contribute to Chisel?

Check out the [Contributor Documentation](https://github.com/chipsalliance/chisel3#contributor-documentation) in the chisel3 repository.

### Why DecoupledIO instead of ReadyValidIO?

There are multiple kinds of Ready/Valid interfaces that impose varying restrictions on the producers and consumers. Chisel currently provides the following:

* [DecoupledIO](https://chisel.eecs.berkeley.edu/api/index.html#chisel3.util.DecoupledIO) - No guarantees
* [IrrevocableIO](https://chisel.eecs.berkeley.edu/api/index.html#chisel3.util.IrrevocableIO) - Producer promises to not change the value of 'bits' after a cycle where 'valid' is high and 'ready' is low. Additionally, once 'valid' is raised it will never be lowered until after 'ready' has also been raised.

### Why do I have to wrap module instantiations in `Module(...)`?

In short: Limitations of Scala

Chisel Modules are written by defining a [Scala class](http://docs.scala-lang.org/tutorials/tour/classes.html) and implementing its constructor. As elaboration runs, Chisel constructs a hardware AST from these Modules. The compiler needs hooks to run before and after the actual construction of the Module object. In Scala, superclasses are fully initialized before subclasses, so by extending Module, Chisel has the ability to run some initialization code before the user's Module is constructed. However, there is no such hook to run after the Module object is initialized. By wrapping Module instantiations in the Module object's apply method (ie. `Module(...)`), Chisel is able to perform post-initialization actions. There is a [proposed solution](https://issues.scala-lang.org/browse/SI-4330), so eventually this requirement will be lifted, but for now, wrap those Modules!

### Why Chisel?

Please see [Chisel Motivation](../explanations/motivation)

### Does Chisel support X and Z logic values

Chisel does not directly support Verilog logic values ```x``` *unknown* and ```z``` *high-impedance*.  There are a number of reasons to want to avoid these values.  See:[The Dangers of Living With An X](http://infocenter.arm.com/help/topic/com.arm.doc.arp0009a/Verilog_X_Bugs.pdf) and [Malicious LUT: A stealthy FPGA Trojan injected and triggered by the design flow](http://ieeexplore.ieee.org/document/7827620/).  Chisel has its own eco-system of unit and functional testers that limit the need for ```x``` and ```z``` and their omission simplify language implementation, design, and testing.  The circuits created by chisel do not preclude developers from using ```x``` and ```z``` in downstream toolchains as they see fit.

### Get me Verilog
I wrote a module and I want to see the Verilog; what do I do?

Here's a simple hello world module in a file HelloWorld.scala.


```scala
package intro
```
```scala mdoc:silent
import chisel3._
class HelloWorld extends Module {
  val io = IO(new Bundle{})
  printf("hello world\n")
}
```

Add the following
```scala mdoc:silent
import circt.stage.ChiselStage
object VerilogMain extends App {
  ChiselStage.emitSystemVerilog(new HelloWorld)
}
```
Now you can get some Verilog. Start sbt:
```
bash> sbt
> run-main intro.VerilogMain
[info] Running intro.VerilogMain
[info] [0.004] Elaborating design...
[info] [0.100] Done elaborating.
[success] Total time: 1 s, completed Jan 12, 2017 6:24:03 PM
```
or as a one-liner:
```
bash> sbt 'runMain intro.VerilogMain'
```
After either of the above there will be a HelloWorld.v file in the current directory:
```scala mdoc:invisible
val verilog = ChiselStage.emitSystemVerilog(new HelloWorld)
```
```scala mdoc:passthrough
println("```verilog")
println(verilog)
println("```")
```

You can see additional options with
```
bash> sbt 'runMain intro.HelloWorld --help'
```
This will return a comprehensive usage line with available options.

For example to place the output in a directory name buildstuff use
```
bash> sbt 'runMain intro.HelloWorld --target-dir buildstuff --top-name HelloWorld'
```

Alternatively, you can also use the sbt console to invoke the Verilog driver:

```
$ sbt
> console
[info] Starting scala interpreter...
Welcome to Scala 2.12.13 (OpenJDK 64-Bit Server VM, Java 1.8.0_275).
Type in expressions for evaluation. Or try :help.

scala> (new circt.stage.ChiselStage).emitSystemVerilog(new HelloWorld())
chisel3.Driver.execute(Array[String](), () => new HelloWorld)
Elaborating design...
Done elaborating.
res1: String =
"module HelloWorld(
  input   clock,
  input   reset
);
...
```

As before, there should be a HelloWorld.v file in the current directory.

Note: Using the following, without the `new`,
will ONLY return the string representation, and will not emit a `.v` file:

```scala mdoc:silent
ChiselStage.emitSystemVerilog(new HelloWorld())
```

### Get me FIRRTL

If for some reason you don't want the Verilog (e.g. maybe you want to run some custom transformations before exporting to Verilog), then use something along these lines:

```scala
package intro
```
```scala mdoc:silent:reset

import chisel3._
import circt.stage.ChiselStage

class MyFirrtlModule extends Module {
  val io = IO(new Bundle{})
}

object FirrtlMain extends App {
  ChiselStage.emitCHIRRTL(new MyFirrtlModule)
}
```

Run it with:

```
sbt 'runMain intro.FirrtlMain'
```
```scala mdoc:invisible
val theFirrtl = ChiselStage.emitCHIRRTL(new MyFirrtlModule)
```
```scala mdoc:passthrough
println("```")
println(theFirrtl)
println("```")
```

Alternatively, you can also use the sbt console to invoke the FIRRTL driver directly (replace MyFirrtlModule with your module name):

```
$ sbt
> console
[info] Starting scala interpreter...
Welcome to Scala 2.12.13 (OpenJDK 64-Bit Server VM, Java 1.8.0_275).
Type in expressions for evaluation. Or try :help.

scala> circt.stage.ChiselStage.emitCHIRRTL(new MyFirrtlModule)
Elaborating design...
Done elaborating.
res3: String = ...
```

### Why doesn't Chisel tell me which wires aren't connected?

As long as your code uses `import chisel3._` (and not `import Chisel._`), it does!
See [Unconnected Wires](../explanations/unconnected-wires) for details.

### What does `Reference ... is not fully initialized.` mean?

It means that you have unconnected wires in your design which could be an indication of a design bug.

In Chisel2 compatibility mode (`NotStrict` compile options), chisel generates firrtl code that disables firrtl's initialized wire checks.
In pure chisel3 (`Strict` compile options), the generated firrtl code does not contain these disablers (`is invalid`).
Output wires that are not driven (not connected) are reported by firrtl as `not fully initialized`.
Read more at [Unconnected Wires](../explanations/unconnected-wires) for details on solving the problem.

### Can I specify behavior before and after generated initial blocks?
Users may define the following macros if they wish to specify behavior before or after emitted initial blocks.

* `BEFORE_INITIAL`, which is called before the emitted (non-empty) initial block if it is defined
* `AFTER_INITIAL`, which is called after the emitted (non-empty) initial block if it is defined

These macros may be useful for turning coverage on and off.
