---
layout: docs
title:  "Frequently Asked Questions"
section: "chisel3"
---

* [Where should I start if I want to learn Chisel?](#where-should-i-start-if-i-want-to-learn-chisel)
* [How do I ... in Chisel?](#how-do-i-do--eg-like-that-in-verilog-in-chisel)
* [How can I contribute to Chisel?](#how-can-i-contribute-to-chisel)
* [What is the difference between release and master branches?](#what-is-the-difference-between-release-and-master-branches)
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

See the [cookbook](cookbook).

### How can I contribute to Chisel?

A good to place to start is to fill out the [How Can I Contribute Form](https://docs.google.com/forms/d/e/1FAIpQLSfwTTY8GkfSZ2sU2T2mNpfNMpIM70GlXOrjqiHoC9ZBvwn_CA/viewform).

### What is the difference between release and master branches?

We have two main branches for each main Chisel project:

- `master`
- `release`

`master` is the main development branch and it is updated frequently (often several times a day).
Although we endeavour to keep the `master` branches in sync, they may drift out of sync for a day or two.
We do not publish the `master` branches.
If you wish to use them, you need to clone the GitHub repositories and use `sbt publishLocal` to make them available on your local machine.

The `release` branches are updated less often (currently bi-weekly) and we try to guarantee they are in sync.
We publish these to Sonatype/Maven on a bi-weekly basis.

In general, you can not mix `release` and `master` branches and assume they will work.

The default branches for the user-facing repositories (chisel-template and chisel-tutorial) are the `release` branches - these should always *just work* for new users as they use the `release` branches of chisel projects.

If you want to use something more current than the `release` branch, you should `git checkout master` for all the chisel repos you intend to use, then `sbt publishLocal` them in this order:

- firrtl
- firrtl-interpreter
- chisel3
- chisel-testers

Then, if you're working with the user-facing repositories:

- chisel-tutorial
- chisel-template

Since this is a substantial amount of work (with no guarantee of success), unless you are actively involved in Chisel development, we encourage you to stick with the `release` branches and their respective dependencies.

### Why DecoupledIO instead of ReadyValidIO?

There are multiple kinds of Ready/Valid interfaces that impose varying restrictions on the producers and consumers. Chisel currently provides the following:

* [DecoupledIO](https://chisel.eecs.berkeley.edu/api/index.html#chisel3.util.DecoupledIO) - No guarantees
* [IrrevocableIO](https://chisel.eecs.berkeley.edu/api/index.html#chisel3.util.IrrevocableIO) - Producer promises to not change the value of 'bits' after a cycle where 'valid' is high and 'ready' is low. Additionally, once 'valid' is raised it will never be lowered until after 'ready' has also been raised.

### Why do I have to wrap module instantiations in `Module(...)`?

In short: Limitations of Scala

Chisel Modules are written by defining a [Scala class](http://docs.scala-lang.org/tutorials/tour/classes.html) and implementing its constructor. As elaboration runs, Chisel constructs a hardware AST from these Modules. The compiler needs hooks to run before and after the actual construction of the Module object. In Scala, superclasses are fully initialized before subclasses, so by extending Module, Chisel has the ability to run some initialization code before the user's Module is constructed. However, there is no such hook to run after the Module object is initialized. By wrapping Module instantiations in the Module object's apply method (ie. `Module(...)`), Chisel is able to perform post-initialization actions. There is a [proposed solution](https://issues.scala-lang.org/browse/SI-4330), so eventually this requirement will be lifted, but for now, wrap those Modules!

### Why Chisel?

Borrowed from [Chisel Introduction](introduction)

>We were motivated to develop a new hardware language by years of
struggle with existing hardware description languages in our research
projects and hardware design courses.  _Verilog_ and _VHDL_ were developed
as hardware _simulation_ languages, and only later did they become
a basis for hardware _synthesis_.  Much of the semantics of these
languages are not appropriate for hardware synthesis and, in fact,
many constructs are simply not synthesizable.  Other constructs are
non-intuitive in how they map to hardware implementations, or their
use can accidently lead to highly inefficient hardware structures.
While it is possible to use a subset of these languages and still get
acceptable results, they nonetheless present a cluttered and confusing
specification model, particularly in an instructional setting.

>However, our strongest motivation for developing a new hardware
language is our desire to change the way that electronic system design
takes place.  We believe that it is important to not only teach
students how to design circuits, but also to teach them how to design
*circuit generators* ---programs that automatically generate
designs from a high-level set of design parameters and constraints.
Through circuit generators, we hope to leverage the hard work of
design experts and raise the level of design abstraction for everyone.
To express flexible and scalable circuit construction, circuit
generators must employ sophisticated programming techniques to make
decisions concerning how to best customize their output circuits
according to high-level parameter values and constraints.  While
Verilog and VHDL include some primitive constructs for programmatic
circuit generation, they lack the powerful facilities present in
modern programming languages, such as object-oriented programming,
type inference, support for functional programming, and reflection.

>Instead of building a new hardware design language from scratch, we
chose to embed hardware construction primitives within an existing
language.  We picked Scala not only because it includes the
programming features we feel are important for building circuit
generators, but because it was specifically developed as a base for
domain-specific languages.

### Does Chisel support X and Z logic values

Chisel does not directly support Verilog logic values ```x``` *unknown* and ```z``` *high-impedance*.  There are a number of reasons to want to avoid these values.  See:[The Dangers of Living With An X](http://infocenter.arm.com/help/topic/com.arm.doc.arp0009a/Verilog_X_Bugs.pdf) and [Malicious LUT: A stealthy FPGA Trojan injected and triggered by the design flow](http://ieeexplore.ieee.org/document/7827620/).  Chisel has it's own eco-system of unit and functional testers that limit the need for ```x``` and ```z``` and their omission simplify language implementation, design, and testing.  The circuits created by chisel do not preclude developers from using ```x``` and ```z``` in downstream toolchains as they see fit.

### Get me Verilog
I wrote a module and I want to see the Verilog; what do I do?

Here's a simple hello world module in a file HelloWorld.scala.

```scala
package intro
import chisel3._
class HelloWorld extends Module {
  val io = IO(new Bundle{})
  printf("hello world\n")
}
```
Add the following
```scala
object HelloWorld extends App {
  chisel3.Driver.execute(args, () => new HelloWorld)
}
```
Now you can get some Verilog. Start sbt:
```
bash> sbt
> run-main intro.HelloWorld
[info] Running examples.HelloWorld
[info] [0.004] Elaborating design...
[info] [0.100] Done elaborating.
[success] Total time: 1 s, completed Jan 12, 2017 6:24:03 PM
```
or as a one-liner:
```
bash> sbt 'runMain intro.HelloWorld'
```
After either of the above there will be a HelloWorld.v file in the current directory.

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
[info]
Welcome to Scala 2.11.8 (OpenJDK 64-Bit Server VM, Java 1.8.0_121).
Type in expressions for evaluation. Or try :help.
scala> chisel3.Driver.execute(Array[String](), () => new HelloWorld)
chisel3.Driver.execute(Array[String](), () => new HelloWorld)
[info] [0.014] Elaborating design...
[info] [0.306] Done elaborating.
Total FIRRTL Compile Time: 838.8 ms
res3: chisel3.ChiselExecutionResult = [...]
```

As before, there should be a HelloWorld.v file in the current directory.

### Get me FIRRTL

If for some reason you don't want the Verilog (e.g. maybe you want to run some custom transformations before exporting to Verilog), then use something along these lines (replace Multiplier with your module):

```scala
package intro

import chisel3._
import java.io.File

object Main extends App {
  val f = new File("Multiplier.fir")
  chisel3.Driver.dumpFirrtl(chisel3.Driver.elaborate(() => new Multiplier), Option(f))
}
```

Run it with:

```
sbt 'runMain intro.Main'
```

Alternatively, you can also use the sbt console to invoke the FIRRTL driver directly (replace HelloWorld with your module name):

```
$ sbt
> console
[info] Starting scala interpreter...
[info]
Welcome to Scala 2.11.11 (OpenJDK 64-Bit Server VM, Java 1.8.0_151).
Type in expressions for evaluation. Or try :help.
scala> chisel3.Driver.dumpFirrtl(chisel3.Driver.elaborate(() => new HelloWorld), Option(new java.io.File("output.fir")))
chisel3.Driver.dumpFirrtl(chisel3.Driver.elaborate(() => new HelloWorld), Option(new java.io.File("output.fir")))
[info] [0.000] Elaborating design...
[info] [0.001] Done elaborating.
res3: java.io.File = output.fir
```

### Why doesn't Chisel tell me which wires aren't connected?

As of commit [c313e13](https://github.com/freechipsproject/chisel3/commit/c313e137d4e562ef20195312501840ceab8cbc6a) it can!
Please visit the wiki page [Unconnected Wires](unconnected-wires) for details.

### What does `Reference ... is not fully initialized.` mean?

It means that you have unconnected wires in your design which could be an indication of a design bug.

In Chisel2 compatibility mode (`NotStrict` compile options), chisel generates firrtl code that disables firrtl's initialized wire checks.
In pure chisel3 (`Strict` compile options), the generated firrtl code does not contain these disablers (`is invalid`).
Output wires that are not driven (not connected) are reported by firrtl as `not fully initialized`.
Please visit the wiki page [Unconnected Wires](unconnected-wires) for details on solving the problem.

### Can I specify behavior before and after generated initial blocks?
Users may define the following macros if they wish to specify behavior before or after emitted initial blocks.

* `BEFORE_INITIAL`, which is called before the emitted (non-empty) initial block if it is defined
* `AFTER_INITIAL`, which is called after the emitted (non-empty) initial block if it is defined

These macros may be useful for turning coverage on and off.
