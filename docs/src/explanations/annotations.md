---
layout: docs
title:  "Annotations"
section: "chisel3"
---

# Annotations

`Annotation`s are metadata containers associated with zero or more "things" in a FIRRTL circuit.
Commonly, `Annotation`s are used to communicate information from Chisel to a specific, custom FIRRTL `Transform`.
In this way `Annotation`s can be viewed as the "arguments" that a specific `Transform` consumes.

This article focuses on the approach to building a basic library that contains `Annotation`s and `Transform`s.

### Imports
We need a few basic imports to reference the components we need.

```scala mdoc:silent
import chisel3._
import chisel3.experimental.{annotate, ChiselAnnotation, RunFirrtlTransform}
import chisel3.internal.InstanceId

import firrtl._
import firrtl.annotations.{Annotation, SingleTargetAnnotation}
import firrtl.annotations.{CircuitTarget, ModuleTarget, InstanceTarget, ReferenceTarget, Target}
```

### Define an `Annotation` and a `Transform`

First, define an `Annotation` that contains a string associated with a `Target` thing in the Chisel circuit.
This `InfoAnnotation` extends [`SingleTargetAnnotation`](https://www.chisel-lang.org/api/firrtl/1.2.0/firrtl/annotations/SingleTargetAnnotation.html), an `Annotation` associated with *one* thing in a FIRRTL circuit:

```scala mdoc:silent
/** An annotation that contains some string information */
case class InfoAnnotation(target: Target, info: String) extends SingleTargetAnnotation[Target] {
  def duplicate(newTarget: Target) = this.copy(target = newTarget)
}
```

Second, define a `Transform` that consumes this `InfoAnnotation`.
This `InfoTransform` simply reads all annotations, prints any `InfoAnnotation`s it finds, and removes them.

```scala mdoc:invisible
object Issue1228 {
  /* Workaround for https://github.com/freechipsproject/firrtl/pull/1228 */
  abstract class Transform extends firrtl.Transform {
    override def name: String = this.getClass.getName
  }
}
import Issue1228.Transform
```

```scala mdoc:silent
/** A transform that reads InfoAnnotations and prints information about them */
class InfoTransform() extends Transform with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.HighForm

  override def execute(state: CircuitState): CircuitState = {
    println("Starting transform 'IdentityTransform'")

    val annotationsx = state.annotations.flatMap{
      case InfoAnnotation(a: CircuitTarget, info) =>
        println(s"  - Circuit '${a.serialize}' annotated with '$info'")
        None
      case InfoAnnotation(a: ModuleTarget, info) =>
        println(s"  - Module '${a.serialize}' annotated with '$info'")
        None
      case InfoAnnotation(a: InstanceTarget, info) =>
        println(s"  - Instance '${a.serialize}' annotated with '$info'")
        None
      case InfoAnnotation(a: ReferenceTarget, info) =>
        println(s"  - Component '${a.serialize} annotated with '$info''")
        None
      case a =>
        Some(a)
    }

    state.copy(annotations = annotationsx)
  }
}
```

> Note: `inputForm` and `outputForm` will be deprecated in favor of a new dependency API that allows transforms to specify their dependencies more specifically than with circuit forms.
> Full backwards compatibility for `inputForm` and `outputForm` will be maintained, however.

### Create a Chisel API/Annotator

Now, define a Chisel API to annotate Chisel things with this `InfoAnnotation`.
This is commonly referred to as an "annotator".

Here, define an object, `InfoAnnotator` with a method `info` that generates `InfoAnnotation`s.
This uses the `chisel3.experimental.annotate` passed an anonymous `ChiselAnnotation` object.
The need for this `ChiselAnnotation` (which is different from an actual FIRRTL `Annotation`) is that no FIRRTL circuit exists at the time the `info` method is called.
This is delaying the generation of the `InfoAnnotation` until the full circuit is available.

This annotator also mixes in the `RunFirrtlTransform` trait (abstract in the `transformClass` method) because this annotator, whenever used, should result in the FIRRTL compiler running the custom `InfoTransform`.

```scala mdoc:silent
object InfoAnnotator {
  def info(component: InstanceId, info: String): Unit = {
    annotate(new ChiselAnnotation with RunFirrtlTransform {
      def toFirrtl: Annotation = InfoAnnotation(component.toTarget, info)
      def transformClass = classOf[InfoTransform]
    })
  }
}
```

> Note: there are a number of different approaches to writing an annotator.
> You could use a trait that you mix into a `Module`, an object (like is done above), or any other software approach.
> The specific choice of how you implement this is up to you!

### Using the Chisel API

Now, we can use the method `InfoAnnotation.info` to create annotations that associate strings with specific things in a FIRRTL circuit.
Below is a Chisel `Module`, `ModC`, where both the actual module is annotated as well as an output.

```scala mdoc:silent
class ModC(widthC: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(widthC.W))
    val out = Output(UInt(widthC.W))
  })
  io.out := io.in

  InfoAnnotator.info(this, s"ModC($widthC)")

  InfoAnnotator.info(io.out, s"ModC(ignore param)")
}
```

### Running the Compilation

Compiling this circuit to Verilog will then result in the `InfoTransform` running and the added `println`s showing information about the components annotated.

```scala mdoc:compile-only
import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.ChiselStage

(new ChiselStage).execute(Array.empty, Seq(ChiselGeneratorAnnotation(() => new ModC(4))))
```
