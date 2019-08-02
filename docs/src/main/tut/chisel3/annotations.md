---
layout: docs
title:  "Annotations"
section: "chisel3"
---
Annotations are used to mark modules and their elements in a way that can be accessed by Firrtl transformation passes.  Custom passes and the annotations that guide their behavior extend the circuit generation capabilities of Chisel/Firrtl.  This article focuses on the approach to building a library that contains Annotations and Transforms.  We will walk through  [src/test/scala/chiselTests/AnnotatingDiamondSpec.scala](https://github.com/ucb-bar/chisel3/blob/master/src/test/scala/chiselTests/AnnotatingDiamondSpec.scala) to see the basic concepts.

### Imports
We need a few basic imports to reference the components we need.  The chisel3 is a standard
```tut
import chisel3._
import chisel3.experimental.ChiselAnnotation
import chisel3.internal.InstanceId
import firrtl.{CircuitForm, CircuitState, LowForm, Transform}
import firrtl.annotations.{Annotation, ModuleName, Named}
```
### Write a transform
This is an identity transform that returns whatever it is passed without alteration.  See [Writing Firrtl Transforms](/ucb-bar/firrtl/wiki) for the gory details on writing transforms that actually do something.
```tut:silent
class IdentityTransform extends Transform {
  override def inputForm: CircuitForm = LowForm
  override def outputForm: CircuitForm = LowForm

  override def execute(state: CircuitState): CircuitState = {
    getMyAnnotations(state) match {
      case Nil => state
      case myAnnotations =>
        // Use annotations contained in the myAnnotations list to modify state
        // and return that modified state.
        state
    }
  }
}
```
This creates a transform that operates on low Firrtl (LowForm) and returns low Firrtl.  ```getMyAnnotations``` returns a list of annotations for your pass.  This example does nothing with those annotations.
### Create an Annotation Factory
The following creates an annotation that is connected to your transform, note the ```classOf[IdentityTransform]```.  The unapply is a convenience method for extracting information from your annotation by using the Scala ```match``` operator.
```tut:silent
object IdentityAnnotation {
  def apply(target: Named, value: String): Annotation = Annotation(target, classOf[IdentityTransform], value)

  def unapply(a: Annotation): Option[(Named, String)] = a match {
    case Annotation(named, t, value) if t == classOf[IdentityTransform] => Some((named, value))
    case _ => None
  }
}
```
> note ```target: Named``` identifies a firrtl circuit component.  Annotations can refer to specific elements of a Module
> such as registers or wires, or can point to a Module in the case of some more generic transformation.

### Create an Annotator
An Annotator is a trait that only be applied to a Module.  It provides an abstraction layer over the underlying Chisel annotation system.  In this example, the ```identify``` annotator takes an kind of circuit component reference (i.e. ```InstanceId```) and packages it with ```value``` to be available in the firrtl pass.  The library writer could place restrictions on the type of component and value.
> The ```value``` passed to the Annotator does not have to be a string, but it must serializable into a string
> for the ```value``` parameter of the ```ChiselAnnotation``` being created.

```tut:silent
trait IdentityAnnotator {
  self: Module =>
  def identify(component: InstanceId, value: String): Unit = {
    annotate(ChiselAnnotation(component, classOf[IdentityTransform], value))
  }
}
```

### Using the Annotator
Here is a module that uses our ```IdentityAnnotator```
```tut:silent
class ModC(widthC: Int) extends Module with IdentityAnnotator {
  val io = IO(new Bundle {
    val in = Input(UInt(widthC.W))
    val out = Output(UInt(widthC.W))
  })
  io.out := io.in

  identify(this, s"ModC($widthC)")

  identify(io.out, s"ModC(ignore param)")
}
```

There are several things to note here.  ModC includes the ```with IdentityAnnotator``` which adds the identity method to it.  The ```identify(this, s"ModC($widthC)")``` annotates an instance of ModC as it is created.  It value annotations includes the ```widthC``` parameter to the constructor.  In this case that could be used to distinguish transformation behavior between different instances of ModC.  The ```identify(io.out, s"ModC(ignore param)")``` annotates io.out but with a fixed string.  In contrast the previous annotation, multiple instances of ModC would have result in a single ```io.out``` annotation here.
