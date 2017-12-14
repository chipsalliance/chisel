// See LICENSE for license details.

package chisel3.core

import scala.language.existentials

import chisel3.internal.{Builder, InstanceId}
import firrtl.Transform
import firrtl.annotations.{Annotation, CircuitName, ComponentName, ModuleName}

/**
  * This is a stand-in for the firrtl.Annotations.Annotation because at the time this annotation
  * is created the component cannot be resolved, into a targetString.  Resolution can only
  * happen after the circuit is elaborated
  * @param component       A chisel thingy to be annotated, could be module, wire, reg, etc.
  * @param transformClass  A fully-qualified class name of the transformation pass
  * @param value           A string value to be used by the transformation pass
  */
case class ChiselAnnotation(component: InstanceId, transformClass: Class[_ <: Transform], value: String) {
  def toFirrtl: Annotation = {
    val circuitName = CircuitName(component.pathName.split("""\.""").head)
    component match {
      case m: BaseModule =>
        Annotation(
          ModuleName(m.name, circuitName), transformClass, value)
      case _ =>
        Annotation(
          ComponentName(
            component.instanceName, ModuleName(component.parentModName, circuitName)), transformClass, value)
    }
  }
}

/** Marks that a signal should not be removed by Chisel and Firrtl optimization passes
  *
  * @example {{{
  * class MyModule extends Module {
  *   val io = IO(new Bundle {
  *     val a = Input(UInt(32.W))
  *     val b = Output(UInt(32.W))
  *   })
  *   io.b := io.a
  *   val dead = io.a +% 1.U // normally dead would be pruned by DCE
  *   dontTouch(dead) // Marking it as such will preserve it
  * }
  * }}}
  *
  * @note Calling this on [[Data]] creates an annotation that Chisel emits to a separate annotations
  * file. This file must be passed to FIRRTL independently of the `.fir` file. The execute methods
  * in [[chisel3.Driver]] will pass the annotations to FIRRTL automatically.
  */
object dontTouch { // scalastyle:ignore object.name
  /** Marks a signal to be preserved in Chisel and Firrtl
    *
    * @note Requires the argument to be bound to hardware
    * @param data The signal to be marked
    * @return Unmodified signal `data`
    */
  def apply[T <: Data](data: T)(implicit compileOptions: CompileOptions): T = {
    if (compileOptions.checkSynthesizable) {
      requireIsHardware(data, "Data marked dontTouch")
    }
    // TODO unify with firrtl.transforms.DontTouchAnnotation
    val anno = ChiselAnnotation(data, classOf[firrtl.Transform], "DONTtouch!")
    Builder.annotations += anno
    data
  }
}

