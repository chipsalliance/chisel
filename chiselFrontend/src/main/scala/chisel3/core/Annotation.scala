// See LICENSE for license details.

package chisel3.core

import scala.language.existentials

import chisel3.internal.{Builder, InstanceId}
import firrtl.Transform
import firrtl.annotations.{Annotation, CircuitName, ComponentName, ModuleName}
import firrtl.transforms.DontTouchAnnotation

/** Interface for Annotations in Chisel
  *
  * Defines a conversion to a corresponding FIRRTL Annotation
  */
trait ChiselAnnotation {
  /** Conversion to FIRRTL Annotation */
  def toFirrtl: Annotation
}
object ChiselAnnotation {
  @deprecated("Write a custom ChiselAnnotation subclass instead", "3.1")
  def apply(component: InstanceId, transformClass: Class[_ <: Transform], value: String) =
    ChiselLegacyAnnotation(component, transformClass, value)
  @deprecated("Write a custom ChiselAnnotation subclass instead", "3.1")
  def unapply(anno: ChiselAnnotation): Option[(InstanceId, Class[_ <: Transform], String)] =
    anno match {
      case ChiselLegacyAnnotation(c, t, v) => Some(c, t, v)
      case _ => None
    }
}

/** Mixin for [[ChiselAnnotation]] that instantiates an associated FIRRTL Transform when this
  * Annotation is present during a run of [[chisel3.Driver.execute]]. Automatic Transform
  * instantiation is *not* supported when the Circuit and Annotations are serialized before invoking
  * FIRRTL.
  */
// TODO There should be a FIRRTL API for this instead
trait RunFirrtlTransform extends ChiselAnnotation {
  def transformClass: Class[_ <: Transform]
}

// This exists for implementation reasons, we don't want people using this type directly
final case class ChiselLegacyAnnotation private[chisel3] (
    component: InstanceId,
    transformClass: Class[_ <: Transform],
    value: String) extends ChiselAnnotation with RunFirrtlTransform {
  def toFirrtl: Annotation = Annotation(component.toNamed, transformClass, value)
}
private[chisel3] object ChiselLegacyAnnotation

object annotate { // scalastyle:ignore object.name
  def apply(anno: ChiselAnnotation): Unit = {
    Builder.annotations += anno
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
    annotate(new ChiselAnnotation { def toFirrtl = DontTouchAnnotation(data.toNamed) })
    data
  }
}

