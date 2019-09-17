// See LICENSE for license details.

package chisel3.experimental

import scala.language.existentials
import chisel3.internal.{Builder, InstanceId, LegacyModule}
import chisel3.{CompileOptions, Data}
import firrtl.Transform
import firrtl.annotations._
import firrtl.options.Unserializable
import firrtl.transforms.{DontTouchAnnotation, NoDedupAnnotation}

/** Interface for Annotations in Chisel
  *
  * Defines a conversion to a corresponding FIRRTL Annotation
  */
trait ChiselAnnotation {
  /** Conversion to FIRRTL Annotation */
  def toFirrtl: Annotation
}

/** Mixin for [[ChiselAnnotation]] that instantiates an associated FIRRTL Transform when this Annotation is present
  * during a run of
  * [[Driver$.execute(args:Array[String],dut:()=>chisel3\.RawModule)* Driver.execute]].
  * Automatic Transform instantiation is *not* supported when the Circuit and Annotations are serialized before invoking
  * FIRRTL.
  */
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

/** Marks that a module to be ignored in Dedup Transform in Firrtl pass
  *
  * @example {{{
  *  def fullAdder(a: UInt, b: UInt, myName: String): UInt = {
  *    val m = Module(new Module {
  *      val io = IO(new Bundle {
  *        val a = Input(UInt(32.W))
  *        val b = Input(UInt(32.W))
  *        val out = Output(UInt(32.W))
  *      })
  *      override def desiredName = "adder_" + myNname
  *      io.out := io.a + io.b
  *    })
  *    doNotDedup(m)
  *    m.io.a := a
  *    m.io.b := b
  *    m.io.out
  *  }
  *
  *class AdderTester extends Module
  *  with ConstantPropagationTest {
  *  val io = IO(new Bundle {
  *    val a = Input(UInt(32.W))
  *    val b = Input(UInt(32.W))
  *    val out = Output(Vec(2, UInt(32.W)))
  *  })
  *
  *  io.out(0) := fullAdder(io.a, io.b, "mod1")
  *  io.out(1) := fullAdder(io.a, io.b, "mod2")
  * }
  * }}}
  *
  * @note Calling this on [[Data]] creates an annotation that Chisel emits to a separate annotations
  * file. This file must be passed to FIRRTL independently of the `.fir` file. The execute methods
  * in [[chisel3.Driver]] will pass the annotations to FIRRTL automatically.
  */

object doNotDedup { // scalastyle:ignore object.name
  /** Marks a module to be ignored in Dedup Transform in Firrtl
    *
    * @param module The module to be marked
    * @return Unmodified signal `module`
    */
   def apply[T <: LegacyModule](module: T)(implicit compileOptions: CompileOptions): Unit = {
    annotate(new ChiselAnnotation { def toFirrtl = NoDedupAnnotation(module.toNamed) })
  }
}
