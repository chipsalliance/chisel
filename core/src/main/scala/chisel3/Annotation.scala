// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import scala.language.existentials
import chisel3.internal.Builder
import chisel3.{Data, InstanceId, RawModule}
import firrtl.annotations._
import firrtl.options.Unserializable
import firrtl.transforms.{DedupGroupAnnotation, DontTouchAnnotation, NoDedupAnnotation}

/** Interface for Annotations in Chisel
  *
  * Defines a conversion to a corresponding FIRRTL Annotation
  */
trait ChiselAnnotation {

  /** Conversion to FIRRTL Annotation */
  def toFirrtl: Annotation
}

/** Enhanced interface for Annotations in Chisel
  *
  *  Defines a conversion to corresponding FIRRTL Annotation(s)
  */
trait ChiselMultiAnnotation {
  def toFirrtl: Seq[Annotation]
}

object annotate {
  def apply(anno: ChiselAnnotation): Unit = {
    Builder.annotations += anno
  }
  def apply(annos: ChiselMultiAnnotation): Unit = {
    Builder.newAnnotations += annos
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
  * class AdderTester extends Module
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
  */
object doNotDedup {

  /** Marks a module to be ignored in Dedup Transform in Firrtl
    *
    * @param module The module to be marked
    * @return Unmodified signal `module`
    */
  def apply[T <: RawModule](module: T): Unit = {
    annotate(new ChiselAnnotation { def toFirrtl = NoDedupAnnotation(module.toNamed) })
  }
}

object dedupGroup {

  /** Assign the targeted module to a dedup group. Only modules in the same group may be deduplicated.
    *
    * @param module The module to be marked
    * @param group The name of the dedup group to assign the module to
    * @return Unmodified signal `module`
    */
  def apply[T <: BaseModule](module: T, group: String): Unit = {
    annotate(new ChiselAnnotation { def toFirrtl = DedupGroupAnnotation(module.toTarget, group) })
  }
}
