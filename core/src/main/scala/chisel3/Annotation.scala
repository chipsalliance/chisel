// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import scala.language.existentials
import scala.annotation.nowarn
import chisel3.internal.Builder
import chisel3.{Data, HasTarget, InstanceId, RawModule}
import chisel3.experimental.AnyTargetable
import firrtl.annotations._
import firrtl.options.Unserializable
import firrtl.transforms.{DedupGroupAnnotation, NoDedupAnnotation}

/** Interface for Annotations in Chisel
  *
  * Defines a conversion to a corresponding FIRRTL Annotation
  */
@deprecated(
  "Avoid custom annotations. If you must use annotations, use annotate.apply method that takes Data",
  "Chisel 6.7.0"
)
trait ChiselAnnotation {

  /** Conversion to FIRRTL Annotation */
  def toFirrtl: Annotation
}

/** Enhanced interface for Annotations in Chisel
  *
  *  Defines a conversion to corresponding FIRRTL Annotation(s)
  */
@deprecated(
  "Avoid custom annotations. If you must use annotations, use annotate.apply method that takes Data",
  "Chisel 6.7.0"
)
trait ChiselMultiAnnotation {
  def toFirrtl: Seq[Annotation]
}

@nowarn("msg=Avoid custom annotations")
object annotate {
  @deprecated(
    "Avoid custom annotations. If you must use annotations, use annotate.apply method that takes Data",
    "Chisel 6.7.0"
  )
  def apply(anno: ChiselAnnotation): Unit =
    Builder.annotations += anno

  @deprecated(
    "Avoid custom annotations. If you must use annotations, use annotate.apply method that takes Data",
    "Chisel 6.7.0"
  )
  def apply(annos: ChiselMultiAnnotation): Unit =
    Builder.newAnnotations += annos

  /** Create annotations.
    *
    * Avoid this API if possible.
    *
    * Anything being annotated must be passed as arguments so that Chisel can do safety checks.
    * The caller is still responsible for calling .toTarget on those arguments in mkAnnos.
    */
  def apply(targets: AnyTargetable*)(mkAnnos: => Seq[Annotation]): Unit = {
    targets.map(_.a).foreach {
      case d: Data => requireIsAnnotatable(d, "Data marked with annotation")
      case _ => ()
    }
    Builder.newAnnotations += new ChiselMultiAnnotation {
      def toFirrtl: Seq[Annotation] = mkAnnos
    }
  }

  /** Create annotations.
    *
    * Avoid this API if possible.
    *
    * Anything being annotated must be passed as arguments so that Chisel can do safety checks.
    * The caller is still responsible for calling .toTarget on those arguments in mkAnnos.
    */
  def apply[T: Targetable](targets: Seq[T])(mkAnnos: => Seq[Annotation]): Unit = {
    annotate(targets.map(t => AnyTargetable.toAnyTargetable(t)): _*)(mkAnnos)
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
    annotate(module)(Seq(NoDedupAnnotation(module.toNamed)))
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
    annotate(module)(Seq(DedupGroupAnnotation(module.toTarget, group)))
  }
}
