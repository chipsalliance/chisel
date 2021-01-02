// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform, annotate}
import chisel3.internal.InstanceId
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.testers.BasicTester
import firrtl.{CircuitForm, CircuitState, DependencyAPIMigration, LowForm, Transform}
import firrtl.annotations.{
  CircuitName,
  CircuitTarget,
  SingleTargetAnnotation,
  ReferenceTarget
}
import firrtl.stage.Forms
import org.scalatest._
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

/** Simple Annotation to demonstrate [[ReferenceTarget]] annotation behavior.
  * 
  * @param target the FIRRTL target to annotate
  * @param value a simple documentation string
  */
case class ReferenceTargetAnnotation(target: ReferenceTarget, value: String) extends SingleTargetAnnotation[ReferenceTarget] {
  def duplicate(n: ReferenceTarget): ReferenceTargetAnnotation = this.copy(target = n)
}

/** ChiselAnnotation that corresponds to the [[ReferenceTargetAnnotation]] FIRRTL annotation
  *
  *  @param target the Data which will be converted to a FIRRTL ReferenceTarget
  *  @param value the documentation string which will be copied directly to the FIRRTL annotation
  */
case class ReferenceTargetChiselAnnotation(target: Data, value: String)
    extends ChiselAnnotation {
  def toFirrtl: ReferenceTargetAnnotation = ReferenceTargetAnnotation(target.toTarget, value)
  //def transformClass: Class[ReferenceTargetTransform] = classOf[ReferenceTargetTransform]
}

/** Simple module for demonstrating ReferenceTarget annotation behavior
  *
  *  @param annotateLiteral if true, will illegally attempt to annotate a Literal value as a ReferenceTarget
  */
class Mod(annotateLiteral: Boolean) extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(4, Bool()))
    val out = Output(Vec(4, Bool()))
  })

  io.out := io.in

  val a = false.B
  val b = WireInit(a)

  annotate(ReferenceTargetChiselAnnotation(io.in(0), s"io.in[0]: Needs Tokenizing"))
  annotate(ReferenceTargetChiselAnnotation(b, s"b: Constant Wire"))
  if (annotateLiteral){
    annotate(ReferenceTargetChiselAnnotation(a, s"a: Literal Value"))
  }
}

/** Tester which instantiates a [[Mod]] */
class ModTester(annotateLiteral: Boolean) extends BasicTester {
  val dut = Module(new Mod(annotateLiteral))

  stop()
}


class ReferenceTargetSpec extends AnyFreeSpec with Matchers {

  "Data should successfully be annotated as ReferenceTargets" in {

    val annos = (new ChiselStage)
      .execute(Array("--target-dir", "test_run_dir", "--no-run-firrtl"),
        Seq(ChiselGeneratorAnnotation(() => new Mod(annotateLiteral=false))))
      .filter {
      case _: ReferenceTargetAnnotation => true
      case _                            => false
    }.toSeq

    info(annos.toString)
    annos should have length (2)
  }


  "Literal Data should fail to be annotated as ReferenceTarget" in {

    intercept[firrtl.options.StageError] {

      val annos = (new ChiselStage)
        .execute(Array("--target-dir", "test_run_dir", "--no-run-firrtl"),
          Seq(ChiselGeneratorAnnotation(() => new Mod(annotateLiteral=true))))
        .filter {
        case _: ReferenceTargetAnnotation => true
        case _                            => false
      }.toSeq
    }
  }
}
