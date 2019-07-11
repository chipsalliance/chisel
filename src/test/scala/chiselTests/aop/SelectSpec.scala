// See LICENSE for license details.

package chiselTests.aop

import chisel3.testers.BasicTester
import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.aop.{Aspect, Select}
import chisel3.experimental.RawModule
import firrtl.AnnotationSeq
import scala.reflect.runtime.universe.TypeTag

class SelectTester(results: Seq[Int]) extends BasicTester {
  val values = VecInit(results.map(_.U))
  val counter = RegInit(0.U(results.length.W))
  counter := counter + 1.U
  when(counter >= values.length.U) {
    stop()
  }.otherwise {
    when(reset.asBool() === false.B) {
      printf("values(%d) = %d\n", counter, values(counter))
      assert(counter === values(counter))
    }
  }
}

case class SelectAspect[T <: RawModule, X](selector: T => Seq[X], desired: T => Seq[X])(implicit tTag: TypeTag[T]) extends Aspect[T] {
  override def toAnnotation(top: T): AnnotationSeq = {
    val results = selector(top)
    val desiredSeq = desired(top)
    assert(results.length == desiredSeq.length, s"Failure! Results $results have different length than desired $desiredSeq!")
    val mismatches = results.zip(desiredSeq).flatMap {
      case (res, des) if res != des => Seq((res, des))
      case other => Nil
    }
    assert(mismatches.isEmpty,s"Failure! The following selected items do not match their desired item:\n"
        + mismatches.map{ case (res, des) => s"  $res does not match $des" }.mkString("\n"))
    Nil
  }
}

class SelectSpec extends ChiselFlatSpec {

  def execute[T <: RawModule, X](dut: () => T, selector: T => Seq[X], desired: T => Seq[X])(implicit tTag: TypeTag[T]) = {
    new chisel3.stage.ChiselStage().run(
      Seq(
        new chisel3.stage.ChiselGeneratorAnnotation(dut),
        SelectAspect(selector, desired)
      )
    )
  }

  "Test" should "pass if selecting correct registers" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Select.registers(dut) },
      { dut: SelectTester => Seq(dut.counter) }
    )
  }

  "Test" should "pass if selecting correct wires" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Select.wires(dut) },
      { dut: SelectTester => Seq(dut.values) }
    )
  }


}

