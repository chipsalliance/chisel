// See LICENSE for license details.

package chiselTests.aop

import chisel3.testers.BasicTester
import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.aop.Select.{PredicatedConnect, When, WhenNot}
import chisel3.aop.{Aspect, Select}
import chisel3.experimental.ExtModule
import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}
import firrtl.AnnotationSeq

import scala.reflect.runtime.universe.TypeTag

class SelectTester(results: Seq[Int]) extends BasicTester
    with chisel3.internal.DisableCommandMemoization {
  val values = VecInit(results.map(_.U))
  val counter = RegInit(0.U(results.length.W))
  val added = counter + 1.U
  counter := added
  val overflow = counter >= values.length.U
  val nreset = reset.asBool() === false.B
  val selected = values(counter)
  val zero = 0.U + 0.U
  when(overflow) {
    counter := zero
    stop()
  }.otherwise {
    when(nreset) {
      assert(counter === values(counter))
      printf("values(%d) = %d\n", counter, selected)
    }
  }
}

case class SelectAspect[T <: RawModule, X](selector: T => Seq[X], desired: T => Seq[X]) extends Aspect[T] {
  override def toAnnotation(top: T): AnnotationSeq = {
    val results = selector(top)
    val desiredSeq = desired(top)
    assert(results.length == desiredSeq.length, s"Failure! Results $results have different length than desired $desiredSeq!")
    val mismatches = results.zip(desiredSeq).flatMap {
      case (res, des) if res != des => Seq((res, des))
      case other => Nil
    }
    assert(mismatches.isEmpty,s"Failure! The following selected items do not match their desired item:\n" + mismatches.map{
      case (res: Select.Serializeable, des: Select.Serializeable) => s"  ${res.serialize} does not match:\n  ${des.serialize}"
      case (res, des) => s"  $res does not match:\n  $des"
    }.mkString("\n"))
    Nil
  }
}

class SelectSpec extends ChiselFlatSpec {

  def execute[T <: RawModule, X](dut: () => T, selector: T => Seq[X], desired: T => Seq[X])(implicit tTag: TypeTag[T]): Unit = {
    val ret = new chisel3.stage.ChiselStage().run(
      Seq(
        new chisel3.stage.ChiselGeneratorAnnotation(dut),
        SelectAspect(selector, desired),
        new chisel3.stage.ChiselOutputFileAnnotation("test_run_dir/Select.fir")
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

  "Test" should "pass if selecting correct printfs" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Seq(Select.printfs(dut).last) },
      { dut: SelectTester =>
        Seq(Select.Printf(
          Seq(
            When(Select.ops("eq")(dut).last.asInstanceOf[Bool]),
            When(dut.nreset),
            WhenNot(dut.overflow)
          ),
          Printable.pack("values(%d) = %d\n", dut.counter, dut.selected),
          dut.clock
        ))
      }
    )
  }

  "Test" should "pass if selecting correct connections" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Select.connectionsTo(dut)(dut.counter) },
      { dut: SelectTester =>
        Seq(PredicatedConnect(Nil, dut.counter, dut.added, false),
          PredicatedConnect(Seq(When(dut.overflow)), dut.counter, dut.zero, false))
      }
    )
  }

  "Test" should "pass if selecting ops by kind" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Select.ops("tail")(dut) },
      { dut: SelectTester => Seq(dut.added, dut.zero) }
    )
  }

  "Test" should "pass if selecting ops" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Select.ops(dut).collect { case ("tail", d) => d} },
      { dut: SelectTester => Seq(dut.added, dut.zero) }
    )
  }

  "Test" should "pass if selecting correct stops" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Seq(Select.stops(dut).last) },
      { dut: SelectTester =>
        Seq(Select.Stop(
          Seq(
            When(Select.ops("eq")(dut).dropRight(1).last.asInstanceOf[Bool]),
            When(dut.nreset),
            WhenNot(dut.overflow)
          ),
          1,
          dut.clock
        ))
      }
    )
  }

  "Blackboxes" should "be supported in Select.instances" in {
    class BB extends ExtModule { }
    class Top extends RawModule {
      val bb = Module(new BB)
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate(1).asInstanceOf[DesignAnnotation[Top]].design
    val bbs = Select.collectDeep(top) { case b: BB => b }
    assert(bbs.size == 1)
  }

}

