package chiselTests.aop

import chisel3._
import chisel3.experimental.hierarchy.Definition
import scala.reflect.runtime.universe.TypeTag

trait Utils {
  def execute[T <: RawModule : TypeTag, X](dut: () => T, selector: Definition[T] => Seq[X], desired: Definition[T] => Seq[X]): Unit = {
    val ret = new chisel3.stage.ChiselStage().run(
      Seq(
        new chisel3.stage.ChiselGeneratorAnnotation(dut),
        TestAspects.SelectAspect(selector, desired),
        new chisel3.stage.ChiselOutputFileAnnotation("test_run_dir/Select.fir")
      )
    )
  }
}