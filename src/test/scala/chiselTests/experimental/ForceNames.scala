// See LICENSE for license details.

package chiselTests

import firrtl._
import chisel3._
import chisel3.experimental.annotate
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util.experimental.{forceName, InlineInstance}
import circt.stage.ChiselStage
import firrtl.annotations.{Annotation, ReferenceTarget}
import firrtl.options.{Dependency, TargetDirAnnotation}
import logger.{LogLevel, LogLevelAnnotation}

/** Object containing Modules used for testing */
object ForceNamesHierarchy {
  class WrapperExample extends Module {
    val in = IO(Input(UInt(3.W)))
    val out = IO(Output(UInt(3.W)))
    val inst = Module(new Wrapper)
    inst.in := in
    out := inst.out
  }
  class Wrapper extends Module with InlineInstance {
    val in = IO(Input(UInt(3.W)))
    val out = IO(Output(UInt(3.W)))
    val inst = Module(new MyLeaf)
    forceName(inst, "inst")
    inst.in := in
    out := inst.out
  }
  class MyLeaf extends Module {
    val in = IO(Input(UInt(3.W)))
    val out = IO(Output(UInt(3.W)))
    out := in
  }
}

class ForceNamesSpec extends ChiselFlatSpec {

  def run[T <: RawModule](
    dut:        => T,
    testName:   String,
    inputAnnos: Seq[Annotation] = Nil,
    info:       LogLevel.Value = LogLevel.None
  ): Iterable[String] = {
    val stage = new ChiselStage

    val annos = List(
      TargetDirAnnotation("test_run_dir/ForceNames"),
      ChiselGeneratorAnnotation(() => dut)
    ) ++ inputAnnos

    val ret = stage.execute(Array("--target", "systemverilog"), annos)
    val verilog = ret.collectFirst {
      case e: EmittedVerilogCircuitAnnotation => e.value.value
    }.get

    verilog.split("\\\n")
  }
  "Force Names on a wrapping instance" should "work" in {
    val verilog = run(new ForceNamesHierarchy.WrapperExample, "wrapper")
    exactly(1, verilog) should include("MyLeaf inst")
  }
}
