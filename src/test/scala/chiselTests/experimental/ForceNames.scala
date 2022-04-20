// See LICENSE for license details.

package chiselTests

import firrtl._
import chisel3._
import chisel3.experimental.annotate
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.util.experimental.{forceName, ForceNameAnnotation, ForceNamesTransform, InlineInstance}
import firrtl.annotations.{Annotation, ReferenceTarget}
import firrtl.options.{Dependency, TargetDirAnnotation}
import firrtl.stage.RunFirrtlTransformAnnotation
import logger.{LogLevel, LogLevelAnnotation}

/** Object containing Modules used for testing */
object ForceNamesHierarchy {
  class WrapperExample extends Module {
    val in = IO(Input(UInt(3.W)))
    val out = IO(Output(UInt(3.W)))
    val inst = Module(new Wrapper)
    inst.in := in
    out := inst.out
    forceName(out, "outt")
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
  class RenamePortsExample extends Module {
    val in = IO(Input(UInt(3.W)))
    val out = IO(Output(UInt(3.W)))
    val inst = Module(new MyLeaf)
    inst.in := in
    out := inst.out
    forceName(inst.in, "inn")
  }
  class ConflictingName extends Module {
    val in = IO(Input(UInt(3.W)))
    val out = IO(Output(UInt(3.W)))
    out := in
    forceName(out, "in")
  }
  class BundleName extends Module {
    val in = IO(new Bundle {
      val a = Input(UInt(3.W))
      val b = Input(UInt(3.W))
    })
    val out = IO(Output(UInt(3.W)))
    out := in.a + in.b
  }
}

class ForceNamesSpec extends ChiselFlatSpec {

  def run[T <: RawModule](
    dut:        => T,
    testName:   String,
    inputAnnos: Seq[Annotation] = Nil,
    info:       LogLevel.Value = LogLevel.None
  ): Iterable[String] = {
    def stage = new ChiselStage {
      override val targets = Seq(
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.Convert],
        Dependency[firrtl.stage.phases.Compiler]
      )
    }

    val annos = List(
      TargetDirAnnotation("test_run_dir/ForceNames"),
      LogLevelAnnotation(info),
      ChiselGeneratorAnnotation(() => dut)
    ) ++ inputAnnos

    val ret = stage.execute(Array(), annos)
    val verilog = ret.collectFirst {
      case e: EmittedVerilogCircuitAnnotation => e.value.value
    }.get

    verilog.split("\\\n")
  }
  "Force Names on a wrapping instance" should "work" in {
    val verilog = run(new ForceNamesHierarchy.WrapperExample, "wrapper")
    exactly(1, verilog) should include("MyLeaf inst")
  }
  "Force Names on an instance port" should "work" in {
    val verilog = run(new ForceNamesHierarchy.RenamePortsExample, "instports")
    atLeast(1, verilog) should include("input  [2:0] inn")
  }
  "Force Names with a conflicting name" should "error" in {
    intercept[CustomTransformException] {
      run(new ForceNamesHierarchy.ConflictingName, "conflicts")
    }
  }
  "Force Names of an intermediate bundle" should "error" in {
    intercept[CustomTransformException] {
      run(
        new ForceNamesHierarchy.BundleName,
        "bundlename",
        Seq(ForceNameAnnotation(ReferenceTarget("BundleName", "BundleName", Nil, "in", Nil), "inn"))
      )
    }
  }
}
