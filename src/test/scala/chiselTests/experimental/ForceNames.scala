// See LICENSE for license details.

package chiselTests

import firrtl._
import chisel3._
import chisel3.core.annotate
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.util.experimental.{ForceNameAnnotation, ForceNamesTransform, InlineInstance, forceName}
import firrtl.annotations.{Annotation, ReferenceTarget}
import firrtl.options.{Dependency, TargetDirAnnotation}
import firrtl.stage.RunFirrtlTransformAnnotation
import logger.{LogLevel, LogLevelAnnotation}

/** Object containing Modules used for testing */
object ForceNamesHierarchy {
  class WrapperExample extends MultiIOModule {
    val IN = IO(Input(UInt(3.W)))
    val OUT = IO(Output(UInt(3.W)))
    val inst = Module(new Wrapper)
    inst.IN := IN
    OUT := inst.OUT
    forceName(OUT, "out")
  }
  class Wrapper extends MultiIOModule with InlineInstance {
    val IN = IO(Input(UInt(3.W)))
    val OUT = IO(Output(UInt(3.W)))
    val inst = Module(new MyLeaf)
    forceName(inst, "INST")
    inst.IN := IN
    OUT := inst.OUT
  }
  class MyLeaf extends MultiIOModule {
    val IN = IO(Input(UInt(3.W)))
    val OUT = IO(Output(UInt(3.W)))
    OUT := IN
  }
  class RenamePortsExample extends MultiIOModule {
    val IN = IO(Input(UInt(3.W)))
    val OUT = IO(Output(UInt(3.W)))
    val inst = Module(new MyLeaf)
    inst.IN := IN
    OUT := inst.OUT
    forceName(inst.IN, "in")
  }
  class ConflictingName extends MultiIOModule {
    val IN = IO(Input(UInt(3.W)))
    val OUT = IO(Output(UInt(3.W)))
    OUT := IN
    forceName(OUT, "IN")
  }
  class BundleName extends MultiIOModule {
    val IN = IO(new Bundle {
      val a = Input(UInt(3.W))
      val b = Input(UInt(3.W))
    })
    val OUT = IO(Output(UInt(3.W)))
    OUT := IN.a + IN.b
  }
}

class ForceNamesSpec extends ChiselFlatSpec {

  def run[T <: RawModule](dut: => T, testName: String, inputAnnos: Seq[Annotation] = Nil, info: LogLevel.Value = LogLevel.None): Iterable[String] = {
    def stage = new ChiselStage {
      override val targets = Seq(
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.Convert],
        Dependency[firrtl.stage.phases.Compiler],
      )
    }

    val annos = List(
      TargetDirAnnotation("test_run_dir/ForceNames"),
      LogLevelAnnotation(info),
      RunFirrtlTransformAnnotation(new ForceNamesTransform),
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
    exactly(1, verilog) should include ("MyLeaf INST")
  }
  "Force Names on an instance port" should "work" in {
    val verilog = run(new ForceNamesHierarchy.RenamePortsExample, "instports")
    atLeast(1, verilog) should include ("input  [2:0] in")
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
        Seq(ForceNameAnnotation(ReferenceTarget("BundleName", "BundleName", Nil, "IN", Nil), "in"))
      )
    }
  }
}
