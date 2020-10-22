// See LICENSE for license details.

package chiselTests

import org.scalatest._
import firrtl._
import chisel3._
import java.io.File

import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.util.experimental.InlineInstance
import firrtl.options.{Dependency, TargetDirAnnotation}
import firrtl.stage.RunFirrtlTransformAnnotation
import logger.{LogLevel, LogLevelAnnotation}

import scala.io.Source

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
}

class ForceNamesSpec extends FlatSpec with Matchers with chisel3.BackendCompilationUtilities with Utils with Inside {

  val targetDir = clearTestDir(baseTestDir)

  def recursiveListFiles(f: File): Seq[File] =
    if (f.isDirectory) {
      f.listFiles.flatMap(recursiveListFiles)
    } else {
      Seq(f)
    }

  def stage = new ChiselStage {
    override val targets = Seq(
      Dependency[chisel3.stage.phases.Elaborate],
      Dependency[chisel3.stage.phases.Convert],
      Dependency[firrtl.stage.phases.Compiler],
    )
  }

  def run[T <: RawModule](dut: => T, testName: String, info: LogLevel.Value = LogLevel.None): List[String] = {
    val testDir = targetDir.getAbsolutePath + s"/$testName"
    val annos = List(
      TargetDirAnnotation(testDir),
      LogLevelAnnotation(info),
      RunFirrtlTransformAnnotation(new ForceNamesTransform),
      ChiselGeneratorAnnotation(() => dut)
    )

    stage.execute(Array(), annos)

    val src = Source.fromFile(recursiveListFiles(new File(testDir)).head)
    val list = src.getLines.toList
    src.close()
    list

  }
  "Force Names on a wrapping instance" should "work" in {
    val verilog = run(new ForceNamesHierarchy.WrapperExample, "wrapper")
    exactly(1, verilog) should include ("MyLeaf INST")
  }
  "Force Names on an instance port" should "work" in {
    val verilog = run(new ForceNamesHierarchy.RenamePortsExample, "instports", LogLevel.Info)
    atLeast(1, verilog) should include ("input  [2:0] in")
  }
  "Force Names with a conflicting name" should "work or error" in {

  }
}
