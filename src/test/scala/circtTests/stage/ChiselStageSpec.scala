// SPDX-License-Identifier: Apache-2.0

package circtTests.stage

import chisel3.stage.ChiselGeneratorAnnotation

import circt.stage.ChiselStage

import java.io.File

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object ChiselStageSpec {

  import chisel3._

  class Foo extends RawModule {
    val a = IO(Input(Bool()))
    val b = IO(Output(Bool()))
    b := ~a
  }

}

class ChiselStageSpec extends AnyFlatSpec with Matchers {

  behavior of "ChiselStage"

  it should "compile a Chisel module to FIRRTL dialect" in {

    val targetDir = new File("test_run_dir/ChiselStageSpec")

    val args: Array[String] = Array(
      "--target", "firrtl",
      "--target-dir", targetDir.toString
    )

    val expectedOutput = new File(targetDir, "Foo.fir.mlir")
    expectedOutput.delete()

    (new ChiselStage).execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo)))

    info(s"'$expectedOutput' exists")
    expectedOutput should (exist)

  }

  it should "compile a Chisel module to RTL dialect" in {

    val targetDir = new File("test_run_dir/ChiselStageSpec")

    val args: Array[String] = Array(
      "--target", "rtl",
      "--target-dir", targetDir.toString
    )

    val expectedOutput = new File(targetDir, "Foo.rtl.mlir")
    expectedOutput.delete()

    (new ChiselStage).execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo)))

    info(s"'$expectedOutput' exists")
    expectedOutput should (exist)

  }

  it should "compile a Chisel module to SystemVerilog" in {

    val targetDir = new File("test_run_dir/ChiselStageSpec")

    val args: Array[String] = Array(
      "--target", "systemverilog",
      "--target-dir", targetDir.toString
    )

    val expectedOutput = new File(targetDir, "Foo.sv")
    expectedOutput.delete()

    (new ChiselStage).execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo)))

    info(s"'$expectedOutput' exists")
    expectedOutput should (exist)

  }

  behavior of "ChiselStage$"

  it should "emit FIRRTL dialect" in {

    ChiselStage.emitFIRRTLDialect(new ChiselStageSpec.Foo) should include (" firrtl.module")

  }

  it should "emit RTL dialect" in {

    ChiselStage.emitRTLDialect(new ChiselStageSpec.Foo) should include (" rtl.module")

  }

  it should "emit SystemVerilog" in {

    ChiselStage.emitSystemVerilog(new ChiselStageSpec.Foo) should include ("endmodule")

  }

}
