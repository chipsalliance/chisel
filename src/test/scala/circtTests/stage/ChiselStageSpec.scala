// SPDX-License-Identifier: Apache-2.0

package circtTests.stage

import chisel3.stage.ChiselGeneratorAnnotation

import circt.stage.ChiselStage

import firrtl.annotations.DeletedAnnotation
import firrtl.stage.FirrtlCircuitAnnotation

import java.io.File

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

object ChiselStageSpec {

  import chisel3._

  class FooBundle extends Bundle {
    val a = Input(Bool())
    val b = Output(Bool())
  }

  class Foo extends RawModule {
    val a = IO(new FooBundle)
    val b = IO(Flipped(new FooBundle))
    b <> a
  }

}

class ChiselStageSpec extends AnyFunSpec with Matchers {

  describe("ChiselStage") {

    it("should compile a Chisel module to FIRRTL dialect") {

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

    it("should compile a Chisel module to RTL dialect") {

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

    it("should compile a Chisel module to SystemVerilog") {

      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target", "systemverilog",
        "--target-dir", targetDir.toString
      )

      val expectedOutput = new File(targetDir, "Foo.sv")
      expectedOutput.delete()

      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Foo)))
        .map {
          case DeletedAnnotation(_, a) => a
          case a => a
        }

      info(s"'$expectedOutput' exists")
      expectedOutput should (exist)

    }
  }

  describe("ChiselStage handover to CIRCT") {

    /*Test that widths were not inferred in the FIRRTL passed to CIRCT */
    it("should handover at CHIRRTL") {

      import chisel3._

      class Foo extends RawModule {
        val a = IO(Input(UInt(1.W)))
        val b = IO(Output(UInt()))

        b := a
      }

      val args: Array[String] = Array(
        "--target", "firrtl",
        "--target-dir", "test_run_dir/ChiselStageSpec/handover/low",
        "--handover", "chirrtl"
      )

      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case FirrtlCircuitAnnotation(circuit) => circuit
          case DeletedAnnotation(_, FirrtlCircuitAnnotation(circuit)) => circuit
        }.map(_.serialize)
        .get should not include ("b : UInt<")

    }

    /* Test that widths were inferred in the FIRRTL passed to CIRCT */
    it("should handover at High FIRRTL") {

      import chisel3._

      class Foo extends RawModule {
        val a = IO(Input(UInt(1.W)))
        val b = IO(Output(UInt()))

        b := a
      }

      val args: Array[String] = Array(
        "--target", "firrtl",
        "--target-dir", "test_run_dir/ChiselStageSpec/handover/low",
        "--handover", "high"
      )

      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case FirrtlCircuitAnnotation(circuit) => circuit
          case DeletedAnnotation(_, FirrtlCircuitAnnotation(circuit)) => circuit
        }.map(_.serialize)
        .get should include ("b : UInt<1>")

    }

    /* Test that a subaccess was removed in the FIRRTL passed to CIRCT */
    it("should handover at Middle FIRRTL") {

      import chisel3._

      class Foo extends RawModule {
        val a = IO(Input(Vec(2, Bool())))
        val b = IO(Input(Bool()))
        val c = IO(Output(Bool()))
        c := a(b)
      }

      val args: Array[String] = Array(
        "--target", "firrtl",
        "--target-dir", "test_run_dir/ChiselStageSpec/handover/low",
        "--handover", "middle"
      )

      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case FirrtlCircuitAnnotation(circuit) => circuit
          case DeletedAnnotation(_, FirrtlCircuitAnnotation(circuit)) => circuit
        }.map(_.serialize)
        .get should not include ("a[b]")

    }

    /* Test that aggregates were lowered in the FIRRTL passed to CIRCT */
    it("should handover at Low FIRRTL") {

      import chisel3._

      class Foo extends RawModule {
        val b = IO(
          new Bundle {
            val a = Output(Bool())
            val b = Output(Bool())
          }
        )
        b := DontCare
      }

      val args: Array[String] = Array(
        "--target", "firrtl",
        "--target-dir", "test_run_dir/ChiselStageSpec/handover/low",
        "--handover", "low"
      )

      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case FirrtlCircuitAnnotation(circuit) => circuit
          case DeletedAnnotation(_, FirrtlCircuitAnnotation(circuit)) => circuit
        }.map(_.serialize)
        .get should include ("b_a")

    }

    /* Test that primops were folded in the FIRRTL passed to CIRCT */
    it("should handover at Low Optimized FIRRTL") {

      import chisel3._

      class Foo extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))
        b := a | 1.U(1.W)
      }

      val args: Array[String] = Array(
        "--target", "firrtl",
        "--target-dir", "test_run_dir/ChiselStageSpec/handover/lowopt",
        "--handover", "lowopt"
      )

      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case FirrtlCircuitAnnotation(circuit) => circuit
          case DeletedAnnotation(_, FirrtlCircuitAnnotation(circuit)) => circuit
        }.map(_.serialize)
        .get should include ("""b <= UInt<1>("h1")""")

    }
  }

  describe("ChiselStage$") {

    it("should emit FIRRTL dialect") {

      ChiselStage.emitFIRRTLDialect(new ChiselStageSpec.Foo) should include (" firrtl.module")

    }

    it("should emit RTL dialect") {

      ChiselStage.emitRTLDialect(new ChiselStageSpec.Foo) should include (" rtl.module")

    }

    it("should emit SystemVerilog") {

      ChiselStage.emitSystemVerilog(new ChiselStageSpec.Foo) should include ("endmodule")

    }

  }
}
