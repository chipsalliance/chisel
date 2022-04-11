// SPDX-License-Identifier: Apache-2.0

package circtTests.stage

import chisel3.stage.ChiselGeneratorAnnotation

import circt.stage.{
  ChiselStage,
  FirtoolOption,
  PreserveAggregate
}

import firrtl.annotations.DeletedAnnotation
import firrtl.EmittedVerilogCircuitAnnotation
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

  class Bar extends RawModule {
    val sel = IO(Input(UInt(3.W)))
    val in = IO(Input(Vec(8, UInt(8.W))))
    val out = IO(Output(UInt(8.W)))
    out := in(sel)
  }

  class BazBundle extends Bundle {
    val a = Input(UInt(3.W))
    val b = Input(UInt(4.W))
  }

  class Baz extends RawModule {
    val in = IO(Input(new BazBundle))
    val out = IO(Output(new BazBundle))
    out := in
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
        "--target", "hw",
        "--target-dir", targetDir.toString
      )

      val expectedOutput = new File(targetDir, "Foo.hw.mlir")
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

    it("should support custom firtool options") {
      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target", "systemverilog",
        "--target-dir", targetDir.toString
      )

      info(s"output contains a case statement using --lowering-options=disallowPackedArrays")
      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Bar),
                           FirtoolOption("--lowering-options=disallowPackedArrays")))
        .collectFirst {
          case EmittedVerilogCircuitAnnotation(a) => a
        }.get
        .value should include ("case")
    }

    it("should support aggregate preservation mode") {
      val targetDir = new File("test_run_dir/ChiselStageSpec")

      val args: Array[String] = Array(
        "--target", "systemverilog",
        "--target-dir", targetDir.toString
      )

      info(s"output contains a verilog struct using preserve-aggregate option")
      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new ChiselStageSpec.Baz),
                           PreserveAggregate))
        .collectFirst {
          case EmittedVerilogCircuitAnnotation(a) => a
        }.get
        .value should include ("struct")
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

  describe("ChiselStage custom transform support") {

    it("should work with InlineInstance") {

      import chisel3._
      import chisel3.util.experimental.InlineInstance

      trait SimpleIO { this: RawModule =>
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))
      }

      class Bar extends RawModule with SimpleIO with InlineInstance {
        b := ~a
      }

      class Foo extends RawModule with SimpleIO {
        val bar = Module(new Bar)
        bar.a := a
        b := bar.b
      }

      val targetDir = new File("test_run_dir/InlineInstance")

      val args: Array[String] = Array(
        "--target", "systemverilog",
        "--target-dir", targetDir.toString
      )

      (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case EmittedVerilogCircuitAnnotation(a) => a
        }.get
        .value should not include ("module Bar")
    }

    it("should work with FlattenInstance") {

      import chisel3._
      import chisel3.util.experimental.FlattenInstance

      trait SimpleIO { this: RawModule =>
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))
      }

      class Baz extends RawModule with SimpleIO {
        b := ~a
      }

      class Bar extends RawModule with SimpleIO {
        val baz = Module(new Baz)
        baz.a := a
        b := baz.b
      }

      class Foo extends RawModule with SimpleIO {
        val bar = Module(new Bar with FlattenInstance)
        bar.a := a
        b := bar.b
      }

      val targetDir = new File("test_run_dir/FlattenInstance")

      val args: Array[String] = Array(
        "--target", "systemverilog",
        "--target-dir", targetDir.toString
      )

      val verilog = (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case EmittedVerilogCircuitAnnotation(a) => a
        }.get
        .value

      verilog should not include ("module Baz")
      verilog should not include ("module Bar")

    }

  }

  describe("ChiselStage DontTouchAnnotation support") {

    it("should block removal of wires and nodes") {

      import chisel3._

      class Foo extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        val w = WireDefault(a)
        dontTouch(w)

        val n = ~w
        dontTouch(n)

        b := n
      }

      val targetDir = new File("test_run_dir/DontTouch")

      val args: Array[String] = Array(
        "--target", "systemverilog",
        "--target-dir", targetDir.toString
      )

      val verilog = (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case EmittedVerilogCircuitAnnotation(a) => a
        }.get
        .value

      verilog should include ("wire w")
      verilog should include ("wire n")

    }

  }

  describe("ChiselStage forceName support") {

    it("should work when forcing a module name") {

      import chisel3._
      import chisel3.util.experimental.{forceName, InlineInstance}

      trait SimpleIO { this: RawModule =>
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))
      }

      class Baz extends RawModule with SimpleIO {
        b := ~a
      }

      class Bar extends RawModule with SimpleIO with InlineInstance {
        val baz = Module(new Baz)
        baz.a := a
        b := baz.b
      }

      class Foo extends RawModule with SimpleIO {
        val bar = Module(new Bar)
        bar.a := a
        b := bar.b

        forceName(bar.baz)
      }

      val targetDir = new File("test_run_dir/ForceName")

      val args: Array[String] = Array(
        "--target", "systemverilog",
        "--target-dir", targetDir.toString
      )

      val verilog: String = (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case EmittedVerilogCircuitAnnotation(a) => a
        }.get
        .value

      verilog should include ("module Baz")

    }

  }

  describe("ChiselStage dedup behavior") {

    it("should be on by default and work") {

      import chisel3._

      class Baz extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        b := a
      }

      class Bar extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        b := a
      }

      class Foo extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        val bar = Module(new Bar)
        val baz = Module(new Baz)
        bar.a := a
        baz.a := a
        b := bar.b ^ baz.b
      }

      val targetDir = new File("test_run_dir/Dedup")

      val args: Array[String] = Array(
        "--target", "systemverilog",
        "--target-dir", targetDir.toString
      )

      val verilog: String = (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case EmittedVerilogCircuitAnnotation(a) => a
        }.get
        .value

      verilog should include ("module Bar")
      verilog should not include ("module Baz")

    }

    it("should respect the doNotDedup API") {

      import chisel3._
      import chisel3.experimental.doNotDedup

      class Baz extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        b := a
      }

      class Bar extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        b := a
      }

      class Foo extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        val bar = Module(new Bar)
        val baz = Module(new Baz)
        bar.a := a
        baz.a := a
        b := bar.b ^ baz.b
        doNotDedup(baz)
      }

      val targetDir = new File("test_run_dir/Dedup")

      val args: Array[String] = Array(
        "--target", "systemverilog",
        "--target-dir", targetDir.toString
      )

      val verilog: String = (new ChiselStage)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new Foo)))
        .collectFirst {
          case EmittedVerilogCircuitAnnotation(a) => a
        }.get
        .value

      verilog should include ("module Bar")
      verilog should include ("module Baz")

    }

  }

  describe("ChiselStage$") {

    it("should emit FIRRTL dialect") {

      ChiselStage.emitFIRRTLDialect(new ChiselStageSpec.Foo) should include (" firrtl.module")

    }

    it("should emit HW dialect") {

      ChiselStage.emitHWDialect(new ChiselStageSpec.Foo) should include (" hw.module")

    }

    it("should emit SystemVerilog") {

      ChiselStage.emitSystemVerilog(new ChiselStageSpec.Foo) should include ("endmodule")

    }

  }
}
