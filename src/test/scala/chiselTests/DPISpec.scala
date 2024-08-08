package chiselTests.simulator

import chisel3._
import chisel3.experimental.ExtModule
import chisel3.simulator._
import chisel3.util.circt.dpi._
import circt.stage.ChiselStage
import chisel3.util.{HasExtModuleInline, HasExtModulePath, HasExtModuleResource}
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import chisel3.util.HasBlackBoxInline
import svsim._

private object EmitDPIImplementation {
  def apply() = {
    val dpiImpl = s"""
                     |#include <stdint.h>
                     |#include <iostream>
                     |#include <svdpi.h>
                     |
                     |extern "C" void hello()
                     |{
                     |    std::cout << "hello from c++\\n";
                     |}
                     |
                     |extern "C" void add(int lhs, int rhs, int* result)
                     |{
                     |    *result = lhs + rhs;
                     |}
                     |
                     |extern "C" void sum(const svOpenArrayHandle array, int* result) {
                     |  int size = svSize(array, 1);
                     |  *result = 0;
                     |  for(size_t i = 0; i < size; ++i) {
                     |    svBitVecVal vec;
                     |    svGetBitArrElemVecVal(&vec, array, i);
                     |    *result += vec;
                     |  }
                     |}
                     |extern "C" void print_string(const svOpenArrayHandle array) {
                     |  int size = svSize(array, 1);
                     |  for(size_t i = 0; i < size; ++i) {
                     |    svBitVecVal vec[1];
                     |    svGetBitArrElemVecVal(vec, array, i);
                     |    std::cout << vec[0]&&0xff;
                     |  }
                     |}
  """.stripMargin

    class DummyDPI extends BlackBox with HasBlackBoxInline {
      val io = IO(new Bundle {})
      setInline("dpi.cc", dpiImpl)
      setInline(s"$desiredName.sv", s"module $desiredName(); endmodule")
    }
    val dummy = Module(new DummyDPI)

  }
}

class DPIIntrinsicTest extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val add_clocked_result = Output(UInt(32.W))
    val add_unclocked_result = Output(UInt(32.W))
  })

  EmitDPIImplementation()

  // Void function
  RawClockedVoidFunctionCall("hello")(clock, true.B)

  // Stateless function with result
  val result_clocked =
    RawClockedNonVoidFunctionCall("add", UInt(32.W), Some(Seq("lhs", "rhs")), Some("result"))(clock, true.B, io.a, io.b)
  val result_unclocked =
    RawUnclockedNonVoidFunctionCall("add", UInt(32.W), Some(Seq("lhs", "rhs")), Some("result"))(true.B, io.a, io.b)

  io.add_clocked_result := result_clocked
  io.add_unclocked_result := result_unclocked
}

object Hello extends DPIVoidFunctionImport {
  override val functionName = "hello"
  override val clocked = true
  final def apply() = super.call()
}

object AddClocked extends DPINonVoidFunctionImport[UInt] {
  override val functionName = "add"
  override val ret = UInt(32.W)
  override val clocked = true
  override val inputNames = Some(Seq("lhs", "rhs"))
  override val outputName = Some("result")
  final def apply(lhs: UInt, rhs: UInt): UInt = super.call(lhs, rhs)
}

object AddUnclocked extends DPINonVoidFunctionImport[UInt] {
  override val functionName = "add"
  override val ret = UInt(32.W)
  override val clocked = false
  override val inputNames = Some(Seq("lhs", "rhs"))
  override val outputName = Some("result")
  final def apply(lhs: UInt, rhs: UInt): UInt = super.call(lhs, rhs)
}

object Sum extends DPINonVoidFunctionImport[UInt] {
  override val functionName = "sum"
  override val ret = UInt(32.W)
  override val clocked = false
  override val inputNames = Some(Seq("array"))
  override val outputName = Some("result")
  final def apply(array: Vec[UInt]): UInt = super.call(array)
}

object PrintString extends DPIVoidFunctionImport {
  override val functionName = "print_string"
  final def apply(array: Vec[UInt]) = super.call(array)
}

class DPIAPITest extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val add_clocked_result = Output(UInt(32.W))
    val add_unclocked_result = Output(UInt(32.W))
    val sum_result = Output(UInt(32.W)) 
  })

  EmitDPIImplementation()

  Hello()

  val s = VecInit("string".getBytes().map(_.U(8.W)).toIndexedSeq)
  PrintString(s)
  val result_clocked = AddClocked(io.a, io.b)
  val result_unclocked = AddUnclocked(io.a, io.b)
  val result_sum = Sum(VecInit(Seq(io.a, io.b, io.a)))

  io.add_clocked_result := result_clocked
  io.add_unclocked_result := result_unclocked
  io.sum_result := result_sum
}

class DPISpec extends AnyFunSpec with Matchers {
  describe("DPI") {
    it("DPI intrinsics run correctly") {
      val simulator = new VerilatorSimulator("test_run_dir/simulator/DPIIntrinsic")

      simulator
        .simulate(new DPIIntrinsicTest()) { module =>
          import PeekPokeAPI._
          val dpi = module.wrapped
          dpi.reset.poke(true)
          dpi.clock.step()

          dpi.reset.poke(false)
          dpi.io.a.poke(24.U)
          dpi.io.b.poke(36.U)
          dpi.io.add_unclocked_result.peek()
          dpi.io.add_unclocked_result.expect(60)

          dpi.clock.step()
          dpi.io.a.poke(24.U)
          dpi.io.b.poke(12.U)
          dpi.io.add_clocked_result.peek()
          dpi.io.add_clocked_result.expect(60)
          dpi.io.add_unclocked_result.peek()
          dpi.io.add_unclocked_result.expect(36)
        }
        .result

      val outputFile = io.Source.fromFile("test_run_dir/simulator/DPIIntrinsic/workdir-verilator/simulation-log.txt")
      val output = outputFile.mkString
      outputFile.close()
      output should include("hello from c++")
    }
    it("DPI API run correctly") {
      val simulator = new VerilatorSimulator("test_run_dir/simulator/DPIAPI")

      simulator
        .simulate(new DPIAPITest()) { module =>
          import PeekPokeAPI._
          val dpi = module.wrapped
          dpi.reset.poke(true)
          dpi.clock.step()

          dpi.reset.poke(false)
          dpi.io.a.poke(24.U)
          dpi.io.b.poke(36.U)
          dpi.io.add_unclocked_result.peek()
          dpi.io.add_unclocked_result.expect(60)
          dpi.io.sum_result.peek()
          dpi.io.sum_result.expect(84)

          dpi.clock.step()
          dpi.io.a.poke(24.U)
          dpi.io.b.poke(12.U)
          dpi.io.add_clocked_result.peek()
          dpi.io.add_clocked_result.expect(60)
          dpi.io.add_unclocked_result.peek()
          dpi.io.add_unclocked_result.expect(36)
          dpi.io.sum_result.peek()
          dpi.io.sum_result.expect(60)
        }
        .result

      val outputFile = io.Source.fromFile("test_run_dir/simulator/DPIAPI/workdir-verilator/simulation-log.txt")
      val output = outputFile.mkString
      outputFile.close()
      output should include("hello from c++")
    }
    it("emits DPI correctly") {
      val verilog = ChiselStage.emitSystemVerilog(
        new DPIIntrinsicTest(),
        firtoolOpts = Array("--lowering-options=locationInfoStyle=none,disallowPortDeclSharing")
      )
      verilog should include("import \"DPI-C\" context function void add(")
      verilog should include("input  int lhs,")
      verilog should include("input  int rhs,")
      verilog should include("output int result")
    }
  }
}
