package chiselTests.simulator

import chisel3._
import chisel3.experimental.ExtModule
import chisel3.simulator._
import circt.stage.ChiselStage
import chisel3.util.{HasExtModuleInline, HasExtModulePath, HasExtModuleResource}
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import chisel3.util.HasBlackBoxInline
import svsim._

class DPIVerilatorSimulator(val workspacePath: String) extends SingleBackendSimulator[verilator.Backend] {
  val backend = verilator.Backend.initializeFromProcessEnvironment()
  val tag = "verilator"
  val commonCompilationSettings = CommonCompilationSettings(libraryPaths = Some(Seq("/chisel3/simulator/dpi.cpp")))
  val backendSpecificCompilationSettings = verilator.Backend.CompilationSettings()
}

class DPITest extends Module {
  val dpiImpl = s"""
|#include <stdint.h>
|#include <iostream>
|
|extern "C" void hello()
|{
|    std::cout << "hello from c++\\n";
|}
|
|extern "C" void add(uint32_t* lhs, uint32_t* rhs, uint32_t *result)
|{
|    *result = *lhs + *rhs;
|}
|
|struct CounterState
|{
|    int counter = 0;
|};
|
|extern "C" void create_counter(CounterState** result)
|{
|    *result = new CounterState;
|}
|
| extern "C" void increment_counter(CounterState** state, uint32_t *result)
|{
|    *result = (*state)->counter++;
|}
  """.stripMargin

  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val add_result = Output(UInt(32.W))
    val counter_result = Output(UInt(32.W))
  })

  // Void function
  Intrinsic(
    "circt_dpi_call",
    "functionName" -> "hello",
    "isClocked" -> 1
  )(clock, 1.U)

  // Stateless function with result
  val result = IntrinsicExpr(
    "circt_dpi_call",
    UInt(32.W),
    "functionName" -> "add",
    "isClocked" -> 1
  )(clock, 1.U, io.a, io.b)

  // Statefull function with result
  val counter = IntrinsicExpr(
    "circt_dpi_call",
    UInt(64.W),
    "functionName" -> "create_counter",
    "isClocked" -> 1
  )(clock, reset.asUInt)

  // Workaround weird issue of SimToSV Dialect conversion error.
  val tmpCounter = counter
  // Currently need to insert has been reset to avoid null-pointer dereference.
  val has_been_reset = IntrinsicExpr("circt_has_been_reset", Bool())(clock, reset)
  val counter_result = IntrinsicExpr(
    "circt_dpi_call",
    UInt(32.W),
    "functionName" -> "increment_counter",
    "isClocked" -> 1
  )(clock, has_been_reset, dontTouch(tmpCounter))

  class DummyDPI extends BlackBox with HasBlackBoxInline {
    val io = IO(new Bundle {})
    setInline("dpi.cc", dpiImpl)
    setInline(s"$desiredName.sv", s"module $desiredName(); endmodule")
  }

  val dummy = Module(new DummyDPI)

  io.add_result := result
  io.counter_result := counter_result
}

class DPISpec extends AnyFunSpec with Matchers {
  describe("Chisel Simulator") {
    it("runs DPI correctly") {
      val simulator = new DPIVerilatorSimulator("test_run_dir/simulator/DPISimulator")
      simulator
        .simulate(new DPITest()) { module =>
          import PeekPokeAPI._
          val dpi = module.wrapped
          dpi.reset.poke(true)
          dpi.clock.step()

          dpi.reset.poke(false)
          dpi.io.a.poke(24.U)
          dpi.io.b.poke(36.U)
          dpi.clock.step()

          dpi.io.add_result.expect(60)
          dpi.io.counter_result.peek()
          dpi.io.counter_result.expect(0)

          dpi.clock.step()
          dpi.io.counter_result.peek()
          dpi.io.counter_result.expect(1)
        }
        .result
    }
  }
}
