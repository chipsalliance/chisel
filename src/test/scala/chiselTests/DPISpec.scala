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
                   |    std::cout << "Create counter\\n";
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
  RawClockedVoidFunctionCall("hello")(clock, true.B)

  // Stateless function with result
  val result = RawClockedNonVoidFunctionCall("add", UInt(32.W))(clock, true.B, io.a, io.b)

  // A function that allocates a state and returns a pointer.
  val counter = RawClockedNonVoidFunctionCall("create_counter", UInt(64.W))(clock, reset.asBool)

  // FIXME: Workaround weird CIRCT issue of SimToSV Dialect conversion error.
  val tmpCounter = counter

  // Currently need to insert has been reset to avoid null-pointer dereference.
  val has_been_reset = IntrinsicExpr("circt_has_been_reset", Bool())(clock, reset)
  val counter_result =
    RawClockedNonVoidFunctionCall("increment_counter", UInt(32.W))(clock, has_been_reset, dontTouch(tmpCounter))

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
      val simulator = new VerilatorSimulator("test_run_dir/simulator/DPISimulator")
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
