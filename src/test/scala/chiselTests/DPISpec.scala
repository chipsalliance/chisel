package chiselTests.simulator

import scala.io.Source
import chisel3._
import chisel3.experimental.ExtModule
import chisel3.simulator._
import chisel3.util.circt.dpi._
import circt.stage.ChiselStage
import chisel3.util.{HasExtModuleInline, HasExtModulePath, HasExtModuleResource}

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import chisel3.util.{HasBlackBoxInline, HasBlackBoxPath}
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
                   |extern "C" void print(const char* msg)
                   |{
                   |    std::cout << msg;
                   |}
                   |
                   |extern "C" void add(int lhs, int rhs, int* result)
                   |{
                   |    *result = lhs + rhs;
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
                   |extern "C" void increment_counter(CounterState* state, int* result)
                   |{
                   |    *result = state->counter++;
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

  def serializeSmallString(str: String) = {
    assert(str.length <= 7)
    val nullTerminated: Seq[Byte] = (str.getBytes().toIndexedSeq).padTo(8, 0)
    VecInit(nullTerminated.map(_.U(8.W)))
  }

  object PrintSmallStringFn extends DPIClockedVoidFunctionDecl {
    val functionName = "print"
    val inputs = Seq("str")
    final def apply(self: Vec[UInt]) = super.apply(clock, true.B, self)
  }

  PrintSmallStringFn(serializeSmallString("rabbit\n"))
  PrintSmallStringFn(serializeSmallString("cat\n"))


  // Stateless function with result
  val result = RawClockedNonVoidFunctionCall("add", UInt(32.W))(clock, true.B, io.a, io.b)

  // A function that allocates a state and returns a pointer.
  val counter = RawClockedNonVoidFunctionCall("create_counter", UInt(64.W))(clock, reset.asBool)

  // Currently need to insert has been reset to avoid null-pointer dereference.
  val has_been_reset = IntrinsicExpr("circt_has_been_reset", Bool())(clock, reset)
  val counter_result =
    RawClockedNonVoidFunctionCall("increment_counter", UInt(32.W))(clock, has_been_reset, counter)

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

class FirFitlerDPIStruct(val clock: Clock, val reset: Bool, coeffs: Seq[Int]) extends DPIStruct {
  val bitWidth = 64

  // DPI declarations.
  object FirFilterDPIDecl {
    object ResetFn extends DPIClockedVoidFunctionDecl {
      val functionName = "fir_filter_reset"
      val inputs = Seq("self")
      final def apply(self: DPIType.Chandle) = super.apply(clock, reset, self)
    }

    object CreateFn extends DPIClockedNonVoidFunctionDecl[UInt] {
      val functionName = "fir_filter_new"
      val inputs = Seq("coeff", "length")
      val retType = DPIType.ChandleType
      final def apply(coeff: Vec[UInt], length: UInt) = super.call(coeff, length)
    }

    object TickFn extends DPIClockedNonVoidFunctionDecl[UInt] {
      val functionName = "fir_filter_tick"
      val inputs = Seq("self", "input")
      val retType = UInt(bitWidth.W)
    }
  }

  val self = {
    val coeffsVec = VecInit(coeffs.map(_.U(bitWidth.W)))
    val len = coeffs.length.U(bitWidth.W)
    initialize(FirFilterDPIDecl.CreateFn, coeffsVec, len)
  }

  def tick(input: UInt): UInt = callMemberFn(FirFilterDPIDecl.TickFn)(input)
  def resetFn() = callMemberFn(FirFilterDPIDecl.ResetFn)()
}

class FirFilter(coeffs: Seq[Int]) extends Module {
  val bitWidth = 64
  val io = IO(new Bundle {
    val in = Input(UInt(bitWidth.W))
    val out = Output(UInt(bitWidth.W))
  })

  val firFilter = new FirFitlerDPIStruct(clock, reset.asBool, coeffs)

  val result = firFilter.tick(io.in)
  io.out := result

  class DummyDPI extends BlackBox with HasBlackBoxPath with HasBlackBoxInline {
    val io = IO(new Bundle {})
    addPath("src/test/resources/chisel3/fir-filter.cpp")
    setInline(s"$desiredName.sv", s"module $desiredName(); endmodule")
  }

  val dummy = Module(new DummyDPI)
}

class DPIFirSpec extends AnyFunSpec with Matchers {
  describe("DPI Fir Test") {
    it("runs movingSum3Filter correctly") {
      val simulator = new VerilatorSimulator("test_run_dir/simulator/FirFilterSimulator")
      simulator
        .simulate(new FirFilter(Seq(1, 1, 1))) { module =>
          import PeekPokeAPI._
          val dpi = module.wrapped
          dpi.reset.poke(true)
          dpi.clock.step()

          dpi.reset.poke(false)
          dpi.io.in.poke(1.U)
          dpi.clock.step()

          dpi.io.in.poke(1.U)
          dpi.io.out.peek()
          dpi.io.out.expect(1)

          dpi.clock.step()

          dpi.io.in.poke(1.U)
          dpi.io.out.peek()
          dpi.io.out.expect(2)
        }
        .result
    }
    it("runs delayFilter correctly") {
      val simulator = new VerilatorSimulator("test_run_dir/simulator/FirFilterSimulator2")
      simulator
        .simulate(new FirFilter(Seq(0, 1))) { module =>
          import PeekPokeAPI._
          val dpi = module.wrapped
          dpi.reset.poke(true)
          dpi.clock.step()

          dpi.reset.poke(false)
          dpi.io.in.poke(1.U)
          dpi.clock.step()

          dpi.io.in.poke(1.U)
          dpi.io.out.peek()
          dpi.io.out.expect(0)

          dpi.clock.step()

          dpi.io.in.poke(1.U)
          dpi.io.out.peek()
          dpi.io.out.expect(1)
        }
        .result
    }
  }
}
