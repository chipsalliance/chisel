package chiselTests

import chisel3._
import chisel3.util.Enum
import chisel3.testers._
import chisel3.experimental.inlinetest._
import chisel3.experimental.hierarchy._

class TestResultBundle extends Bundle {
  val finish = Output(Bool())
  val code = Output(UInt(8.W))
}

class TestHarnessWithResultIO[M <: RawModule](test: TestParameters[M, TestResultBundle]) extends Module {
  override def resetType = Module.ResetType.Synchronous
  override val desiredName = s"test_${test.dutName}_${test.testName}"
  val dut = Instance(test.dutDefinition)
  val result = IO(new TestResultBundle)
  result := test.body(dut)
}

object TestHarnessWithResultIO {
  implicit def enhancedTestHarness[M <: RawModule]: TestHarness[M, TestResultBundle] =
    new TestHarness[M, TestResultBundle] {
      def generate(test: TestParameters[M, TestResultBundle]): RawModule with Public =
        new TestHarnessWithResultIO(test) with Public
    }
}

@instantiable
class ModuleWithTests(ioWidth: Int = 32) extends Module with HasTests[ModuleWithTests] {
  @public val io = IO(new Bundle {
    val in = Input(UInt(ioWidth.W))
    val out = Output(UInt(ioWidth.W))
  })

  io.out := io.in

  test("foo") { instance =>
    instance.io.in := 3.U(ioWidth.W)
    assert(instance.io.out === 3.U): Unit
  }

  test("bar") { instance =>
    instance.io.in := 5.U(ioWidth.W)
    assert(instance.io.out =/= 0.U): Unit
  }

  import TestHarnessWithResultIO._
  test("enhanced") { instance =>
    val result = Wire(new TestResultBundle)
    val timer = RegInit(0.U)
    timer := timer + 1.U
    instance.io.in := 5.U(ioWidth.W)
    val outValid = instance.io.out =/= 0.U
    when(outValid) {
      result.code := 0.U
      result.finish := timer > 1000.U
    }.otherwise {
      result.code := 1.U
      result.finish := true.B
    }
    result
  }
}

class HasTestsSpec extends ChiselFlatSpec with FileCheck {
  it should "generate a public module for each test" in {
    generateFirrtlAndFileCheck(new ModuleWithTests)(
      """
      | CHECK: module ModuleWithTests
      |
      | CHECK: public module test_ModuleWithTests_foo
      | CHECK: input clock : Clock
      | CHECK: input reset : UInt<1>
      | CHECK: inst dut of ModuleWithTests
      |
      | CHECK: public module test_ModuleWithTests_bar
      | CHECK: input clock : Clock
      | CHECK: input reset : UInt<1>
      | CHECK: inst dut of ModuleWithTests
      |
      | CHECK: public module test_ModuleWithTests_enhanced
      | CHECK: input clock : Clock
      | CHECK: input reset : UInt<1>
      | CHECK: output result : { finish : UInt<1>, code : UInt<8>}
      | CHECK: inst dut of ModuleWithTests
      """
    )
  }

  it should "compile to verilog" in {
    generateSystemVerilogAndFileCheck(new ModuleWithTests)(
      """
      | CHECK: module test_ModuleWithTests_foo
      | CHECK: module test_ModuleWithTests_bar
      """
    )
  }
}
