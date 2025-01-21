package chiselTests

import chisel3._
import chisel3.util.Enum
import chisel3.testers._
import chisel3.experimental.HasTests
import chisel3.experimental.hierarchy._

@instantiable
class ModuleWithTests(ioWidth: Int = 32) extends Module with HasTests[Unit] {
  @public val io = IO(new Bundle {
    val in = Input(UInt(ioWidth.W))
    val out = Output(UInt(ioWidth.W))
  })

  io.out := io.in

  test("foo") { instance =>
    instance.io.in := 3.U(ioWidth.W)
    assert(instance.io.out === 3.U)
  }

  test("bar") { instance =>
    instance.io.in := 5.U(ioWidth.W)
    assert(instance.io.out =/= 0.U)
  }
}

trait HasTestsWithSuccess extends HasTests[Bool] { module: RawModule =>
  override protected def generateTestHarness(
    testName:   String,
    definition: Definition[module.type],
    body:       Instance[module.type] => Bool
  ): RawModule with Public =
    new Module with Public {
      override val desiredName = s"${module.desiredName}_${testName}"
      val dut = Instance(definition)
      val result = body(dut.asInstanceOf[Instance[module.type]])
      val success = IO(Output(Bool()))
      success := result
    }

}

@instantiable
class ModuleWithTestsWithSuccess(ioWidth: Int = 32) extends Module with HasTestsWithSuccess {
  @public val io = IO(new Bundle {
    val in = Input(UInt(ioWidth.W))
    val out = Output(UInt(ioWidth.W))
  })

  io.out := io.in

  test("foo") { instance =>
    instance.io.in := 3.U(ioWidth.W)
    assert(instance.io.out === 3.U)
    WireInit(true.B)
  }

  test("bar") { instance =>
    instance.io.in := 5.U(ioWidth.W)
    assert(instance.io.out =/= 0.U)
    WireInit(true.B)
  }
}

class HasTestsSpec extends ChiselFlatSpec with FileCheck {
  it should "generate a public module for each test" in {
    generateFirrtlAndFileCheck(new ModuleWithTests)(
      """
      | CHECK: module ModuleWithTests
      |
      | CHECK: public module ModuleWithTests_foo
      | CHECK: input clock
      | CHECK: input reset
      | CHECK: inst dut of ModuleWithTests
      |
      | CHECK: public module ModuleWithTests_bar
      | CHECK: input clock : Clock
      | CHECK: input reset : UInt<1>
      | CHECK: inst dut of ModuleWithTests
      """
    )
  }

  it should "compile to verilog" in {
    generateSystemVerilogAndFileCheck(new ModuleWithTests)(
      """
      | CHECK: module ModuleWithTests_foo
      | CHECK: module ModuleWithTests_foo
      | CHECK: module ModuleWithTests_bar
      """
    )
  }

  it should "generate a public module for each test with a custom testharness" in {
    generateFirrtlAndFileCheck(new ModuleWithTestsWithSuccess)(
      """
      | CHECK: module ModuleWithTestsWithSuccess
      |
      | CHECK: public module ModuleWithTestsWithSuccess_foo
      | CHECK: input clock
      | CHECK: input reset
      | CHECK: output success : UInt<1>
      | CHECK: inst dut of ModuleWithTestsWithSuccess
      |
      | CHECK: public module ModuleWithTestsWithSuccess_bar
      | CHECK: input clock
      | CHECK: input reset
      | CHECK: output success : UInt<1>
      | CHECK: inst dut of ModuleWithTestsWithSuccess
      """
    )
  }
}
