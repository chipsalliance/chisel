package chiselTests

import chisel3._
import chisel3.util.Enum
import chisel3.testers._
import chisel3.experimental.HasTests
import chisel3.experimental.hierarchy._

@instantiable
class ModuleWithTests extends Module with HasTests {
  @public val io = IO(new Bundle {
    val in = Input(UInt(32.W))
    val out = Output(UInt(32.W))
  })

  test("foo") { instance =>
    instance.io.in := 3.U
    assert(instance.io.out === 3.U)
  }

  test("bar") { instance =>
    instance.io.in := 5.U
    assert(instance.io.out =/= 0.U)
  }
}

class HasTestsSpec extends ChiselFlatSpec with FileCheck {
  it should "generate a module for each test" in {
    generateFirrtlAndFileCheck(new ModuleWithTests)(
      """
      | CHECK-DAG: module ModuleWithTests
      |
      | CHECK-DAG: public module ModuleWithTests_test_foo
      | CHECK: inst instance of ModuleWithTests
      |
      | CHECK-DAG: public module ModuleWithTests_test_bar
      | CHECK: inst instance of ModuleWithTests
      """
    )
  }
}
