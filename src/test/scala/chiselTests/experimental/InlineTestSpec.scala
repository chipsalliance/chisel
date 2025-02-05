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

// Here is a testharness that consumes some kind of hardware from the test body, e.g.
// a finish and pass/fail interface.
object TestHarnessWithResultIO {
  class TestHarnessWithResultIOModule[M <: RawModule](val test: TestParameters[M, TestResultBundle])
      extends Module
      with TestHarnessModule[M, TestResultBundle] {
    val result = IO(new TestResultBundle)
    result := elaborateTest()
  }
  implicit def testharnessGenerator[M <: RawModule]: TestHarnessGenerator[M, TestResultBundle] =
    new TestHarnessGenerator[M, TestResultBundle] {
      def generate(test: TestParameters[M, TestResultBundle]) = new TestHarnessWithResultIOModule(test)
    }
}

object TestHarnessWithMonitorSocket {
  // Here is a testharness that expects some sort of interface on its DUT, e.g. a probe
  // socket to which to attach a monitor.
  class TestHarnessWithMonitorSocketModule[M <: RawModule with HasMonitorSocket](val test: TestParameters[M, Unit])
      extends Module
      with TestHarnessModule[M, Unit] {
    val monitor = Module(new ProtocolMonitor(dut.monProbe.cloneType))
    monitor.io :#= probe.read(dut.monProbe)
  }
  implicit def testharnessGenerator[M <: RawModule with HasMonitorSocket]: TestHarnessGenerator[M, Unit] =
    new TestHarnessGenerator[M, Unit] {
      def generate(test: TestParameters[M, Unit]): RawModule with Public = new TestHarnessWithMonitorSocketModule(test)
    }
}

@instantiable
trait HasMonitorSocket { this: RawModule =>
  protected def makeProbe(bundle: ProtocolBundle): ProtocolBundle = {
    val monProbe = IO(probe.Probe(chiselTypeOf(bundle)))
    probe.define(monProbe, probe.ProbeValue(bundle))
    monProbe
  }
  @public val monProbe: ProtocolBundle
}

class ProtocolBundle(width: Int) extends Bundle {
  val in = Flipped(UInt(width.W))
  val out = UInt(width.W)
}

class ProtocolMonitor(bundleType: ProtocolBundle) extends Module {
  val io = IO(Input(bundleType))
  assert(io.in === io.out, "in === out")
}

@instantiable
class ModuleWithTests(ioWidth: Int = 32) extends Module with HasMonitorSocket with HasTests[ModuleWithTests] {
  @public val io = IO(new ProtocolBundle(ioWidth))

  override val monProbe = makeProbe(io)

  io.out := io.in

  test("foo") { instance =>
    instance.io.in := 3.U(ioWidth.W)
    assert(instance.io.out === 3.U): Unit
  }

  test("bar") { instance =>
    instance.io.in := 5.U(ioWidth.W)
    assert(instance.io.out =/= 0.U): Unit
  }

  {
    import TestHarnessWithResultIO._
    test("with_result") { instance =>
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

  {
    import TestHarnessWithMonitorSocket._
    test("with_monitor") { instance =>
      instance.io.in := 5.U(ioWidth.W)
      assert(instance.io.out =/= 0.U): Unit
    }
  }
}

class InlineTestSpec extends ChiselFlatSpec with FileCheck {
  it should "generate a public module for each test" in {
    println(circt.stage.ChiselStage.emitCHIRRTL(new ModuleWithTests))
    generateFirrtlAndFileCheck(new ModuleWithTests)(
      """
      | CHECK:      module ModuleWithTests
      | CHECK:        output monProbe : Probe<{ in : UInt<32>, out : UInt<32>}>
      |
      | CHECK:      public module test_ModuleWithTests_foo
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_bar
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_with_result
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : UInt<1>
      | CHECK-NEXT:   output result : { finish : UInt<1>, code : UInt<8>}
      | CHECK:        inst dut of ModuleWithTests
      |
      | CHECK:      public module test_ModuleWithTests_with_monitor
      | CHECK-NEXT:   input clock : Clock
      | CHECK-NEXT:   input reset : UInt<1>
      | CHECK:        inst dut of ModuleWithTests
      | CHECK:        inst monitor of ProtocolMonitor
      | CHECK-NEXT:   connect monitor.clock, clock
      | CHECK-NEXT:   connect monitor.reset, reset
      | CHECK-NEXT:   connect monitor.io.out, read(dut.monProbe).out
      | CHECK-NEXT:   connect monitor.io.in, read(dut.monProbe).in
      """
    )
  }

  it should "compile to verilog" in {
    generateSystemVerilogAndFileCheck(new ModuleWithTests)(
      """
      | CHECK: module ModuleWithTests
      | CHECK: module test_ModuleWithTests_foo
      | CHECK: module test_ModuleWithTests_bar
      | CHECK: module test_ModuleWithTests_with_result
      | CHECK: module test_ModuleWithTests_with_monitor
      """
    )
  }
}
