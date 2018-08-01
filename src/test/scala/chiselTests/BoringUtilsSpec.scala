// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.util.Counter
import chisel3.testers.{BasicTester, TesterDriver}
import chisel3.util.experimental.BoringUtils
import firrtl.{
  CommonOptions,
  ExecutionOptionsManager,
  HasFirrtlOptions,
  FirrtlExecutionOptions,
  FirrtlExecutionSuccess,
  FirrtlExecutionFailure}
import firrtl.passes.wiring.WiringTransform

class InverterAfterWiring extends Module {
  val io = IO(
    new Bundle {
      val in = Input(Bool())
      val out = Output(Bool())
    }
  )

  val notIn = ~io.in
  io.out := io.in

  BoringUtils.addSource(notIn, "id")
  BoringUtils.addSink(io.out, "id")
}

class InverterAfterWiringTester extends BasicTester {
  val dut = Module(new InverterAfterWiring)

  val (c, done) = Counter(true.B, 2)

  dut.io.in := c(0)
  assert(dut.io.out =/= dut.io.in)

  when (done) { stop() }
}

class BoringUtilsSpec extends ChiselFlatSpec with ChiselRunners {
  it should "connect within a module" in {
    runTester(new InverterAfterWiringTester) should be (true)
  }
}
