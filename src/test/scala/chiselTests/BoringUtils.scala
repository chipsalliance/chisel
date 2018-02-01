// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.util.Counter
import chisel3.testers.BasicTester
import chisel3.util.BoringUtils
import firrtl.{
  CommonOptions,
  ExecutionOptionsManager,
  HasFirrtlOptions,
  FirrtlExecutionOptions,
  FirrtlExecutionSuccess,
  FirrtlExecutionFailure}
import firrtl.passes.wiring.WiringTransform

class InverterAfterWiring extends Module with BoringUtils {
  val io = IO(new Bundle {
    val in = Input(Bool())
    val out = Output(Bool())
  })

  val notIn = ~io.in
  io.out := io.in

  addSource(notIn, "id")
  addSink(io.out, "id")
}

class PassthroughAfterWiringTester extends BasicTester {
  val dut = Module(new InverterAfterWiring)

  val (c, done) = Counter(true.B, 2)

  dut.io.in := c(0)
  printf("[test] (in, out): (%x, %x)\n", dut.io.in, dut.io.out)
  // assert(dut.io.out =/= dut.io.in)

  when (done) { stop() }
}

class BoringUtilsSpec extends ChiselFlatSpec with BackendCompilationUtilities {
  it should "connect within a module" in {
    val target = "PassthroughAfterWiringTester"

    val path = createTestDirectory(target)
    val fname = new File(path, target)

    val cppHarness =  new File(path, "top.cpp")
    copyResourceToFile("/chisel3/top.cpp", cppHarness)

    Driver.execute(Array(
      "--target-dir", path.getAbsolutePath),
      () => new PassthroughAfterWiringTester)
    val passed = if ((verilogToCpp(target, path, Seq.empty, cppHarness) #&&
        cppToExe(target, path)).! == 0) {
      executeExpectingSuccess(target, path)
    } else {
      false
    }
    assert(passed)
  }
}
