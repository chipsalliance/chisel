// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.util.Counter
import chisel3.testers.BasicTester
import chisel3.experimental.MultiIOModule
import chisel3.util.experimental.BoringUtils
import firrtl.{CommonOptions, ExecutionOptionsManager, HasFirrtlOptions, FirrtlExecutionOptions, FirrtlExecutionSuccess,
  FirrtlExecutionFailure}
import firrtl.passes.wiring.WiringTransform

class BoringUtilsSpec extends ChiselFlatSpec with ChiselRunners {
  trait HasInput { this: MultiIOModule =>
    val width: Int
    val in = IO(Input(UInt(width.W)))
  }

  trait HasOutput { this: MultiIOModule =>
    val width: Int
    val out = IO(Output(UInt(width.W)))
  }

  class PassThrough(val width: Int) extends MultiIOModule with HasInput with HasOutput {
    out := in
  }

  trait BoringInverter { this: PassThrough =>
    val notIn = ~in
    BoringUtils.addSource(notIn, "id")
    BoringUtils.addSink(out, "id")
  }

  class InverterAfterWiringTester extends BasicTester {
    val passThrough = Module(new PassThrough(1))
    val inverter = Module(new PassThrough(1) with BoringInverter)

    val (c, done) = Counter(true.B, 2)

    Seq(passThrough, inverter).map( _.in := c(0) )
    chisel3.assert(passThrough.out === passThrough.in, "'PassThrough' was not passthrough")
    chisel3.assert(inverter.out =/= inverter.in,
                   "'PassThrough with BoringInverter' didn't invert (did the WiringTransform run?)")

    when (done) { stop() }
  }

  behavior of "BoringUtils"

  it should "connect within a module" in {
    runTester(new InverterAfterWiringTester) should be (true)
  }

  class Foo(val width: Int) extends MultiIOModule with HasInput {
    val x = Wire(UInt(width.W))
    x := in
    BoringUtils.addSource(x, "uniqueId")
  }

  class Bar(val width: Int) extends MultiIOModule with HasOutput {
    val x = Wire(UInt(width.W))
    out := x
    x := 0.U // Dummy connection to make this a valid circuit
    BoringUtils.addSink(x, "uniqueId")
  }

  class Top(val width: Int) extends MultiIOModule with HasInput with HasOutput {
    val foo = Module(new Foo(width))
    foo.in := in
    val bar = Module(new Bar(width))
    out := bar.out
  }

  class TopTester(width: Int) extends BasicTester {
    val dut = Module(new Top(width))

    val inVec = VecInit(Range(0, math.pow(2, width).toInt).map(_.U))
    val (c, done) = Counter(true.B, inVec.size)
    dut.in := inVec(c)
    printf("dut.in: 0x%x\n", dut.in)
    printf("dut.out: 0x%x\n", dut.out)
    chisel3.assert(dut.out === dut.in)

    when (done) { stop() }
  }

  it should "connect across modules" in {
	  runTester(new TopTester(4)) should be (true)
  }
}
