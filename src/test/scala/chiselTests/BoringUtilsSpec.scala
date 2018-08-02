// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.util.Counter
import chisel3.testers.BasicTester
import chisel3.experimental.{MultiIOModule, BaseModule}
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

  trait WireX { this: BaseModule =>
    val width: Int
    val x = Wire(UInt(width.W))
  }

  class Foo(val width: Int) extends MultiIOModule with HasInput with WireX {
    x := in
  }

  class Bar(val width: Int) extends MultiIOModule with HasOutput with WireX {
    out := x
    x := 0.U // Default value. Output is zero unless we bore...
  }

  class Top(val width: Int) extends MultiIOModule with HasInput with HasOutput {
    // source(0) -> unused
    // source(1) -> source
    val source = Seq.fill(2)(Module(new Foo(width)))
    // sink(0) -> unconnected, always outputs zero
    // sink(1) -> sink
    // sink(2) -> sink
    val sink = Seq.fill(3)(Module(new Bar(width)))
    source.map(_.in := in)
    out := sink.tail.map(_.out).reduce(_ & _)
    chisel3.assert(sink.head.out === 0.U)
  }

  class TopTester extends BasicTester {
    val dut = Module(new Top(4))
    val dut2 = Module(new Top(4))
    BoringUtils.bore(dut.source(1).x, Seq(dut.sink(1).x, dut.sink(2).x))

    val inVec = VecInit(Range(1, math.pow(2, dut.width).toInt).map(_.U))
    val (c, done) = Counter(true.B, inVec.size)
    dut.in := inVec(c)
    dut2.in := inVec(c)
    chisel3.assert(dut.out === dut.in)
    chisel3.assert(dut2.out === 0.U)

    when (done) { stop() }
  }

  it should "connect across modules via bore" in {
	  runTester(new TopTester) should be (true)
  }
}
