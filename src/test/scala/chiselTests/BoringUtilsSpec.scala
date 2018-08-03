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
    val x = Wire(UInt(4.W))
  }

  class Constant(const: Int) extends MultiIOModule with WireX {
    x := const.U
  }

  object Constant { def apply(const: Int): Constant = Module(new Constant(const)) }

  class Expect(const: Int) extends MultiIOModule with WireX {
    x := 0.U // Default value. Output is zero unless we bore...
    chisel3.assert(x === const.U, "x (0x%x) was not const.U (0x%x)", x, const.U)
  }

  object Expect { def apply(const: Int): Expect = Module(new Expect(const)) }

  class Top(val width: Int) extends MultiIOModule {
    /* source(0)   -> unconnected
     * unconnected -> Seq(expect(0))
     * source(1)   -> expect(1)
     * source(2)   -> expect(2) */
    val source = Seq(0, 1, 2).map(x => x -> Constant(x)).toMap
    val expect = Map(0 -> Seq.fill(2)(Expect(0)),
                     1 -> Seq.fill(1)(Expect(1)),
                     2 -> Seq.fill(3)(Expect(2)))
  }

  class TopTester extends BasicTester {
    val dut = Module(new Top(4))
    BoringUtils.bore(dut.source(1).x, dut.expect(1).map(_.x))
    BoringUtils.bore(dut.source(2).x, dut.expect(2).map(_.x))

    val (_, done) = Counter(true.B, 4)
    when (done) { stop() }
  }

  it should "connect across modules via bore" in {
	  runTester(new TopTester) should be (true)
  }
}
