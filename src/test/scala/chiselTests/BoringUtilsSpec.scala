// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import chisel3.util.Counter
import chisel3.testers.BasicTester
import chisel3.experimental.{MultiIOModule, RawModule, BaseModule}
import chisel3.util.experimental.BoringUtils
import firrtl.{CommonOptions, ExecutionOptionsManager, HasFirrtlOptions, FirrtlExecutionOptions, FirrtlExecutionSuccess,
  FirrtlExecutionFailure}
import firrtl.passes.wiring.WiringTransform

abstract class ShouldntAssertTester(cyclesToWait: BigInt = 4) extends BasicTester {
  val dut: BaseModule
  val (_, done) = Counter(true.B, 2)
  when (done) { stop() }
}

class BoringUtilsSpec extends ChiselFlatSpec with ChiselRunners {

  class BoringInverter extends Module {
    val io = IO(new Bundle{})
    val a = Wire(UInt(1.W))
    val notA = Wire(UInt(1.W))
    val b = Wire(UInt(1.W))
    a := 0.U
    notA := ~a
    b := a
    chisel3.assert(b === 1.U)
    BoringUtils.addSource(notA, "x")
    BoringUtils.addSink(b, "x")
  }

  behavior of "BoringUtils.{addSink, addSource}"

  it should "connect two wires within a module" in {
    runTester(new ShouldntAssertTester { val dut = Module(new BoringInverter) } ) should be (true)
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

  // After boring, this will have the following connections:
  //   - source(0)   -> unconnected
  //   - unconnected -> expect(0)
  //   - source(1)   -> expect(1)
  //   - source(2)   -> expect(2)
  class Top(val width: Int) extends MultiIOModule {
    val source = Seq(0, 1, 2).map(x => x -> Constant(x)).toMap
    val expect = Map(0 -> Seq.fill(2)(Expect(0)),
                     1 -> Seq.fill(1)(Expect(1)),
                     2 -> Seq.fill(3)(Expect(2)))
  }

  class TopTester extends ShouldntAssertTester {
    val dut = Module(new Top(4))
    BoringUtils.bore(dut.source(1).x, dut.expect(1).map(_.x))
    BoringUtils.bore(dut.source(2).x, dut.expect(2).map(_.x))
  }

  behavior of "BoringUtils.bore"

  it should "connect across modules using BoringUtils.bore" in {
	  runTester(new TopTester) should be (true)
  }
}
