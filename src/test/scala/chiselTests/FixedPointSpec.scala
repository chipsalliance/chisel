// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import org.scalatest._

//scalastyle:off magic.number
class FixedPointSpec extends FlatSpec with Matchers {
  behavior of "fixed point utilities"

  they should "allow conversion between doubles and the bigints needed to represent them" in {
    val initialDouble = 0.125
    val bigInt = FixedPoint.toBigInt(initialDouble, 4)
    val finalDouble = FixedPoint.toDouble(bigInt, 4)

    initialDouble should be(finalDouble)
  }

  behavior of "fixed point"

  they should "allow fromBits from now fixed point types" in {
    val uint = 3.U
    val sint = -3.S
    val fp_tpe = FixedPoint(width = 4, binaryPoint = 1)
    val uint_result = FixedPoint.fromDouble(1.5, width = 4, binaryPoint = 1)
    val sint_result = FixedPoint.fromDouble(-1.5, width = 4, binaryPoint = 1)

    val uint2fp = fp_tpe.fromBits(uint)
    val sint2fp = fp_tpe.fromBits(sint)

    uint2fp should be (uint_result)
    sint2fp should be (sint_result)
  }
}

class SBP extends Module {
  val io = IO(new Bundle {
    val in =  Input(FixedPoint(6, 2))
    val out = Output(FixedPoint(4, 0))
  })
  io.out := io.in.setBinaryPoint(0)
}
class SBPTester extends BasicTester {
  val dut = Module(new SBP)
  dut.io.in := FixedPoint.fromDouble(3.75, binaryPoint = 2)

  assert(dut.io.out === FixedPoint.fromDouble(3.0, binaryPoint = 0))

  stop()
}
class SBPSpec extends ChiselPropSpec {
  property("should allow set binary point") {
    assertTesterPasses { new SBPTester }
  }
}
