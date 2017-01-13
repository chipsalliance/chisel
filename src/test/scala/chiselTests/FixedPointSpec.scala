// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import org.scalatest._

//scalastyle:off magic.number
class FixedPointLiteralSpec extends FlatSpec with Matchers {
  behavior of "fixed point utilities"

  they should "allow conversion between doubles and the bigints needed to represent them" in {
    val initialDouble = 0.125
    val bigInt = FixedPoint.toBigInt(initialDouble, 4)
    val finalDouble = FixedPoint.toDouble(bigInt, 4)

    initialDouble should be(finalDouble)
  }
}

class FixedPointFromBitsTester extends BasicTester {
    val uint = 3.U(4.W)
    val sint = -3.S
    val fp   = FixedPoint.fromDouble(3.0, width = 4, binaryPoint = 0)
    val fp_tpe = FixedPoint(4.W, 1.BP)
    val uint_result = FixedPoint.fromDouble(1.5, width = 4, binaryPoint = 1)
    val sint_result = FixedPoint.fromDouble(-1.5, width = 4, binaryPoint = 1)
    val fp_result   = FixedPoint.fromDouble(1.5, width = 4, binaryPoint = 1)

    val uint2fp = fp_tpe.fromBits(uint)
    val sint2fp = fp_tpe.fromBits(sint)
    val fp2fp   = fp_tpe.fromBits(fp)

    assert(uint2fp === uint_result)
    assert(sint2fp === sint_result)
    assert(fp2fp   === fp_result)

    stop()
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

class FixedPointSpec extends ChiselPropSpec {
  property("should allow set binary point") {
    assertTesterPasses { new SBPTester }
  }
  property("should allow fromBits") {
    assertTesterPasses { new FixedPointFromBitsTester }
  }
}
