// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.FixedPoint
import chisel3.internal.firrtl.{BinaryPoint, Width}
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

  val fp   = FixedPoint.fromDouble(3.0, 4.W, 0.BP)
  val fp_tpe = FixedPoint(4.W, 1.BP)
  val uint_result = FixedPoint.fromDouble(1.5, 4.W, 1.BP)
  val sint_result = FixedPoint.fromDouble(-1.5, 4.W, 1.BP)
  val fp_result   = FixedPoint.fromDouble(1.5, 4.W, 1.BP)

  val uint2fp = fp_tpe.fromBits(uint)
  val sint2fp = fp_tpe.fromBits(sint)
  val fp2fp   = fp_tpe.fromBits(fp)

  val negativefp = (-3.5).F(4.BP)
  val positivefp = 3.5.F(4.BP)

  assert(uint2fp === uint_result)
  assert(sint2fp === sint_result)
  assert(fp2fp   === fp_result)

  assert(positivefp.abs() === positivefp)
  assert(negativefp.abs() === positivefp)
  assert(negativefp.abs() =/= negativefp)

  val f1p5 = 1.5.F(1.BP)
  val f6p0 = 6.0.F(0.BP)
  val f6p2 = 6.0.F(2.BP)

  val f1p5shiftleft2 = Wire(FixedPoint(Width(), BinaryPoint()))
  val f6p0shiftright2 = Wire(FixedPoint(Width(), BinaryPoint()))
  val f6p2shiftright2 = Wire(FixedPoint(Width(), BinaryPoint()))

  f1p5shiftleft2 := f1p5 << 2
  f6p0shiftright2 := f6p0 >> 2
  f6p2shiftright2 := f6p2 >> 2

  assert(f1p5shiftleft2 === f6p0)
  assert(f1p5shiftleft2 === 6.0.F(8.BP))

  printf("f6p2 %x f6p2shiftright2 %x\n", f6p2.asSInt(), f6p2shiftright2.asSInt())
  assert(f6p0shiftright2 === 1.0.F(0.BP))


  stop()
}

class SBP extends Module {
  val io = IO(new Bundle {
    val in =  Input(FixedPoint(6.W, 2.BP))
    val out = Output(FixedPoint(4.W, 0.BP))
  })
  io.out := io.in.setBinaryPoint(0)
}
class SBPTester extends BasicTester {
  val dut = Module(new SBP)
  dut.io.in := 3.75.F(2.BP)

  assert(dut.io.out === 3.0.F(0.BP))

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
