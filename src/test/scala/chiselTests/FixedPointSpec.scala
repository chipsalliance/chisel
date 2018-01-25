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

//noinspection TypeAnnotation,EmptyParenMethodAccessedAsParameterless
class FixedPointFromBitsTester extends BasicTester {
  val uint = 3.U(4.W)
  val sint = (-3).S

  val fp   = FixedPoint.fromDouble(3.0, 4.W, 0.BP)
  val fp_tpe = FixedPoint(4.W, 1.BP)
  val uint_result = FixedPoint.fromDouble(1.5, 4.W, 1.BP)
  val sint_result = FixedPoint.fromDouble(-1.5, 4.W, 1.BP)
  val fp_result   = FixedPoint.fromDouble(1.5, 4.W, 1.BP)

  val uint2fp = uint.asTypeOf(fp_tpe)
  val sint2fp = sint.asTypeOf(fp_tpe)
  val fp2fp   = fp.asTypeOf(fp_tpe)

  val uintToFp = uint.asFixedPoint(1.BP)
  val sintToFp = sint.asFixedPoint(1.BP)
  val fpToFp   = fp.asFixedPoint(1.BP)

  val negativefp = (-3.5).F(4.BP)
  val positivefp = 3.5.F(4.BP)

  assert(uint2fp === uint_result)
  assert(sint2fp === sint_result)
  assert(fp2fp   === fp_result)

  assert(uintToFp === uint_result)
  assert(sintToFp === sint_result)
  assert(fpToFp   === fp_result)

  assert(positivefp.abs() === positivefp)
  assert(negativefp.abs() === positivefp)
  assert(negativefp.abs() =/= negativefp)

  val f1bp5 = 1.5.F(1.BP)
  val f6bp0 = 6.0.F(0.BP)
  val f6bp2 = 6.0.F(2.BP)

  val f1bp5shiftleft2 = Wire(FixedPoint(Width(), BinaryPoint()))
  val f6bp0shiftright2 = Wire(FixedPoint(Width(), BinaryPoint()))
  val f6bp2shiftright2 = Wire(FixedPoint(Width(), BinaryPoint()))

  f1bp5shiftleft2 := f1bp5 << 2
  f6bp0shiftright2 := f6bp0 >> 2
  f6bp2shiftright2 := f6bp2 >> 2

  assert(f1bp5shiftleft2 === f6bp0)
  assert(f1bp5shiftleft2 === 6.0.F(8.BP))

  // shifting does not move binary point, so in first case below one bit is lost in shift
  assert(f6bp0shiftright2 === 1.0.F(0.BP))
  assert(f6bp2shiftright2 === 1.5.F(2.BP))


  stop()
}

class FixedPointMuxTester extends BasicTester {
  val largeWidthLowPrecision = 6.0.F(4.W, 0.BP)
  val smallWidthHighPrecision = 0.25.F(2.W, 2.BP)
  val unknownWidthLowPrecision = 6.0.F(0.BP)
  val unknownFixed = Wire(FixedPoint())
  unknownFixed := smallWidthHighPrecision
  
  assert(Mux(true.B, largeWidthLowPrecision, smallWidthHighPrecision) === 6.0.F(0.BP))
  assert(Mux(false.B, largeWidthLowPrecision, smallWidthHighPrecision) === 0.25.F(2.BP))
  assert(Mux(false.B, largeWidthLowPrecision, unknownFixed) === 0.25.F(2.BP))
  assert(Mux(true.B, unknownWidthLowPrecision, smallWidthHighPrecision) === 6.0.F(0.BP))
  
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

  val test = Wire(FixedPoint(10.W, 5.BP))
  // Initialize test, avoiding a "Reference test is not fully initialized" error from firrtl.
  test := 0.0.F(5.BP)
  val q = test.setBinaryPoint(18)
  assert(q.getWidth.U === 23.U)

  stop()
}

class FixedPointSpec extends ChiselPropSpec {
  property("should allow set binary point") {
    assertTesterPasses { new SBPTester }
  }
  property("should allow fromBits") {
    assertTesterPasses { new FixedPointFromBitsTester }
  }
  property("should mux different widths and binary points") {
    assertTesterPasses { new FixedPointMuxTester }
  }
  property("Negative shift amounts are invalid") {
    a [ChiselException] should be thrownBy { elaborate(new NegativeShift(FixedPoint(1.W, 0.BP))) }
  }
}
