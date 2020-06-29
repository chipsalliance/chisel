// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental._
import chisel3.experimental.BundleLiterals._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import scala.collection.immutable.ListMap

class LiteralExtractorSpec extends ChiselFlatSpec {
  "litValue" should "return the literal value" in {
    assert(0.U.litValue === BigInt(0))
    assert(1.U.litValue === BigInt(1))
    assert(42.U.litValue === BigInt(42))
    assert(42.U.litValue === 42.U.litValue)

    assert(0.S.litValue === BigInt(0))
    assert(-1.S.litValue === BigInt(-1))
    assert(-42.S.litValue === BigInt(-42))

    assert(true.B.litValue === BigInt(1))
    assert(false.B.litValue === BigInt(0))

    assert(1.25.F(2.BP).litValue === BigInt("101", 2))
    assert(2.25.F(2.BP).litValue === BigInt("1001", 2))

    assert(-1.25.F(2.BP).litValue === BigInt("-101", 2))
    assert(-2.25.F(2.BP).litValue === BigInt("-1001", 2))
  }

  "litToBoolean" should "return the literal value" in {
    assert(true.B.litToBoolean === true)
    assert(false.B.litToBoolean === false)

    assert(1.B.litToBoolean === true)
    assert(0.B.litToBoolean === false)
  }

  "litToDouble" should "return the literal value" in {
    assert(1.25.F(2.BP).litToDouble == 1.25)
    assert(2.25.F(2.BP).litToDouble == 2.25)

    assert(-1.25.F(2.BP).litToDouble == -1.25)
    assert(-2.25.F(2.BP).litToDouble == -2.25)

    // test rounding
    assert(1.24.F(1.BP).litToDouble == 1.0)
    assert(1.25.F(1.BP).litToDouble == 1.5)
  }

  "litOption" should "return None for non-literal hardware" in {
    ChiselStage.elaborate { new RawModule {
      val a = Wire(UInt())
      assert(a.litOption == None)
    }}
  }

  "doubles and big decimals" should "round trip properly" in {
    intercept[ChiselException] {
      Num.toBigDecimal(BigInt("1" * 109, 2), 0.BP)  // this only works if number takes less than 109 bits
    }

    intercept[ChiselException] {
      Num.toDouble(BigInt("1" * 54, 2), 0.BP)  // this only works if number takes less than 54 bits
    }

    val bigInt108 = BigInt("1" * 108, 2)
    val bigDecimal = Num.toBigDecimal(bigInt108, 2)

    val bigIntFromBigDecimal = Num.toBigInt(bigDecimal, 2)

    bigIntFromBigDecimal should be (bigInt108)

    val bigInt53 = BigInt("1" * 53, 2)

    val double  = Num.toDouble(bigInt53, 2)

    val bigIntFromDouble = Num.toBigInt(double, 2)

    bigIntFromDouble should be (bigInt53)
  }

  "encoding and decoding of Intervals" should "round trip" in {
    val rangeMin = BigDecimal(-31.5)
    val rangeMax = BigDecimal(32.5)
    val range = range"($rangeMin, $rangeMax).2"
    for(value <- (rangeMin - 4) to (rangeMax + 4) by 2.25) {
      if (value < rangeMin || value > rangeMax) {
        intercept[ChiselException] {
          val literal = value.I(range)
        }
      } else {
        val literal = value.I(range)
        literal.isLit() should be(true)
        literal.litValue().toDouble / 4.0 should be(value)
      }
    }
  }

  "literals declared outside a builder context" should "compare with those inside builder context" in {
    class InsideBundle extends Bundle {
      val x = SInt(8.W)
      val y = FixedPoint(8.W, 4.BP)
    }

    class LitInsideOutsideTester(outsideLiteral: InsideBundle) extends BasicTester {
      val insideLiteral = (new InsideBundle).Lit(_.x -> 7.S, _.y -> 6.125.F(4.BP))

      // the following errors with "assertion failed"

      println(outsideLiteral === insideLiteral) // scalastyle:ignore regex
      // chisel3.assert(outsideLiteral === insideLiteral)

      // the following lines of code error
      // with "chisel3.internal.BundleLitBinding cannot be cast to chisel3.internal.ElementLitBinding"

      chisel3.assert(outsideLiteral.x === insideLiteral.x)
      chisel3.assert(outsideLiteral.y === insideLiteral.y)
      chisel3.assert(outsideLiteral.x === 7.S)
      chisel3.assert(outsideLiteral.y === 6.125.F(4.BP))

      stop()
    }

    val outsideLiteral = (new InsideBundle).Lit(_.x -> 7.S, _.y -> 6.125.F(4.BP))
    assertTesterPasses{ new LitInsideOutsideTester(outsideLiteral) }

  }


  "bundle literals" should "do the right thing" in {
    class MyBundle extends Bundle {
      val a = UInt(8.W)
      val b = Bool()
    }
    val myBundleLiteral = (new MyBundle).Lit(_.a -> 42.U, _.b -> true.B)
    assert(myBundleLiteral.a.litValue == 42)
    assert(myBundleLiteral.b.litValue == 1)
    assert(myBundleLiteral.b.litToBoolean == true)
  }

  "record literals" should "do the right thing" in {
    class MyRecord extends Record{
      override val elements = ListMap(
        "a" -> UInt(8.W),
        "b" -> Bool()
      )
      override def cloneType = (new MyRecord).asInstanceOf[this.type]
    }

    val myRecordLiteral = new (MyRecord).Lit(_.elements("a") -> 42.U, _.elements("b") -> true.B)
    assert(myRecordLiteral.elements("a").litValue == 42)
    assert(myRecordLiteral.elements("b").litValue == 1)
  }
}
