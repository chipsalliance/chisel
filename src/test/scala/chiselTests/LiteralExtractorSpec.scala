// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.BundleLiterals._
import chisel3.experimental._
import chisel3.testers.BasicTester
import circt.stage.ChiselStage

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
  }

  "litToBoolean" should "return the literal value" in {
    assert(true.B.litToBoolean === true)
    assert(false.B.litToBoolean === false)

    assert(1.B.litToBoolean === true)
    assert(0.B.litToBoolean === false)
  }

  "litOption" should "return None for non-literal hardware" in {
    ChiselStage.elaborate {
      new RawModule {
        val a = Wire(UInt())
        assert(a.litOption == None)
      }
    }
  }

  "doubles and big decimals" should "round trip properly" in {
    intercept[ChiselException] {
      Num.toBigDecimal(BigInt("1" * 109, 2), 0.BP) // this only works if number takes less than 109 bits
    }

    intercept[ChiselException] {
      Num.toDouble(BigInt("1" * 54, 2), 0.BP) // this only works if number takes less than 54 bits
    }

    val bigInt108 = BigInt("1" * 108, 2)
    val bigDecimal = Num.toBigDecimal(bigInt108, 2)

    val bigIntFromBigDecimal = Num.toBigInt(bigDecimal, 2)

    bigIntFromBigDecimal should be(bigInt108)

    val bigInt53 = BigInt("1" * 53, 2)

    val double = Num.toDouble(bigInt53, 2)

    val bigIntFromDouble = Num.toBigInt(double, 2)

    bigIntFromDouble should be(bigInt53)
  }

  "encoding and decoding of Intervals" should "round trip" in {
    val rangeMin = BigDecimal(-31.5)
    val rangeMax = BigDecimal(32.5)
    val range = range"($rangeMin, $rangeMax).2"
    for (value <- (rangeMin - 4) to (rangeMax + 4) by 2.25) {
      if (value < rangeMin || value > rangeMax) {
        intercept[ChiselException] {
          val literal = value.I(range)
        }
      } else {
        val literal = value.I(range)
        literal.isLit should be(true)
        literal.litValue.toDouble / 4.0 should be(value)
      }
    }
  }

  "literals declared outside a builder context" should "compare with those inside builder context" in {
    class InsideBundle extends Bundle {
      val x = SInt(8.W)
      val y = UInt(88.W)
    }

    class LitInsideOutsideTester(outsideLiteral: InsideBundle) extends BasicTester {
      val insideLiteral = (new InsideBundle).Lit(_.x -> 7.S, _.y -> 7777.U)

      // the following errors with "assertion failed"

      println(outsideLiteral === insideLiteral)
      // chisel3.assert(outsideLiteral === insideLiteral)

      // the following lines of code error
      // with "chisel3.internal.BundleLitBinding cannot be cast to chisel3.internal.ElementLitBinding"

      chisel3.assert(outsideLiteral.x === insideLiteral.x)
      chisel3.assert(outsideLiteral.y === insideLiteral.y)
      chisel3.assert(outsideLiteral.x === 7.S)
      chisel3.assert(outsideLiteral.y === 7777.U)

      stop()
    }

    val outsideLiteral = (new InsideBundle).Lit(_.x -> 7.S, _.y -> 7777.U)
    assertTesterPasses { new LitInsideOutsideTester(outsideLiteral) }

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
    class MyRecord extends Record {
      override val elements = ListMap(
        "a" -> UInt(8.W),
        "b" -> Bool()
      )
    }

    val myRecordLiteral = new (MyRecord).Lit(_.elements("a") -> 42.U, _.elements("b") -> true.B)
    assert(myRecordLiteral.elements("a").litValue == 42)
    assert(myRecordLiteral.elements("b").litValue == 1)
  }
}
