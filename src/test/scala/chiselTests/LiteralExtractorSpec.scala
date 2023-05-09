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
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val a = Wire(UInt())
        assert(a.litOption == None)
      }
    }
  }

  "doubles and big decimals" should "round trip properly" in {
    val bigInt108 = BigInt("1" * 108, 2)
    val bigDecimal = Num.toBigDecimal(bigInt108, 2)

    val bigIntFromBigDecimal = Num.toBigInt(bigDecimal, 2)

    bigIntFromBigDecimal should be(bigInt108)

    val bigInt53 = BigInt("1" * 53, 2)

    val double = Num.toDouble(bigInt53, 2)

    val bigIntFromDouble = Num.toBigInt(double, 2)

    bigIntFromDouble should be(bigInt53)
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
