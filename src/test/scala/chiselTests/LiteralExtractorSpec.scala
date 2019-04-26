// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.core.FixedPoint
import chisel3.experimental.RawModule
import chisel3.experimental.BundleLiterals._
import chisel3.testers.BasicTester
import org.scalatest._

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
    elaborate { new RawModule {
      val a = Wire(UInt())
      assert(a.litOption == None)
    }}
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
      // chisel3.core.assert(outsideLiteral === insideLiteral)

      // the following lines of code error
      // with "chisel3.core.BundleLitBinding cannot be cast to chisel3.core.ElementLitBinding"

      chisel3.core.assert(outsideLiteral.x === insideLiteral.x)
      chisel3.core.assert(outsideLiteral.y === insideLiteral.y)
      chisel3.core.assert(outsideLiteral.x === 7.S)
      chisel3.core.assert(outsideLiteral.y === 6.125.F(4.BP))

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
}
