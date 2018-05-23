// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.core.FixedPoint
import chisel3.experimental.RawModule
import chisel3.testers.BasicTester
import chisel3.util.Counter
import org.scalatest._

class LiteralExtractorSpec extends ChiselFlatSpec {
  "litToBigInt" should "return the literal value" in {
    assert(0.U.litToBigInt === BigInt(0))
    assert(1.U.litToBigInt === BigInt(1))
    assert(42.U.litToBigInt === BigInt(42))
    assert(42.U.litToBigInt === 42.U.litToBigInt)

    assert(0.S.litToBigInt === BigInt(0))
    assert(-1.S.litToBigInt === BigInt(-1))
    assert(-42.S.litToBigInt === BigInt(-42))

    assert(true.B.litToBigInt === BigInt(1))
    assert(false.B.litToBigInt === BigInt(0))

    assert(1.25.F(2.BP).litToBigInt === BigInt("101", 2))
    assert(2.25.F(2.BP).litToBigInt === BigInt("1001", 2))

    assert(-1.25.F(2.BP).litToBigInt === BigInt("-101", 2))
    assert(-2.25.F(2.BP).litToBigInt === BigInt("-1001", 2))
  }

  "litToBoolean" should "return the literal value" in {
    assert(true.B.litToBoolean === true)
    assert(false.B.litToBoolean === false)
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

  "litToBigIntOption" should "return None for non-literal hardware" in {
    elaborate { new RawModule {
      val a = Wire(UInt())
      assert(a.litToBigIntOption == None)
    }}
  }


  "literals declared outside a builder context" should "compare with those inside builder context" in {
    class InsideBundle extends Bundle {
      val x = SInt(8.W)
      val y = FixedPoint(8.W, 4.BP)

      import chisel3.core.BundleLitBinding
      def Lit(aVal: SInt, bVal: FixedPoint): InsideBundle = {
        val clone = cloneType
        clone.selfBind(BundleLitBinding(Map(
          clone.x -> aVal.elementLitArg.get,
          clone.y -> bVal.elementLitArg.get
        )))
        clone
      }
    }

    class LitInsideOutsideTester(outsideLiteral: InsideBundle) extends BasicTester {
      val insideLiteral = (new InsideBundle).Lit(7.S, 6.125.F(4.BP))

      // the following errors with "assertion failed"

      chisel3.core.assert(outsideLiteral === insideLiteral)

      // the following lines of code error
      // with "chisel3.core.BundleLitBinding cannot be cast to chisel3.core.ElementLitBinding"

      chisel3.core.assert(outsideLiteral.x === insideLiteral.x)
      chisel3.core.assert(outsideLiteral.y === insideLiteral.y)
      chisel3.core.assert(outsideLiteral.x === 7.S)
      chisel3.core.assert(outsideLiteral.y === 6.125.F(4.BP))

      stop()
    }

    val outsideLiteral = (new InsideBundle).Lit(7.S, 6.125.F(4.BP))
    assertTesterPasses{ new LitInsideOutsideTester(outsideLiteral) }

  }


  "bundle literals" should "do the right thing" in {
    class MyBundle extends Bundle {
      val a = UInt(8.W)
      val b = Bool()

      // Bundle literal constructor code, which will be auto-generated using macro annotations in
      // the future.
      import chisel3.core.BundleLitBinding
      import chisel3.internal.firrtl.{ULit, Width}
      def Lit(aVal: UInt, bVal: Bool): MyBundle = {
        val clone = cloneType
        clone.selfBind(BundleLitBinding(Map(
          clone.a -> aVal.elementLitArg.get,
          clone.b -> bVal.elementLitArg.get
        )))
        clone
      }
    }
    val myBundleLiteral = (new MyBundle).Lit(42.U, true.B)
    assert(myBundleLiteral.a.litToBigInt == 42)
    assert(myBundleLiteral.b.litToBigInt == 1)
    assert(myBundleLiteral.b.litToBoolean == true)
  }
}
