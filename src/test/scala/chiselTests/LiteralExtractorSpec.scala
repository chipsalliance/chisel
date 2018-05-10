// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.RawModule
import org.scalatest._

class LiteralExtractorSpec extends ChiselFlatSpec {
  "litToBigInt" should "return the literal value" in {
    assert(0.U.litToBigInt === BigInt(0))
    assert(1.U.litToBigInt === BigInt(1))
    assert(42.U.litToBigInt === BigInt(42))

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

  "bundle literals" should "do the right thing" in {
    class MyBundle extends Bundle {
      val a = UInt(8.W)
      val b = Bool()

      // Bundle literal constructor code, which will be auto-generated using macro annotations in
      // the future.
      import chisel3.core.BundleLitBinding
      import chisel3.internal.firrtl.{ULit, Width}
      def Lit(aVal: BigInt, bVal: Boolean): MyBundle = {
        val clone = cloneType
        clone.selfBind(BundleLitBinding(Map(
            clone.a -> ULit(aVal, Width()),
            clone.b -> ULit(if (bVal) 1 else 0, Width(1)))
        ))
        clone
      }
    }
    val myBundleLiteral = (new MyBundle).Lit(42, true)
    println(myBundleLiteral.a)
    println(myBundleLiteral.b)
    println(myBundleLiteral.a == myBundleLiteral.b)
    assert(myBundleLiteral.a.litToBigInt == 42)
    assert(myBundleLiteral.b.litToBigInt == 1)
    assert(myBundleLiteral.b.litToBoolean == true)
  }
}
