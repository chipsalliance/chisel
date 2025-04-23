package chiselTests.simulator

import chisel3._
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.VecLiterals._
import chisel3.util._
import chisel3.simulator._
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

class PeekPokeAPISpec extends AnyFunSpec with ChiselSim with Matchers {
  val rand = scala.util.Random

  import PeekPokeTestModule._

  describe("PeekPokeAPI with TestableData") {
    val w = 32
    it("should peek and poke various data types correctly") {
      val numTests = 100
      simulate(new PeekPokeTestModule(w)) { dut =>
        assert(w == dut.io.in.bits.a.getWidth)
        val vecDim = dut.vecDim
        val truncationMask = (BigInt(1) << w) - 1
        for {
          _ <- 0 until numTests
          a = BigInt(w, rand)
          b = BigInt(w, rand)
          v1 = Seq.fill(vecDim)(BigInt(w, rand))
          v2 = Seq.fill(vecDim)(BigInt(w, rand))
          op <- TestOp.all
        } {

          dut.io.in.bits.poke(
            chiselTypeOf(dut.io.in.bits).Lit(
              _.a -> a.U,
              _.b -> b.U,
              _.v1 -> Vec.Lit(v1.map(_.U(w.W)): _*),
              _.v2 -> Vec.Lit(v2.map(_.U(w.W)): _*)
            )
          )
          dut.io.in.valid.poke(true)
          dut.io.op.poke(op)
          dut.clock.step()
          dut.io.in.valid.poke(false)

          val peekedOp = dut.io.op.peek()
          assert(peekedOp.litValue == op.litValue)
          assert(peekedOp.toString.contains(TestOp.getClass.getSimpleName.stripSuffix("$")))

          val expected = op match {
            case TestOp.Add => a + b
            case TestOp.Sub => a - b
            case TestOp.Mul => a * b
            case _          => throw new Exception("Invalid operation")
          }
          val expectedCmp = a.compare(b) match {
            case -1 => CmpResult.LT
            case 0  => CmpResult.EQ
            case 1  => CmpResult.GT
          }

          dut.io.out.valid.expect(true.B)
          dut.io.out.valid.expect(true)
          dut.io.out.bits.c.expect(expected & truncationMask)

          assert(dut.io.out.bits.cmp.peek().litValue == expectedCmp.litValue)
          dut.io.out.bits.cmp.expect(expectedCmp)

          val expectedVSum = Vec.Lit(v1.zip(v2).map { case (x, y) => (x + y).U((w + 1).W) }: _*)

          dut.io.out.bits.vSum.expect(expectedVSum)

          val expVOutProduct = Vec.Lit(
            v1.map { x =>
              Vec.Lit(v2.map { y => (x * y).U((2 * w).W) }: _*)
            }: _*
          )

          dut.io.out.bits.vOutProduct.expect(expVOutProduct)

          val expectedBits = chiselTypeOf(dut.io.out.bits).Lit(
            _.c -> (expected & truncationMask).U,
            _.cmp -> expectedCmp,
            _.vSum -> expectedVSum,
            _.vDot -> v1.zip(v2).map { case (x, y) => x * y }.reduce(_ + _).U((2 * w + vecDim - 1).W),
            _.vOutProduct -> expVOutProduct
          )

          dut.io.out.bits.expect(expectedBits)

          val peekedBits = dut.io.out.bits.peek()
          assert(peekedBits.c.litValue == expectedBits.c.litValue)
          assert(peekedBits.cmp.litValue == expectedBits.cmp.litValue)

          assert(peekedBits.elements.forall { case (name, el) => expectedBits.elements(name).litValue == el.litValue })

          dut.io.out.expect(
            chiselTypeOf(dut.io.out).Lit(
              _.valid -> true.B,
              _.bits -> expectedBits
            )
          )
        }
      }
    }

    it("reports failed expects correctly") {
      val thrown = the[FailedExpectationException[_]] thrownBy {
        simulate(new PeekPokeTestModule(w)) { dut =>
          assert(w == dut.io.in.bits.a.getWidth)
          val vecDim = dut.vecDim
          val truncationMask = (BigInt(1) << w) - 1

          dut.io.in.bits.poke(
            chiselTypeOf(dut.io.in.bits).Lit(
              _.a -> 1.U,
              _.b -> 2.U,
              _.v1 -> Vec.Lit(Seq.fill(vecDim)(3.U(w.W)): _*),
              _.v2 -> Vec.Lit(Seq.fill(vecDim)(4.U(w.W)): _*)
            )
          )
          dut.io.in.valid.poke(true)
          dut.io.op.poke(TestOp.Add)
          dut.clock.step()

          dut.io.out.bits.c.expect(5.U)
        }
      }
      thrown.getMessage must include("observed value UInt<32>(3) != UInt<3>(5)")
    }

    it("reports failed Record expects correctly") {
      val thrown = the[FailedExpectationException[_]] thrownBy {
        simulate(new PeekPokeTestModule(w)) { dut =>
          assert(w == dut.io.in.bits.a.getWidth)
          val vecDim = dut.vecDim
          val truncationMask = (BigInt(1) << w) - 1

          dut.io.in.bits.poke(
            chiselTypeOf(dut.io.in.bits).Lit(
              _.a -> 1.U,
              _.b -> 2.U,
              _.v1 -> Vec.Lit(Seq.fill(vecDim)(3.U(w.W)): _*),
              _.v2 -> Vec.Lit(Seq.fill(vecDim)(4.U(w.W)): _*)
            )
          )
          dut.io.in.valid.poke(true)
          dut.io.op.poke(TestOp.Add)
          dut.clock.step()

          val expectedBits = chiselTypeOf(dut.io.out.bits).Lit(
            _.c -> 3.U,
            _.cmp -> CmpResult.LT,
            _.vSum -> Vec.Lit(Seq.fill(vecDim)(7.U((w + 1).W)): _*),
            _.vDot -> 35.U((2 * w + vecDim - 1).W),
            _.vOutProduct -> Vec.Lit(
              Seq.fill(vecDim)(Vec.Lit(Seq.tabulate(vecDim)(i => (12 + i).U((2 * w).W)): _*)): _*
            )
          )
          dut.io.out.bits.expect(expectedBits)
        }
      }
      thrown.getMessage must include("Observed value: 'UInt<66>(36)")
      thrown.getMessage must include("Expected value: 'UInt<66>(35)")
      thrown.getMessage must include("Expectation failed: observed value AnonymousBundle")
    }

    it("reports failed expect of Records with Vec fields correctly") {
      val thrown = the[FailedExpectationException[_]] thrownBy {
        simulate(new PeekPokeTestModule(w)) { dut =>
          assert(w == dut.io.in.bits.a.getWidth)
          val vecDim = dut.vecDim
          val truncationMask = (BigInt(1) << w) - 1

          dut.io.in.bits.poke(
            chiselTypeOf(dut.io.in.bits).Lit(
              _.a -> 1.U,
              _.b -> 2.U,
              _.v1 -> Vec.Lit(Seq.fill(vecDim)(3.U(w.W)): _*),
              _.v2 -> Vec.Lit(Seq.fill(vecDim)(4.U(w.W)): _*)
            )
          )
          dut.io.in.valid.poke(true)
          dut.io.op.poke(TestOp.Add)
          dut.clock.step()

          dut.io.out.bits.expect(
            chiselTypeOf(dut.io.out.bits).Lit(
              _.c -> 3.U,
              _.cmp -> CmpResult.LT,
              _.vSum -> Vec.Lit(Seq.fill(vecDim)(7.U((w + 1).W)): _*),
              _.vDot -> 36.U((2 * w + vecDim - 1).W),
              _.vOutProduct -> Vec.Lit(
                Seq.fill(vecDim)(Vec.Lit(Seq.tabulate(vecDim)(i => (12 + i).U((2 * w).W)): _*)): _*
              )
            )
          )
        }
      }
      thrown.getMessage must include("Expectation failed: observed value")
    }
  }
}
