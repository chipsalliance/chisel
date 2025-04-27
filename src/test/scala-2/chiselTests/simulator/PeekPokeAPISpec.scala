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

  val numTests = 50

  describe("PeekPokeAPI with TestableData") {
    val w = 32
    it("should correctly poke, peek, and peekValue Elements and Aggregates") {
      simulate(new PeekPokeTestModule(w)) { dut =>
        assert(w == dut.io.in.bits.a.getWidth)
        val vecDim = dut.vecDim
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
          dut.io.in.valid.poke(0.B)

          val peekedOp = dut.io.op.peek()
          assert(peekedOp.litValue == op.litValue)
          assert(peekedOp.toString.startsWith(TestOp.getClass.getSimpleName.stripSuffix("$")))

          val expectedScalar = calcExpectedScalarOpResult(op, a, b, w)
          val expectedCmp = calcExpectedCmp(a, b)
          val expectedVSum = calcExpectedVSum(v1, v2, w)
          val expVecProduct = calcExpectedVecProduct(v1, v2, w)
          val expectedVDot = calcExpectedVDot(v1, v2, w)

          val expectedBits = chiselTypeOf(dut.io.out.bits).Lit(
            _.c -> expectedScalar.U,
            _.cmp -> expectedCmp,
            _.vSum -> expectedVSum,
            _.vDot -> expectedVDot,
            _.vOutProduct -> expVecProduct
          )
          assert(dut.io.out.bits.cmp.peekValue().asBigInt == expectedCmp.litValue)
          assert(dut.io.out.bits.cmp.peek() == expectedCmp)
          assert(dut.io.out.bits.vSum.zip(expectedVSum).forall { case (portEl, expEl) =>
            portEl.peekValue().asBigInt == expEl.litValue
          })
          assert(dut.io.out.bits.vSum.peek().litValue == expectedVSum.litValue)
          assert(dut.io.out.bits.vOutProduct.zip(expVecProduct).forall { case (portVecEl, expVecEl) =>
            portVecEl.zip(expVecEl).forall { case (portEl, expEl) =>
              portEl.peek().litValue == expEl.litValue
            }
          })

          val peekedBits = dut.io.out.bits.peek()
          assert(peekedBits.c.litValue == expectedBits.c.litValue)
          assert(peekedBits.cmp.litValue == expectedBits.cmp.litValue)
          assert(peekedBits.elements.forall { case (name, el) => expectedBits.elements(name).litValue == el.litValue })
        }
      }
    }

    it("should expect various data types correctly") {
      simulate(new PeekPokeTestModule(w)) { dut =>
        assert(w == dut.io.in.bits.a.getWidth)
        val vecDim = dut.vecDim
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
          dut.io.in.valid.poke(true.B)
          dut.io.op.poke(op)
          dut.clock.step()
          dut.io.in.valid.poke(false.B)

          val expectedScalar = calcExpectedScalarOpResult(op, a, b, w)
          val expectedCmp = calcExpectedCmp(a, b)

          dut.io.out.valid.expect(true.B)
          dut.io.out.valid.expect(true)
          dut.io.out.valid.expect(1)
          dut.io.out.bits.c.expect(expectedScalar)
          dut.io.out.bits.c.expect(expectedScalar.U)
          dut.io.out.bits.c.expect(expectedScalar.U(w.W))
          dut.io.out.bits.cmp.expect(expectedCmp)

          val expectedVSum = calcExpectedVSum(v1, v2, w)
          dut.io.out.bits.vSum.expect(expectedVSum)

          val expVecProduct = calcExpectedVecProduct(v1, v2, w)
          val expVDot = calcExpectedVDot(v1, v2, w)

          dut.io.out.bits.vOutProduct.expect(expVecProduct)

          val expectedBits = chiselTypeOf(dut.io.out.bits).Lit(
            _.c -> expectedScalar.U,
            _.cmp -> expectedCmp,
            _.vSum -> expectedVSum,
            _.vDot -> expVDot,
            _.vOutProduct -> expVecProduct
          )

          dut.io.out.bits.expect(expectedBits)

          dut.io.out.expect(
            chiselTypeOf(dut.io.out).Lit(
              _.valid -> true.B,
              _.bits -> expectedBits
            )
          )
        }
      }
    }

    it("should report failed expects correctly") {
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

          dut.io.out.bits.c.expect(3) // correct
          dut.io.out.bits.c.expect(5.U) // incorrect
        }
      }
      thrown.getMessage must include("dut.io.out.bits.c.expect(5.U)")
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
            _.vDot -> 35.U,
            _.vOutProduct -> Vec.Lit(
              Seq.fill(vecDim)(Vec.Lit(Seq.tabulate(vecDim)(i => (12 + i).U((2 * w).W)): _*)): _*
            )
          )
          dut.io.out.bits.expect(expectedBits)
        }
      }
      thrown.getMessage must include("dut.io.out.bits.expect(expectedBits)")
      thrown.getMessage must include("Observed value: 'UInt<66>(36)")
      thrown.getMessage must include("Expected value: 'UInt<66>(35)")
      thrown.getMessage must include("dut.io.out.bits.expect(expectedBits)")
      thrown.getMessage must include("Expected the value of element 'vDot' to be ")
    }

    it("should report failed expect of Records with Vec fields correctly") {
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

          val expRecord = chiselTypeOf(dut.io.out.bits).Lit(
            _.c -> 3.U,
            _.cmp -> CmpResult.LT,
            _.vSum -> Vec.Lit(Seq.fill(vecDim)(7.U((w + 1).W)): _*),
            _.vDot -> 36.U((2 * w + vecDim - 1).W),
            _.vOutProduct -> Vec.Lit(
              Seq.fill(vecDim)(Vec.Lit(Seq.tabulate(vecDim)(i => (12 + i).U((2 * w).W)): _*)): _*
            )
          )
          dut.io.out.bits.expect(expRecord)
        }
      }
      thrown.getMessage must include("dut.io.out.bits.expect(expRecord)")
      thrown.getMessage must include("Expected the value of element 'vOutProduct' to be ")
    }
  }
}
