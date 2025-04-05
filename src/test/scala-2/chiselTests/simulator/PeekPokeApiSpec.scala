package chiselTests.simulator

import chisel3._
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.VecLiterals._
import chisel3.util._
import chisel3.simulator._
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

object TestOp extends ChiselEnum {
  val Add, Sub, Mul = Value
}

class TestPeekPokeEnum(w: Int) extends Module {
  object CmpResult extends ChiselEnum {
    val LT, EQ, GT = Value
  }

  val vecDim = 3

  val io = IO(new Bundle {
    val in = Input(Valid(new Bundle {
      val a = UInt(w.W)
      val b = UInt(w.W)
      val v1 = Vec(vecDim, UInt(w.W))
      val v2 = Vec(vecDim, UInt(w.W))
    }))
    val op = Input(TestOp())
    val out = Valid(new Bundle {
      val c = UInt(w.W)
      val cmp = CmpResult()
      val vSum = Vec(vecDim, UInt((w + 1).W))
      val vOutProduct = Vec(vecDim, Vec(vecDim, UInt((2 * w).W)))
      val vDot = UInt((2 * w + vecDim - 1).W)
    })
  })

  val a = io.in.bits.a
  val b = io.in.bits.b

  val result = Wire(chiselTypeOf(io.out.bits))

  result.c :#= MuxCase(
    0.U,
    Seq(
      (io.op === TestOp.Add) -> (a + b),
      (io.op === TestOp.Sub) -> (a - b),
      (io.op === TestOp.Mul) -> (a * b).take(w)
    )
  )

  // Supress the following warning:
  //   [W001] Casting non-literal UInt to chiselTests.simulator.TestPeekPokeEnum$CmpResult.
  //   You can use chiselTests.simulator.TestPeekPokeEnum$CmpResult.safe to cast without this warning.
  // The warning seems to be a (unrelated) bug
  suppressEnumCastWarning {
    result.cmp :#= Mux1H(
      Seq(
        (a < b) -> CmpResult.LT,
        (a === b) -> CmpResult.EQ,
        (a > b) -> CmpResult.GT
      )
    )
  }

  // addition of vectors
  result.vSum :#= io.in.bits.v1.zip(io.in.bits.v2).map { case (x, y) => x +& y }
  // inner product
  result.vDot :#= io.in.bits.v1.zip(io.in.bits.v2).map { case (x, y) => x * y }.reduce(_ +& _)
  // outer product
  result.vOutProduct :#= io.in.bits.v1.map { x => VecInit(io.in.bits.v2.map { y => x * y }) }

  io.out :#= Pipe(io.in.valid, result)
}

class PeekPokeAPISpec extends AnyFunSpec with ChiselSim with Matchers {
  val rand = scala.util.Random

  describe("PeekPokeAPI with testableData") {
    val w = 32
    it("should peek and poke various data types correctly") {
      val numTests = 100
      simulate(new TestPeekPokeEnum(w)) { dut =>
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
            case -1 => dut.CmpResult.LT
            case 0  => dut.CmpResult.EQ
            case 1  => dut.CmpResult.GT
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
      val thrown = the[PeekPokeAPI.FailedExpectationException[_]] thrownBy {
        simulate(new TestPeekPokeEnum(w)) { dut =>
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
      (thrown.getMessage must include).regex(
        """ @\[.*chiselTests/simulator/PeekPokeApiSpec\.scala:\d+:\d+\]"""
      )
      thrown.getMessage must include("dut.io.out.bits.c.expect(5.U)")
      thrown.getMessage must include("                        ^")
    }

    it("reports failed Record expects correctly") {
      val thrown = the[PeekPokeAPI.FailedExpectationException[_]] thrownBy {
        simulate(new TestPeekPokeEnum(w)) { dut =>
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
            _.cmp -> dut.CmpResult.LT,
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
      (thrown.getMessage must include).regex(
        """ @\[.*chiselTests/simulator/PeekPokeApiSpec\.scala:\d+:\d+\]"""
      )
      thrown.getMessage must include("dut.io.out.bits.expect(expectedBits)")
      thrown.getMessage must include("                      ^")
    }

    it("reports failed expect of Records with Vec fields correctly") {
      val thrown = the[PeekPokeAPI.FailedExpectationException[_]] thrownBy {
        simulate(new TestPeekPokeEnum(w)) { dut =>
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
              _.cmp -> dut.CmpResult.LT,
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
      (thrown.getMessage must include).regex(
        """ @\[.*chiselTests/simulator/PeekPokeApiSpec\.scala:\d+:\d+\]"""
      )
      thrown.getMessage must include("dut.io.out.bits.expect(")
      thrown.getMessage must include("                      ^")
    }
  }
}
