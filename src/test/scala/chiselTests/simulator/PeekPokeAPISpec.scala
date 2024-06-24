package chiselTests.simulator

import chisel3._
import chisel3.util._
import chisel3.simulator._
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

object TestOp extends ChiselEnum {
  val Add, Sub, Mul = Value
}

class TestPeekPokeEnum extends Module {
  object CmpResult extends ChiselEnum {
    val LT, EQ, GT = Value
  }

  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val op = Input(TestOp())
    val c = Output(UInt(32.W))
    val cmp = Output(CmpResult())
  })

  io.c :#= MuxCase(
    0.U,
    Seq(
      (io.op === TestOp.Add) -> (io.a + io.b),
      (io.op === TestOp.Sub) -> (io.a - io.b),
      (io.op === TestOp.Mul) -> (io.a * io.b).take(32)
    )
  )

  io.cmp :#= Mux1H(
    Seq(
      (io.a < io.b) -> CmpResult.LT,
      (io.a === io.b) -> CmpResult.EQ,
      (io.a > io.b) -> CmpResult.GT
    )
  )
}

class PeekPokeAPISpec extends AnyFunSpec with ChiselSimTester with Matchers {
  val rand = scala.util.Random

  describe("testableEnum") {
    it("testableEnum") {
      val numTests = 100
      test(new TestPeekPokeEnum()) { dut =>
        val truncationMask = (BigInt(1) << dut.io.c.getWidth) - 1
        for {
          _ <- 0 until numTests
          a = BigInt(dut.io.a.getWidth, rand)
          b = BigInt(dut.io.b.getWidth, rand)
          op <- TestOp.all
        } {

          dut.io.a.poke(a)
          dut.io.b.poke(b)
          dut.io.op.poke(op)
          dut.clock.step()

          val peekedOp = dut.io.op.peek()
          assert(peekedOp.litValue == op.litValue)
          assert(peekedOp.toString.contains(TestOp.getClass.getSimpleName.stripSuffix("$")))

          val expected = op match {
            case TestOp.Add => a + b
            case TestOp.Sub => a - b
            case TestOp.Mul => a * b
            case _          => throw new Exception("Invalid operation")
          }
          dut.io.c.expect(expected & truncationMask)

          val expectedCmp = a.compare(b) match {
            case -1 => dut.CmpResult.LT
            case 0  => dut.CmpResult.EQ
            case 1  => dut.CmpResult.GT
          }
          assert(dut.io.cmp.peek().litValue == expectedCmp.litValue)
          assert(dut.io.cmp.peekValue().asBigInt == expectedCmp.litValue)
          dut.io.cmp.expect(expectedCmp)
        }
      }
    }
  }
}
