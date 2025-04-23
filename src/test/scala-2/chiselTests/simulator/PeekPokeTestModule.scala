package chiselTests.simulator

import chisel3._
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.VecLiterals._
import chisel3.util._
import chisel3.simulator._
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.must.Matchers

object PeekPokeTestModule {
  object TestOp extends ChiselEnum {
    val Add, Sub, Mul = Value
  }
  object CmpResult extends ChiselEnum {
    val LT, EQ, GT = Value
  }
}

class PeekPokeTestModule(w: Int, val vecDim: Int = 3) extends Module {
  import PeekPokeTestModule._

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
