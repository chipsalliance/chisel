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

  def calcExpectedScalarOpResult(op: TestOp.Type, a: BigInt, b: BigInt, w: Int): BigInt = {
    val truncationMask = (BigInt(1) << w) - 1
    (op match {
      case TestOp.Add => a + b
      case TestOp.Sub => a - b
      case TestOp.Mul => a * b
      case _          => throw new Exception("Invalid operation")
    }) & truncationMask
  }

  def calcExpectedCmp(a: BigInt, b: BigInt): CmpResult.Type = {
    a.compare(b) match {
      case -1 => CmpResult.LT
      case 0  => CmpResult.EQ
      case 1  => CmpResult.GT
    }
  }

  def calcExpectedVSumBigInts(v1: Seq[BigInt], v2: Seq[BigInt]): Seq[BigInt] = v1.zip(v2).map { case (x, y) => x + y }

  def calcExpectedVSum(v1: Seq[BigInt], v2: Seq[BigInt]): Vec[UInt] =
    Vec.Lit(calcExpectedVSumBigInts(v1, v2).map(_.U): _*)

  def calcExpectedVSum(v1: Seq[BigInt], v2: Seq[BigInt], w: Int): Vec[UInt] =
    Vec.Lit(calcExpectedVSumBigInts(v1, v2).map(_.U((w + 1).W)): _*)

  def calcExpectedVecProductBigInts(v1: Seq[BigInt], v2: Seq[BigInt]): Seq[Seq[BigInt]] = v1.map { x => v2.map(x * _) }

  def calcExpectedVecProduct(v1: Seq[BigInt], v2: Seq[BigInt], w: Int): Vec[Vec[UInt]] = Vec.Lit(
    calcExpectedVecProductBigInts(v1, v2).map { x =>
      Vec.Lit(x.map { y =>
        val p = y
        if (w > 0) p.U((2 * w).W) else p.U
      }: _*)
    }: _*
  )

  def calcExpectedVecProduct(v1: Vec[UInt], v2: Vec[UInt], w: Int): Vec[Vec[UInt]] =
    calcExpectedVecProduct(v1.map(_.litValue), v2.map(_.litValue), w)

  def calcExpectedVDotBigInt(v1: Seq[BigInt], v2: Seq[BigInt]): BigInt = v1.zip(v2).map { case (x, y) => x * y }.sum

  def calcExpectedVDot(v1: Seq[BigInt], v2: Seq[BigInt], w: Int = -1): UInt = {
    val dp = calcExpectedVDotBigInt(v1, v2)
    if (w > 0) dp.U((2 * w + v1.length - 1).W) else dp.U
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
