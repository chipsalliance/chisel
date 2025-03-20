// See LICENSE for license details.

package chisel3Base.examples

import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester

object HammingMatrices {
  // Identity Matrix
  private def I(n: Int) = {
    val x = Array.fill(n)(Array.fill(n)(0))
    0 until n map (i => x(i)(i) = 1)
    x
  }

  // Generating Matrix
  def G(n: Int, k: Int): Seq[UInt] = { (n, k) match {
    case (7, 4) => Seq(
      "b1000".U,
      "b0100".U,
      "b0010".U,
      "b0001".U,
      "b1101".U,
      "b1011".U,
      "b0111".U).reverse
    case _ => throw new Exception(s"No known generator matrix for n: $n, k: $k")
  }}

  // Parity Check Matrix
  def H(n: Int, k: Int): Seq[UInt] = { (n, k) match {
    case (7, 4) => Seq(
      "b1101100".U,
      "b1011010".U,
      "b0111001".U).reverse
    case _ => throw new Exception(s"No known parity check matrix for n: $n, k: $k")
  }}

  // Extraction of the data-carrying values
  def R(n: Int, k: Int): Seq[UInt] = { (n, k) match {
    case (7, 4) => Seq(
      "b1000000".U,
      "b0100000".U,
      "b0010000".U,
      "b0001000".U).reverse
    case _ => throw new Exception(s"No known decode matrix for n: $n, k: $k")
  }}
}

trait Hamming extends Ecc {
  def encode(x: UInt): UInt = {
    val y = Seq.fill(n)(x)
    val g = HammingMatrices.G(n, k)
    Vec(y zip g map { case(a, b) => (a & b).xorR}).asUInt
  }

  private def syndrome(x: UInt): UInt = {
    val z = Seq.fill(n)(x)
    val h = HammingMatrices.H(n, k)
    Vec(z zip h map { case(a, b) => (a & b).xorR }).asUInt
  }

  def error(x: UInt): Bool = syndrome(x).orR

  def decode(y: UInt): UInt = {
    // [TODO] Add error correction
    val z = Seq.fill(n)(y)
    val r = HammingMatrices.R(n, k)
    Vec(z zip r map { case(a, b) => (a & b).xorR }).asUInt
  }
}

trait Hamming_7_4 extends Hamming {
  def n: Int = 7
  def k: Int = 4
}

object HammingEncoder {
  def apply[A <: Data](gen: A) = Module(new EccEncode(gen) with Hamming_7_4)
}

object HammingDecoder {
  def apply[A <: Data](gen: A) = Module(new EccDecode(gen) with Hamming_7_4)
}

class HammingCounter(n: Int) extends Module {
  val io = IO(new Bundle{
    val in = Input(UInt(n.W))
    val out = Output(UInt(n.W))
    val err = Output(Bool())
  })

  val enc = HammingEncoder(io.in)
  val dec = HammingDecoder(io.in)

  enc.io.in := io.in
  dec.io.in := enc.io.out
  io.out := dec.io.out
}

class HammingCounterTest extends BasicTester {
  val n = 4
  val dut = Module(new HammingCounter(n))

  val s_INIT :: s_RUN :: s_DONE :: Nil = Enum(3)
  val state = Reg(init = s_INIT)

  when (state === s_INIT) { state := s_RUN }

  val (count, done) = Counter(state === s_RUN, math.pow(2, n).toInt)
  dut.io.in := count

  assert(dut.io.in === dut.io.out)
  assert(dut.io.err === false.B)

  when (done) { state := s_DONE }
  when (state === s_DONE) { stop }
}
