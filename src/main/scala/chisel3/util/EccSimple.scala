// See LICENSE for license details.

package chisel3Base.examples

import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester

trait Parity extends Ecc {
  // (n, k) are arbitrary here, only their relationship is important
  def n: Int = 1
  def k: Int = 0

  def encode(x: UInt): UInt = Cat(x.asUInt, x.asUInt.xorR)
  def decode(y: UInt): UInt = y(y.getWidth - 1, 1)
  def error(x: UInt): Bool = encode(decode(x.asUInt))(0) =/= x(0)
}

trait TwoOutOfFive {
  // [TODO] Implement
}

trait Repetition {
  // [TODO] Implement
}

object ParityEncoder {
  def apply[T <: Data](gen: T) = Module(new EccEncode(gen) with Parity)
}

object ParityDecoder {
  def apply[T <: Data](gen: T) = Module(new EccDecode(gen) with Parity)
}

class ParityCounter(n: Int) extends Module {
  val io = IO(new Bundle{
    val in = Input(UInt(n.W))
    val out = Output(UInt(n.W))
    val err = Output(Bool())
  })

  val enc = ParityEncoder(io.in)
  val dec = ParityDecoder(io.in)

  enc.io.in := io.in
  dec.io.in := enc.io.out
  io.out := dec.io.out
}

class ParityCounterTest extends BasicTester {
  val n = 4
  val dut = Module(new ParityCounter(n))

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
