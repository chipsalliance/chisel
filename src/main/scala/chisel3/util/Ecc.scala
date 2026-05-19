// See LICENSE for license details.

package chisel3Base.examples

import chisel3._
import chisel3.util._
import chisel3.internal.firrtl.Width

trait Ecc {
  def n: Int
  def k: Int
  require(n > k)
  def encode(x: UInt): UInt
  def decode(y: UInt): UInt
  def error(y: UInt): Bool
}

class EccEncodeIO[T <: Data](gen: T, n: Int, k: Int) extends Bundle {
  val in = Input(gen.chiselCloneType)
  val out = Output(UInt((gen.getWidth + (n - k)).W))

  override def cloneType = new EccEncodeIO(gen, n, k).asInstanceOf[this.type]
}

class EccDecodeIO[T <: Data](gen: T, n: Int, k: Int) extends Bundle {
  val in = Input(UInt((gen.getWidth + (n - k)).W))
  val out = Output(gen.chiselCloneType)
  val err = Output(Bool())

  override def cloneType = new EccDecodeIO(gen, n, k).asInstanceOf[this.type]
}

abstract class EccEncode[T <: Data](gen: T) extends Module {
  this: Ecc =>

  lazy val io = IO(new EccEncodeIO[T](gen, this.n, this.k))

  io.out := this.encode(io.in.asUInt)
}

abstract class EccDecode[T <: Data](gen: T) extends Module {
  this: Ecc =>

  val io = IO(new EccDecodeIO[T](gen, this.n, this.k))

  io.out := gen.chiselCloneType.fromBits(this.decode(io.in))
  io.err := this.error(io.in)
}
