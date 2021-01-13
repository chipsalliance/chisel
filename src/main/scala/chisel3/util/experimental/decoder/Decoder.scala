package chisel3.util.experimental.decoder

import chisel3._
import chisel3.util.BitPat

abstract class Decoder {
  def decode(addr: UInt, default: BitPat, mapping: Iterable[(BitPat, BitPat)]): UInt
}