package chisel3.util.experimental.decoder

import chisel3._
import chisel3.util.BitPat

abstract class Decoder {
  /**
    * Minimize a logic function `F` with one multiple 1-bit output values.
    *
    * @param addr      Input values of the logic function, packed in a UInt.
    * @param default   Default output values of the logic function, used for inputs that are not specified in the truth table.
    * @param mapping   Truth table of the logic function.
    *                  [(inputPattern -> outputs), ...]
    * @return          F(addr)
    */
  def decode(addr: UInt, default: BitPat, mapping: Iterable[(BitPat, BitPat)]): UInt
}