package chisel3.util.experimental

import chisel3.{Bool, UInt}
import chisel3.util.BitPat

import scala.collection.mutable.ArrayBuffer

package object decoder {
  def decode(addr: UInt, default: BitPat, mapping: Iterable[(BitPat, BitPat)], decoder: Decoder): UInt =
    decoder.decode(addr, default, mapping)

  def decode(addr: UInt, default: Seq[BitPat], mappingIn: Iterable[(BitPat, Seq[BitPat])], decoder: Decoder): Seq[UInt] = {
    val mapping = ArrayBuffer.fill(default.size)(ArrayBuffer[(BitPat, BitPat)]())
    for ((key, values) <- mappingIn)
      for ((value, i) <- values.zipWithIndex)
        mapping(i) += key -> value
    for ((thisDefault, thisMapping) <- default.zip(mapping))
      yield decode(addr, thisDefault, thisMapping, decoder)
  }

  def decode(addr: UInt, default: Seq[BitPat], mappingIn: List[(UInt, Seq[BitPat])], decoder: Decoder): Seq[UInt] =
    decode(
      addr,
      default,
      mappingIn.map(m => (BitPat(m._1), m._2)).asInstanceOf[Iterable[(BitPat, Seq[BitPat])]],
      decoder
    )

  def decode(addr: UInt, trues: Iterable[UInt], falses: Iterable[UInt], decoder: Decoder): Bool =
    decode(
      addr,
      BitPat.dontCare(1),
      trues.map(BitPat(_) -> BitPat("b1")) ++ falses.map(BitPat(_) -> BitPat("b0")),
      decoder
    ).asBool
}
