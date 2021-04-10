package chisel3.util.experimental

import chisel3._
import chisel3.util.BitPat
import chisel3.util.experimental.minimizer._

import scala.collection.mutable.ArrayBuffer

object decode {
  def apply(addr: UInt, default: BitPat, mapping: Iterable[(BitPat, BitPat)], minimizer: Minimizer = QMCMinimizer()): UInt = {
    val minimizedTable = minimizer.minimize(default, mapping.toSeq)
    val (inputs, orPlaneOutputs) = pla(minimizedTable)
    inputs := addr

    val widthOfOutputs = orPlaneOutputs.width
    val outputs = Wire(UInt(widthOfOutputs))
    // invert outputs which defaults to `1`
    outputs := orPlaneOutputs ^ default.value.U(widthOfOutputs)
    outputs
  }

  def apply(addr: UInt, default: Seq[BitPat], mappingIn: Iterable[(BitPat, Seq[BitPat])], minimizer: Minimizer = QMCMinimizer()): Seq[UInt] = {
    val mapping = ArrayBuffer.fill(default.size)(ArrayBuffer[(BitPat, BitPat)]())
    for ((key, values) <- mappingIn)
      for ((value, i) <- values.zipWithIndex)
        mapping(i) += key -> value
    for ((thisDefault, thisMapping) <- default.zip(mapping))
      yield apply(addr, thisDefault, thisMapping, minimizer)
  }

  def apply(addr: UInt, default: Seq[BitPat], mappingIn: List[(UInt, Seq[BitPat])], minimizer: Minimizer = QMCMinimizer()): Seq[UInt] =
    apply(
      addr,
      default,
      mappingIn.map(m => (BitPat(m._1), m._2)).asInstanceOf[Iterable[(BitPat, Seq[BitPat])]],
      minimizer
    )

  def apply(addr: UInt, onSet: Iterable[UInt], offSet: Iterable[UInt], minimizer: Minimizer = QMCMinimizer()): Bool =
    apply(
      addr,
      BitPat.dontCare(1),
      onSet.map(BitPat(_) -> BitPat("b1")) ++ offSet.map(BitPat(_) -> BitPat("b0")),
      minimizer
    ).asBool()
}
