// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal.sourceinfo._

private[chisel3] object SeqUtils {
  /** Concatenates the data elements of the input sequence, in sequence order, together.
    * The first element of the sequence forms the least significant bits, while the last element
    * in the sequence forms the most significant bits.
    *
    * Equivalent to r(n-1) ## ... ## r(1) ## r(0).
    */
  def asUInt[T <: Bits](in: Seq[T]): UInt = macro SourceInfoTransform.inArg

  def do_asUInt[T <: Bits](in: Seq[T])(implicit sourceInfo: SourceInfo): UInt = {
    if (in.tail.isEmpty) {
      in.head.asUInt
    } else {
      val left = asUInt(in.slice(0, in.length/2))
      val right = asUInt(in.slice(in.length/2, in.length))
      right ## left
    }
  }

  /** Outputs the number of elements that === Bool(true).
    */
  def count(in: Seq[Bool]): UInt = macro SourceInfoTransform.inArg

  def do_count(in: Seq[Bool])(implicit sourceInfo: SourceInfo): UInt = in.size match {
    case 0 => UInt(0)
    case 1 => in.head
    case n => count(in take n/2) +& count(in drop n/2)
  }

  /** Returns the data value corresponding to the first true predicate.
    */
  def priorityMux[T <: Data](in: Seq[(Bool, T)]): T = macro SourceInfoTransform.inArg

  def do_priorityMux[T <: Data](in: Seq[(Bool, T)])(implicit sourceInfo: SourceInfo): T = {
    if (in.size == 1) {
      in.head._2
    } else {
      Mux(in.head._1, in.head._2, priorityMux(in.tail))
    }
  }

  /** Returns the data value corresponding to the lone true predicate.
    *
    * @note assumes exactly one true predicate, results undefined otherwise
    */
  def oneHotMux[T <: Data](in: Iterable[(Bool, T)]): T = macro CompileOptionsTransform.inArg

  def do_oneHotMux[T <: Data](in: Iterable[(Bool, T)])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    if (in.tail.isEmpty) {
      in.head._2
    } else {
      val masked = for ((s, i) <- in) yield Mux(s, i.asUInt, UInt(0))
      val width = in.map(_._2.width).reduce(_ max _)
      in.head._2.cloneTypeWidth(width).fromBits(masked.reduceLeft(_|_))
    }
  }
}
