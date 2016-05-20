// See LICENSE for license details.

package Chisel

import scala.language.experimental.macros

import internal.sourceinfo.{SourceInfo, SourceInfoTransform}

private[Chisel] object SeqUtils {
  /** Equivalent to Cat(r(n-1), ..., r(0)) */
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

  /** Counts the number of true Bools in a Seq */
  def count(in: Seq[Bool]): UInt = macro SourceInfoTransform.inArg

  def do_count(in: Seq[Bool])(implicit sourceInfo: SourceInfo): UInt = {
    if (in.size == 0) {
      UInt(0)
    } else if (in.size == 1) {
      in.head
    } else {
      count(in.slice(0, in.size/2)) + (UInt(0) ## count(in.slice(in.size/2, in.size)))
    }
  }

  /** Returns data value corresponding to first true predicate */
  def priorityMux[T <: Bits](in: Seq[(Bool, T)]): T = macro SourceInfoTransform.inArg

  def do_priorityMux[T <: Bits](in: Seq[(Bool, T)])(implicit sourceInfo: SourceInfo): T = {
    if (in.size == 1) {
      in.head._2
    } else {
      Mux(in.head._1, in.head._2, priorityMux(in.tail))
    }
  }

  /** Returns data value corresponding to lone true predicate */
  def oneHotMux[T <: Data](in: Iterable[(Bool, T)]): T = macro SourceInfoTransform.inArg

  def do_oneHotMux[T <: Data](in: Iterable[(Bool, T)])(implicit sourceInfo: SourceInfo): T = {
    if (in.tail.isEmpty) {
      in.head._2
    } else {
      val masked = for ((s, i) <- in) yield Mux(s, i.toBits, Bits(0))
      val width = in.map(_._2.width).reduce(_ max _)
      in.head._2.cloneTypeWidth(width).fromBits(masked.reduceLeft(_|_))
    }
  }
}
