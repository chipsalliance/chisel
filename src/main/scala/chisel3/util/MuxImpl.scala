// SPDX-License-Identifier: Apache-2.0

/** Mux circuit generators.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.internal.Builder

private[chisel3] trait Mux1HImpl {
  protected def _applyImpl[T <: Data](sel: Seq[Bool], in: Seq[T])(implicit sourceInfo: SourceInfo): T = {
    if (sel.size != in.size) {
      Builder.error(s"Mux1H: input Seqs must have the same length, got sel ${sel.size} and in ${in.size}")
    }
    _applyImpl(sel.zip(in))
  }

  protected def _applyImpl[T <: Data](in: Iterable[(Bool, T)])(implicit sourceInfo: SourceInfo): T =
    SeqUtils.oneHotMux(in)

  protected def _applyImpl[T <: Data](sel: UInt, in: Seq[T])(implicit sourceInfo: SourceInfo): T =
    _applyImpl((0 until in.size).map(sel(_)), in)

  protected def _applyImpl(sel: UInt, in: UInt)(implicit sourceInfo: SourceInfo): Bool = (sel & in).orR
}

private[chisel3] trait PriorityMuxImpl {

  protected def _applyImpl[T <: Data](in: Seq[(Bool, T)]): T = SeqUtils.priorityMux(in)

  protected def _applyImpl[T <: Data](sel: Seq[Bool], in: Seq[T])(implicit sourceInfo: SourceInfo): T = {
    if (sel.size != in.size) {
      Builder.error(s"PriorityMux: input Seqs must have the same length, got sel ${sel.size} and in ${in.size}")
    }
    _applyImpl(sel.zip(in))
  }

  protected def _applyImpl[T <: Data](sel: Bits, in: Seq[T])(implicit sourceInfo: SourceInfo): T =
    _applyImpl((0 until in.size).map(sel(_)), in)
}

private[chisel3] trait MuxLookupImpl {

  protected def _applyEnumImpl[S <: EnumType, T <: Data](
    key:     S,
    default: T,
    mapping: Seq[(S, T)]
  )(
    implicit sourceinfo: SourceInfo
  ): T =
    _applyImpl[UInt, T](key.asUInt, default, mapping.map { case (s, t) => (s.asUInt, t) })

  protected def _applyImpl[S <: UInt, T <: Data](
    key:     S,
    default: T,
    mapping: Seq[(S, T)]
  )(
    implicit sourceinfo: SourceInfo
  ): T = {
    /* If the mapping is defined for all possible values of the key, then don't use the default value */
    val (defaultx, mappingx) = key.widthOption match {
      case Some(width) =>
        val keySetSize = BigInt(1) << width
        val keyMask = keySetSize - 1
        val distinctLitKeys = mapping.flatMap(_._1.litOption).map(_ & keyMask).distinct
        if (distinctLitKeys.size == keySetSize) {
          (mapping.head._2, mapping.tail)
        } else {
          (default, mapping)
        }
      case None => (default, mapping)
    }

    mappingx.foldLeft(defaultx) { case (d, (k, v)) => Mux(k === key, v, d) }
  }
}

/** Given an association of values to enable signals, returns the first value with an associated
  * high enable signal.
  *
  * @example {{{
  * MuxCase(default, Array(c1 -> a, c2 -> b))
  * }}}
  */
object MuxCase {

  /** @param default the default value if none are enabled
    * @param mapping a set of data values with associated enables
    * @return the first value in mapping that is enabled
    */
  def apply[T <: Data](default: T, mapping: Seq[(Bool, T)]): T = {
    var res = default
    for ((t, v) <- mapping.reverse) {
      res = Mux(t, v, res)
    }
    res
  }
}
