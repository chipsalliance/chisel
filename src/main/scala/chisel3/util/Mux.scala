// See LICENSE for license details.

/** Mux circuit generators.
  */

package chisel3.util

import chisel3._
import chisel3.core.SeqUtils

/** Builds a Mux tree out of the input signal vector using a one hot encoded
  * select signal. Returns the output of the Mux tree.
  *
  * @example {{{
  * val hotValue = chisel3.util.Mux1H(Seq(
  *  io.selector(0) -> 2.U,
  *  io.selector(1) -> 4.U,
  *  io.selector(2) -> 8.U,
  *  io.selector(4) -> 11.U,
  * ))
  * }}}
  *
  * @note results undefined if multiple select signals are simultaneously high
  */
object Mux1H {
  def apply[T <: Data](sel: Seq[Bool], in: Seq[T]): T =
    apply(sel zip in)
  def apply[T <: Data](in: Iterable[(Bool, T)]): T = SeqUtils.oneHotMux(in)
  def apply[T <: Data](sel: UInt, in: Seq[T]): T =
    apply((0 until in.size).map(sel(_)), in)
  def apply(sel: UInt, in: UInt): Bool = (sel & in).orR
}

/** Builds a Mux tree under the assumption that multiple select signals
  * can be enabled. Priority is given to the first select signal.
  *
  * @example {{{
  * val hotValue = chisel3.util.PriorityMux(Seq(
  *  io.selector(0) -> 2.U,
  *  io.selector(1) -> 4.U,
  *  io.selector(2) -> 8.U,
  *  io.selector(4) -> 11.U,
  * ))
  * }}}
  * Returns the output of the Mux tree.
  */
object PriorityMux {
  def apply[T <: Data](in: Seq[(Bool, T)]): T = SeqUtils.priorityMux(in)
  def apply[T <: Data](sel: Seq[Bool], in: Seq[T]): T = apply(sel zip in)
  def apply[T <: Data](sel: Bits, in: Seq[T]): T = apply((0 until in.size).map(sel(_)), in)
}

/** Creates a cascade of n Muxs to search for a key value.
  *
  * @example {{{
  * MuxLookup(idx, default,
  *     Array(0.U -> a, 1.U -> b))
  * }}}
  */
object MuxLookup {
  /** @param key a key to search for
    * @param default a default value if nothing is found
    * @param mapping a sequence to search of keys and values
    * @return the value found or the default if not
    */
  def apply[S <: UInt, T <: Data] (key: S, default: T, mapping: Seq[(S, T)]): T = {
    var res = default
    for ((k, v) <- mapping.reverse)
      res = Mux(k === key, v, res)
    res
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
    * @return the first value in mapping that is enabled */
  def apply[T <: Data] (default: T, mapping: Seq[(Bool, T)]): T = {
    var res = default
    for ((t, v) <- mapping.reverse){
      res = Mux(t, v, res)
    }
    res
  }
}
