// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._

/** For each element in a list, muxes (looks up) between cases (one per list element) based on a
  * common address.
  *
  * @note This appears to be an odd, specialized operator that we haven't seen used much, and seems
  *       to be a holdover from chisel2. This may be deprecated and removed, usage is not
  *       recommended.
  *
  * @param addr common select for cases, shared (same) across all list elements
  * @param default default value for each list element, should the address not match any case
  * @param mapping list of cases, where each entry consists of a [[chisel3.util.BitPat BitPath]] (compared against addr) and
  *                a list of elements (same length as default) that is the output value for that
  *                element (will have the same index in the output).
  *
  * @example {{{
  * ListLookup(2.U,  // address for comparison
  *                          List(10.U, 11.U, 12.U),   // default "row" if none of the following cases match
  *     Array(BitPat(2.U) -> List(20.U, 21.U, 22.U),  // this "row" hardware-selected based off address 2.U
  *           BitPat(3.U) -> List(30.U, 31.U, 32.U))
  * ) // hardware-evaluates to List(20.U, 21.U, 22.U)
  * // Note: if given address 0.U, the above would hardware evaluate to List(10.U, 11.U, 12.U)
  * }}}
  */
object ListLookup {
  def apply[T <: Data](addr: UInt, default: List[T], mapping: Array[(BitPat, List[T])]): List[T] = {
    val map = mapping.map(m => (m._1 === addr, m._2))
    default.zipWithIndex.map {
      case (d, i) =>
        map.foldRight(d)((m, n) => Mux(m._1, m._2(i), n))
    }
  }
}

/** Muxes between cases based on whether an address matches any pattern for a case.
  * Similar to [[chisel3.util.MuxLookup MuxLookup]], but uses [[chisel3.util.BitPat BitPat]] for address comparison.
  *
  * @note This appears to be an odd, specialized operator that we haven't seen used much, and seems
  *       to be a holdover from chisel2. This may be deprecated and removed, usage is not
  *       recommended.
  *
  * @param addr address to select between cases
  * @param default default value should the address not match any case
  * @param mapping list of cases, where each entry consists of a [[chisel3.util.BitPat BitPat]] (compared against addr) and the
  *          output value if the BitPat matches
  */
object Lookup {
  def apply[T <: Bits](addr: UInt, default: T, mapping: Seq[(BitPat, T)]): T =
    ListLookup(addr, List(default), mapping.map(m => (m._1, List(m._2))).toArray).head
}
