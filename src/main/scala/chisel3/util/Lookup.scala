// See LICENSE for license details.

package chisel3.util

import chisel3._

object ListLookup {
  def apply[T <: Data](addr: UInt, default: List[T], mapping: Array[(BitPat, List[T])]): List[T] = {
    val map = mapping.map(m => (m._1 === addr, m._2))
    default.zipWithIndex map { case (d, i) =>
      map.foldRight(d)((m, n) => Mux(m._1, m._2(i), n))
    }
  }
}

object Lookup {
  def apply[T <: Bits](addr: UInt, default: T, mapping: Seq[(BitPat, T)]): T =
    ListLookup(addr, List(default), mapping.map(m => (m._1, List(m._2))).toArray).head
}
