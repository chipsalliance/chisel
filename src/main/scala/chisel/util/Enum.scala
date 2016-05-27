// See LICENSE for license details.

/** Enum generators, allowing circuit constants to have more meaningful names.
  */

package chisel.util

import chisel._

object Enum {
  /** Returns a sequence of Bits subtypes with values from 0 until n. Helper method. */
  private def createValues[T <: Bits](nodeType: T, n: Int): Seq[T] =
    (0 until n).map(x => nodeType.fromInt(x, log2Up(n)))

  /** create n enum values of given type */
  def apply[T <: Bits](nodeType: T, n: Int): List[T] = createValues(nodeType, n).toList

  /** create enum values of given type and names */
  def apply[T <: Bits](nodeType: T, l: Symbol *): Map[Symbol, T] = (l zip createValues(nodeType, l.length)).toMap

  /** create enum values of given type and names */
  def apply[T <: Bits](nodeType: T, l: List[Symbol]): Map[Symbol, T] = (l zip createValues(nodeType, l.length)).toMap
}
