// See LICENSE for license details.

/** Enum generators, allowing circuit constants to have more meaningful names.
  */

package chisel3.util

import chisel3._

object Enum {
  /** Returns a sequence of Bits subtypes with values from 0 until n. Helper method. */
  private def createValues[T <: Bits](nodeType: T, n: Int): Seq[T] =
    (0 until n).map(x => nodeType.fromInt(x, log2Up(n)))

  /** Returns n unique values of the specified type. Can be used with unpacking to define enums.
    *
    * @example {{{
    * val state_on :: state_off :: Nil = Enum(UInt(), 2)
    * val current_state = UInt()
    * switch (current_state) {
    *   is (state_on) {
    *      ...
    *   }
    *   if (state_off) {
    *      ...
    *   }
    * }
    * }}}
    *
    */
  def apply[T <: Bits](nodeType: T, n: Int): List[T] = createValues(nodeType, n).toList

  /** Returns a map of the input symbols to unique values of the specified type.
    *
    * @example {{{
    * val states = Enum(UInt(), 'on, 'off)
    * val current_state = UInt()
    * switch (current_state) {
    *   is (states('on)) {
    *     ...
    *   }
    *   if (states('off)) {
    *     ..
    *   }
    * }
    * }}}
    */
  def apply[T <: Bits](nodeType: T, l: Symbol *): Map[Symbol, T] = (l zip createValues(nodeType, l.length)).toMap

  /** Returns a map of the input symbols to unique values of the specified type.
    */
  def apply[T <: Bits](nodeType: T, l: List[Symbol]): Map[Symbol, T] = (l zip createValues(nodeType, l.length)).toMap
}
