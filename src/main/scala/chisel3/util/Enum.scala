// See LICENSE for license details.

/** Enum generators, allowing circuit constants to have more meaningful names.
  */

package chisel3.util

import chisel3._

object Enum {
  /** Returns a sequence of Bits subtypes with values from 0 until n. Helper method. */
  private def createValues(n: Int): Seq[UInt] =
    (0 until n).map(_.U(log2Up(n).W))

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
  @deprecated("Use Enum(n), nodeType is always UInt now", "chisel3")
  def apply[T <: Bits](nodeType: T, n: Int): List[T] = {
    require(nodeType.isInstanceOf[UInt], "Only UInt supported for enums")
    apply(n).asInstanceOf[List[T]]
  }

  def apply(n: Int): List[UInt] = createValues(n).toList

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
  @deprecated("Use Enum(l), nodeType is always UInt now", "chisel3")
  def apply[T <: Bits](nodeType: T, l: Symbol *): Map[Symbol, T] = {
    require(nodeType.isInstanceOf[UInt], "Only UInt supported for enums")
    apply(l: _*).asInstanceOf[Map[Symbol, T]]
  }

  def apply(l: Symbol *): Map[Symbol, UInt] = {
    (l zip createValues(l.length)).toMap
  }

  /** Returns a map of the input symbols to unique values of the specified type.
    */
  @deprecated("Use Enum(l), nodeType is always UInt now", "chisel3")
  def apply[T <: Bits](nodeType: T, l: List[Symbol]): Map[Symbol, T] = {
    require(nodeType.isInstanceOf[UInt], "Only UInt supported for enums")
    apply(l).asInstanceOf[Map[Symbol, T]]
  }

  def apply(l: List[Symbol]): Map[Symbol, UInt] = {
    (l zip createValues(l.length)).toMap
  }
}
