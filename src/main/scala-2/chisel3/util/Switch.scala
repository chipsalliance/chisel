// SPDX-License-Identifier: Apache-2.0

/** Conditional blocks.
  */

package chisel3.util

import scala.language.experimental.macros
import scala.reflect.macros.blackbox._

import chisel3._

/** Conditional logic to form a switch block. See [[is$ is]] for the case API.
  *
  * @example {{{
  * switch (myState) {
  *   is (state1) {
  *     // some logic here that runs when myState === state1
  *   }
  *   is (state2) {
  *     // some logic here that runs when myState === state2
  *   }
  * }
  * }}}
  */
object switch {
  def apply[T <: Element](cond: T)(x: => Any): Unit = macro impl
  def impl(c: Context)(cond: c.Tree)(x: c.Tree): c.Tree = {
    import c.universe._
    val q"..$body" = x
    val res = body.foldLeft(q"""new chisel3.util.SwitchContext($cond, None, Set.empty)""") {
      case (acc, tree) =>
        tree match {
          // TODO: remove when Chisel compatibility package is removed
          case q"Chisel.`package`.is.apply( ..$params )( ..$body )" => q"$acc.is( ..$params )( ..$body )"
          case q"chisel3.util.is.apply( ..$params )( ..$body )"     => q"$acc.is( ..$params )( ..$body )"
          case b                                                    => throw new Exception(s"Cannot include blocks that do not begin with is() in switch.")
        }
    }
    q"""{ $res }"""
  }
}
