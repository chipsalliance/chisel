// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait when$Intf { self: when.type =>

  /** Create a `when` condition block, where whether a block of logic is
    * executed or not depends on the conditional.
    *
    * @param cond condition to execute upon
    * @param block logic that runs only if `cond` is true
    *
    * @example
    * {{{
    * when ( myData === 3.U ) {
    *   // Some logic to run when myData equals 3.
    * } .elsewhen ( myData === 1.U ) {
    *   // Some logic to run when myData equals 1.
    * } .otherwise {
    *   // Some logic to run when myData is neither 3 nor 1.
    * }
    * }}}
    */
  def apply(
    cond: => Bool
  )(block: => Any)(
    implicit sourceInfo: SourceInfo
  ): WhenContext = _applyImpl(cond)(block)
}

private[chisel3] trait WhenContext$Intf { self: WhenContext =>

  /** This block of logic gets executed if above conditions have been
    * false and this condition is true. The lazy argument pattern
    * makes it possible to delay evaluation of cond, emitting the
    * declaration and assignment of the Bool node of the predicate in
    * the correct place.
    */
  def elsewhen(
    elseCond: => Bool
  )(block: => Any)(
    implicit sourceInfo: SourceInfo
  ): WhenContext = _elsewhenImpl(elseCond)(block)

  /** This block of logic gets executed only if the above conditions
    * were all false. No additional logic blocks may be appended past
    * the `otherwise`. The lazy argument pattern makes it possible to
    * delay evaluation of cond, emitting the declaration and
    * assignment of the Bool node of the predicate in the correct
    * place.
    */
  def otherwise(block: => Any)(implicit sourceInfo: SourceInfo): Unit =
    _otherwiseImpl(block)
}
