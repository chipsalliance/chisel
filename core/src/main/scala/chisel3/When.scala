// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.experimental.{SourceInfo, UnlocatableSourceInfo}

object when {

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
    cond:  => Bool
  )(block: => Any
  )(
    implicit sourceInfo: SourceInfo
  ): WhenContext = {
    new WhenContext(sourceInfo, Some(() => cond), block, 0, Nil)
  }

  /** Returns the current `when` condition
    *
    * This is the conjunction of conditions for all `whens` going up the call stack
    * {{{
    * when (a) {
    *   when (b) {
    *     when (c) {
    *     }.otherwise {
    *       when.cond // this is equal to: a && b && !c
    *     }
    *   }
    * }
    * }}}
    */
  def cond: Bool = {
    implicit val sourceInfo = UnlocatableSourceInfo
    val whens = Builder.whenStack
    whens.foldRight(true.B) {
      case (ctx, acc) => acc && ctx.localCond
    }
  }
}

/**  A WhenContext may represent a when, and elsewhen, or an
  *  otherwise. Since FIRRTL does not have an "elsif" statement,
  *  alternatives must be mapped to nested if-else statements inside
  *  the alternatives of the preceeding condition. In order to emit
  *  proper FIRRTL, it is necessary to keep track of the depth of
  *  nesting of the FIRRTL whens. Due to the "thin frontend" nature of
  *  Chisel3, it is not possible to know if a when or elsewhen has a
  *  succeeding elsewhen or otherwise; therefore, this information is
  *  added by preprocessing the command queue.
  */
final class WhenContext private[chisel3] (
  sourceInfo:  SourceInfo,
  cond:        Option[() => Bool],
  block:       => Any,
  firrtlDepth: Int,
  // For capturing conditions from prior whens or elsewhens
  altConds: List[() => Bool]) {

  private var scopeOpen = false

  /** Returns the local condition, inverted for an otherwise */
  private[chisel3] def localCond: Bool = {
    implicit val sourceInfo = UnlocatableSourceInfo
    val alt = altConds.foldRight(true.B) {
      case (c, acc) => acc & !c()
    }
    cond
      .map(alt && _())
      .getOrElse(alt)
  }

  /** This block of logic gets executed if above conditions have been
    * false and this condition is true. The lazy argument pattern
    * makes it possible to delay evaluation of cond, emitting the
    * declaration and assignment of the Bool node of the predicate in
    * the correct place.
    */
  def elsewhen(
    elseCond: => Bool
  )(block:    => Any
  )(
    implicit sourceInfo: SourceInfo
  ): WhenContext = {
    new WhenContext(sourceInfo, Some(() => elseCond), block, firrtlDepth + 1, cond ++: altConds)
  }

  /** This block of logic gets executed only if the above conditions
    * were all false. No additional logic blocks may be appended past
    * the `otherwise`. The lazy argument pattern makes it possible to
    * delay evaluation of cond, emitting the declaration and
    * assignment of the Bool node of the predicate in the correct
    * place.
    */
  def otherwise(block: => Any)(implicit sourceInfo: SourceInfo): Unit =
    new WhenContext(sourceInfo, None, block, firrtlDepth + 1, cond ++: altConds)

  def active: Boolean = scopeOpen

  /*
   *
   */
  if (firrtlDepth > 0) { pushCommand(AltBegin(sourceInfo)) }
  cond.foreach(c => pushCommand(WhenBegin(sourceInfo, c().ref)))
  Builder.pushWhen(this)
  try {
    scopeOpen = true
    block
  } catch {
    case _: scala.runtime.NonLocalReturnControl[_] =>
      throwException(
        "Cannot exit from a when() block with a \"return\"!" +
          " Perhaps you meant to use Mux or a Wire as a return value?"
      )
  }
  scopeOpen = false
  Builder.popWhen()
  cond.foreach(_ => pushCommand(WhenEnd(sourceInfo, firrtlDepth)))
  if (cond.isEmpty) { pushCommand(OtherwiseEnd(sourceInfo, firrtlDepth)) }
}
