// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo}

object when {  // scalastyle:ignore object.name
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

  def apply(cond: => Bool)(block: => Unit)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): WhenContext = {
    new WhenContext(sourceInfo, Some(() => cond), block)
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
final class WhenContext(sourceInfo: SourceInfo, cond: Option[() => Bool], block: => Unit, firrtlDepth: Int = 0) {

  /** This block of logic gets executed if above conditions have been
    * false and this condition is true. The lazy argument pattern
    * makes it possible to delay evaluation of cond, emitting the
    * declaration and assignment of the Bool node of the predicate in
    * the correct place.
    */
  def elsewhen (elseCond: => Bool)(block: => Unit)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): WhenContext = {
    new WhenContext(sourceInfo, Some(() => elseCond), block, firrtlDepth+1)
  }

  /** This block of logic gets executed only if the above conditions
    * were all false. No additional logic blocks may be appended past
    * the `otherwise`. The lazy argument pattern makes it possible to
    * delay evaluation of cond, emitting the declaration and
    * assignment of the Bool node of the predicate in the correct
    * place.
    */
  def otherwise(block: => Unit)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit =
    new WhenContext(sourceInfo, None, block, firrtlDepth+1)

  /*
   * 
   */
  if (firrtlDepth > 0) { pushCommand(AltBegin(sourceInfo)) }
  cond.foreach( c => pushCommand(WhenBegin(sourceInfo, c().ref)) )
  Builder.whenDepth += 1
  try {
    block
  } catch {
    case ret: scala.runtime.NonLocalReturnControl[_] =>
      throwException("Cannot exit from a when() block with a \"return\"!" +
        " Perhaps you meant to use Mux or a Wire as a return value?"
      )
  }
  Builder.whenDepth -= 1
  cond.foreach( c => pushCommand(WhenEnd(sourceInfo,firrtlDepth)) )
  if (cond.isEmpty) { pushCommand(OtherwiseEnd(sourceInfo,firrtlDepth)) }
}
