// See LICENSE for license details.

package Chisel

import internal._
import internal.Builder.pushCommand
import internal.firrtl._

object when {  // scalastyle:ignore object.name
  /** Create a `when` condition block, where whether a block of logic is
    * executed or not depends on the conditional.
    *
    * @param cond condition to execute upon
    * @param block logic that runs only if `cond` is true
    *
    * @example
    * {{{
    * when ( myData === UInt(3) ) {
    *   // Some logic to run when myData equals 3.
    * } .elsewhen ( myData === UInt(1) ) {
    *   // Some logic to run when myData equals 1.
    * } .otherwise {
    *   // Some logic to run when myData is neither 3 nor 1.
    * }
    * }}}
    */
  def apply(cond: Bool)(block: => Unit): WhenContext = {
    new WhenContext(cond, !cond)(block)
  }
}

/** Internal mechanism for generating a when. Because of the way FIRRTL
  * commands are emitted, generating a FIRRTL elsewhen or nested whens inside
  * elses would be difficult. Instead, this keeps track of the negative of the
  * previous conditions, so when an elsewhen or otherwise is used, it checks
  * that both the condition is true and all the previous conditions have been
  * false.
  */
class WhenContext(cond: Bool, prevCond: => Bool)(block: => Unit) {
  /** This block of logic gets executed if above conditions have been false
    * and this condition is true.
    */
  def elsewhen (elseCond: Bool)(block: => Unit): WhenContext = {
    new WhenContext(prevCond && elseCond, prevCond && !elseCond)(block)
  }

  /** This block of logic gets executed only if the above conditions were all
    * false. No additional logic blocks may be appended past the `otherwise`.
    */
  def otherwise(block: => Unit): Unit =
    new WhenContext(prevCond, null)(block)

  pushCommand(WhenBegin(cond.ref))
  block
  pushCommand(WhenEnd())
}
