// See LICENSE for license details.

package Chisel
import Builder.pushCommand

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
  def apply(cond: => Bool)(block: => Unit): WhenContext = {
    new WhenContext(cond)(block)
  }
}

class WhenContext(cond: => Bool)(block: => Unit) {
  /** This block of logic gets executed if above conditions have been false
    * and this condition is true.
    */
  def elsewhen (cond: => Bool)(block: => Unit): WhenContext =
    doOtherwise(when(cond)(block))

  /** This block of logic gets executed only if the above conditions were all
    * false. No additional logic blocks may be appended past the `otherwise`.
    */
  def otherwise(block: => Unit): Unit =
    doOtherwise(block)

  pushCommand(WhenBegin(cond.ref))
  block
  pushCommand(WhenEnd())

  private def doOtherwise[T](block: => T): T = {
    pushCommand(WhenElse())
    val res = block
    pushCommand(WhenEnd())
    res
  }
}
