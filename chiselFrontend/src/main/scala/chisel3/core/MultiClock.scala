// See LICENSE for license details.

package chisel3.core

import chisel3.internal._

import scala.language.experimental.macros

object withClockAndReset {  // scalastyle:ignore object.name
  /** Creates a new Clock and Reset scope
    *
    * @param clock the new implicit Clock
    * @param reset the new implicit Reset
    * @param block the block of code to run with new implicit Clock and Reset
    * @return the result of the block
    */
  def apply[T](clock: Clock, reset: Reset)(block: => T): T = {
    // Save parentScope
    val parentClock = Builder.currentClock
    val parentReset = Builder.currentReset

    Builder.currentClock = Some(clock)
    Builder.currentReset = Some(reset)

    val res = block // execute block

    // Return to old scope
    Builder.currentClock = parentClock
    Builder.currentReset = parentReset
    res
  }
}

object withClock {  // scalastyle:ignore object.name
  /** Creates a new Clock scope
    *
    * @param clock the new implicit Clock
    * @param block the block of code to run with new implicit Clock
    * @return the result of the block
    */
  def apply[T](clock: Clock)(block: => T): T =  {
    // Save parentScope
    val parentClock = Builder.currentClock
    Builder.currentClock = Some(clock)
    val res = block // execute block
    // Return to old scope
    Builder.currentClock = parentClock
    res
  }
}

object withReset {  // scalastyle:ignore object.name
  /** Creates a new Reset scope
    *
    * @param reset the new implicit Reset
    * @param block the block of code to run with new implicit Reset
    * @return the result of the block
    */
  def apply[T](reset: Reset)(block: => T): T = {
    // Save parentScope
    val parentReset = Builder.currentReset
    Builder.currentReset = Some(reset)
    val res = block // execute block
    // Return to old scope
    Builder.currentReset = parentReset
    res
  }

}

