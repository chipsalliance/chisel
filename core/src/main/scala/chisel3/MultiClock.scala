// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._

import scala.language.experimental.macros

object withClockAndReset {

  /** Creates a new Clock and Reset scope
    *
    * @param clock the new implicit Clock
    * @param reset the new implicit Reset
    * @param block the block of code to run with new implicit Clock and Reset
    * @return the result of the block
    */
  def apply[T](clock: Clock, reset: Reset)(block: => T): T = apply(Some(clock), Some(reset))(block)

  /** Creates a new Clock and Reset scope
    *
    * @param clock the new implicit Clock, None will set no implicit clock
    * @param reset the new implicit Reset, None will set no implicit reset
    * @param block the block of code to run with new implicit Clock and Reset
    * @return the result of the block
    */
  def apply[T](clock: Option[Clock], reset: Option[Reset])(block: => T): T = {
    // Save parentScope
    val parentClock = Builder.currentClock
    val parentReset = Builder.currentReset

    Builder.currentClock = clock
    Builder.currentReset = reset

    val res = block // execute block

    // Return to old scope
    Builder.currentClock = parentClock
    Builder.currentReset = parentReset
    res
  }
}

object withClock {

  /** Creates a new Clock scope
    *
    * @param clock the new implicit Clock
    * @param block the block of code to run with new implicit Clock
    * @return the result of the block
    */
  def apply[T](clock: Clock)(block: => T): T = apply(Some(clock))(block)

  /** Creates a new Clock scope
    *
    * @param clock the new implicit Clock, None will set no implicit clock
    * @param block the block of code to run with new implicit Clock
    * @return the result of the block
    */
  def apply[T](clock: Option[Clock])(block: => T): T = {
    // Save parentScope
    val parentClock = Builder.currentClock
    Builder.currentClock = clock
    val res = block // execute block
    // Return to old scope
    Builder.currentClock = parentClock
    res
  }
}

object withReset {

  /** Creates a new Reset scope
    *
    * @param reset the new implicit Reset
    * @param block the block of code to run with new implicit Reset
    * @return the result of the block
    */
  def apply[T](reset: Reset)(block: => T): T = apply(Some(reset))(block)

  /** Creates a new Reset scope
    *
    * @param reset the new implicit Reset, None will set no implicit reset
    * @param block the block of code to run with new implicit Reset
    * @return the result of the block
    */
  def apply[T](reset: Option[Reset])(block: => T): T = {
    // Save parentScope
    val parentReset = Builder.currentReset
    Builder.currentReset = reset
    val res = block // execute block
    // Return to old scope
    Builder.currentReset = parentReset
    res
  }

}

object withoutIO {

  /** Creates a new scope in which IO creation causes a runtime error
    *
    * @param block the block of code to run with where IO creation is illegal
    * @return the result of the block
    */
  def apply[T](block: => T)(implicit si: experimental.SourceInfo): T = {
    Builder.currentModule.get.disallowIO(block)
  }
}
