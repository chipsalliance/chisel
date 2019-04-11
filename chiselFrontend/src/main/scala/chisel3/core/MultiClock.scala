// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo}

private[chisel3] final class ClockAndReset(val clockOpt: Option[Clock], val resetOpt: Option[Reset]) {
  def clock: Clock = clockOpt.get
  def reset: Reset = resetOpt.get

  def setClock(newClock: Clock): ClockAndReset = new ClockAndReset(Some(newClock), resetOpt)
  def setReset(newReset: Reset): ClockAndReset = new ClockAndReset(clockOpt, Some(newReset))
}

private[chisel3] final object ClockAndReset {
  def apply(clock: Clock, reset: Reset): ClockAndReset = {
    new ClockAndReset(Some(clock), Some(reset))
  }
  def empty: ClockAndReset = new ClockAndReset(None, None)

  def unapply(arg: ClockAndReset): Option[(Option[Clock], Option[Reset])] = {
    Some((arg.clockOpt, arg.resetOpt))
  }
}

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
    val parentScope = Builder.currentClockAndReset
    Builder.currentClockAndReset = ClockAndReset(clock, reset)
    val res = block // execute block
    // Return to old scope
    Builder.currentClockAndReset = parentScope
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
    val parentScope = Builder.currentClockAndReset
    Builder.currentClockAndReset = Builder.currentClockAndReset.setClock(clock)
    val res = block // execute block
    // Return to old scope
    Builder.currentClockAndReset = parentScope
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
    val parentScope = Builder.currentClockAndReset
    Builder.currentClockAndReset = Builder.currentClockAndReset.setReset(reset)
    val res = block // execute block
    // Return to old scope
    Builder.currentClockAndReset = parentScope
    res
  }

}

