// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo}

private[chisel3] final case class ClockAndReset(clock: Clock, reset: Bool)

object withClockAndReset {  // scalastyle:ignore object.name
  def apply[T](clock: Clock, reset: Bool)(block: => T)(implicit sourceInfo: SourceInfo): T = {
    // Save parentScope
    val parentScope = Builder.currentClockAndReset
    Builder.currentClockAndReset = Some(ClockAndReset(clock, reset))
    val res = block // execute block
    // Return to old scope
    Builder.currentClockAndReset = parentScope
    res
  }
}

