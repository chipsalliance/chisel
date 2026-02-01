// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import chisel3.internal.NamedComponent
import chisel3.internal.Builder
import chisel3.internal.Builder.pushCommand
import chisel3.layer.block

/** Base class for all verification statements: Assert, Assume, Cover, Stop and Printf. */
abstract class VerificationStatement extends NamedComponent {
  _parent.foreach(_.addId(this))
}

object assert extends Assert$Intf {

  /** Named class for assertions. */
  final class Assert private[chisel3] () extends VerificationStatement

  /** An elaboration-time assertion. Calls the built-in Scala assert function. */
  def apply(cond: Boolean, message: => String): Unit = Predef.assert(cond, message)

  /** An elaboration-time assertion. Calls the built-in Scala assert function. */
  def apply(cond: Boolean): Unit = Predef.assert(cond, "")
}

object assume extends Assume$Intf {

  /** Named class for assumptions. */
  final class Assume private[chisel3] () extends VerificationStatement

  /** An elaboration-time assumption. Calls the built-in Scala assume function. */
  def apply(cond: Boolean, message: => String): Unit = Predef.assume(cond, message)

  /** An elaboration-time assumption. Calls the built-in Scala assume function. */
  def apply(cond: Boolean): Unit = Predef.assume(cond, "")
}

object cover extends Cover$Impl {

  /** Named class for cover statements. */
  final class Cover private[chisel3] () extends VerificationStatement

  type SourceLineInfo = (String, Int)
}

object stop extends Stop$Intf {

  /** Named class for [[stop]]s. */
  final class Stop private[chisel3] () extends VerificationStatement

  private[chisel3] def _applyImpl(message: String)(implicit sourceInfo: SourceInfo): Stop =
    buildStopCommand(Some(PString(message)))

  private[chisel3] def _applyImpl(message: Printable)(implicit sourceInfo: SourceInfo): Stop =
    buildStopCommand(Some(message))

  private[chisel3] def _applyImpl()(implicit sourceInfo: SourceInfo): Stop =
    buildStopCommand(None)

  private def buildStopCommand(message: Option[Printable])(implicit sourceInfo: SourceInfo): Stop = {
    val stopId = new Stop()
    block(layers.Verification, skipIfAlreadyInBlock = true, skipIfLayersEnabled = true) {
      message.foreach(Printable.checkScope(_))
      when(!Module.reset.asBool) {
        message.foreach(SimLog.StdErr.printfWithoutReset(_))
        pushCommand(chisel3.internal.firrtl.ir.Stop(stopId, sourceInfo, Builder.forcedClock.ref, 0))
      }
    }
    stopId
  }
}
