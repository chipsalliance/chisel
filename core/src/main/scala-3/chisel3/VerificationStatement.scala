// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir._
import chisel3.layer.block

import scala.language.experimental.macros

/** Base class for all verification statements: Assert, Assume, Cover, Stop and Printf. */
abstract class VerificationStatement extends NamedComponent {
  _parent.foreach(_.addId(this))
}

/** Scaladoc information for internal verification statement macros
  * that are used in objects assert, assume and cover.
  *
  * @groupdesc VerifPrintMacros
  *
  * <p>
  * '''These internal methods are not part of the public-facing API!'''
  * </p>
  * <br>
  *
  * @groupprio VerifPrintMacros 1001
  */
trait VerifPrintMacrosDoc

object assert extends VerifPrintMacrosDoc {

  /** Named class for assertions. */
  final class Assert private[chisel3] () extends VerificationStatement

  /** An elaboration-time assertion. Calls the built-in Scala assert function. */
  def apply(cond: Boolean, message: => String): Unit = Predef.assert(cond, message)

  /** An elaboration-time assertion. Calls the built-in Scala assert function. */
  def apply(cond: Boolean): Unit = Predef.assert(cond, "")
}

object assume extends VerifPrintMacrosDoc {

  /** Named class for assumptions. */
  final class Assume private[chisel3] () extends VerificationStatement

  /** An elaboration-time assumption. Calls the built-in Scala assume function. */
  def apply(cond: Boolean, message: => String): Unit = Predef.assume(cond, message)

  /** An elaboration-time assumption. Calls the built-in Scala assume function. */
  def apply(cond: Boolean): Unit = Predef.assume(cond, "")
}

object cover extends VerifPrintMacrosDoc {

  /** Named class for cover statements. */
  final class Cover private[chisel3] () extends VerificationStatement

  type SourceLineInfo = (String, Int)
}

object stop {

  /** Terminate execution, indicating success and printing a message.
    *
    * @param message a message describing why simulation was stopped
    */
  def apply(message: String)(implicit sourceInfo: SourceInfo): Stop = buildStopCommand(Some(PString(message)))

  /** Terminate execution, indicating success and printing a message.
    *
    * @param message a printable describing why simulation was stopped
    */
  def apply(message: Printable)(implicit sourceInfo: SourceInfo): Stop = buildStopCommand(Some(message))

  /** Terminate execution, indicating success.
    */
  def apply()(implicit sourceInfo: SourceInfo): Stop = buildStopCommand(None)

  /** Named class for [[stop]]s. */
  final class Stop private[chisel3] () extends VerificationStatement

  private def buildStopCommand(message: Option[Printable])(implicit sourceInfo: SourceInfo): Stop = {
    val stopId = new Stop()
    block(layers.Verification, skipIfAlreadyInBlock = true, skipIfLayersEnabled = true) {
      message.foreach(Printable.checkScope(_))
      when(!Module.reset.asBool) {
        message.foreach(PrintfMacrosCompat.printfWithoutReset(_))
        pushCommand(chisel3.internal.firrtl.ir.Stop(stopId, sourceInfo, Builder.forcedClock.ref, 0))
      }
    }
    stopId
  }
}
