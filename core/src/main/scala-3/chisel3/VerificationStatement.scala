// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir._
import chisel3.layer.block

import scala.language.experimental.macros

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
