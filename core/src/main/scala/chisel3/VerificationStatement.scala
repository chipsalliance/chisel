// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.reflect.macros.blackbox.Context
import scala.language.experimental.macros
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.experimental.SourceInfo

import scala.annotation.nowarn
import scala.reflect.macros.blackbox

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

  /** Checks for a condition to be valid in the circuit at rising clock edge
    * when not in reset. If the condition evaluates to false, the circuit
    * simulation stops with an error.
    *
    * @param cond condition, assertion fires (simulation fails) when false
    * @param message optional format string to print when the assertion fires
    * @param data optional bits to print in the message formatting
    *
    * @note See [[printf.apply(fmt:String* printf]] for format string documentation
    * @note currently cannot be used in core Chisel / libraries because macro
    * defs need to be compiled first and the SBT project is not set up to do
    * that
    */
  // Macros currently can't take default arguments, so we need two functions to emulate defaults.
  def apply(
    cond:    Bool,
    message: String,
    data:    Bits*
  )(
    implicit sourceInfo: SourceInfo
  ): Assert = macro _applyMacroWithInterpolatorCheck

  /** Checks for a condition to be valid in the circuit at all times. If the
    * condition evaluates to false, the circuit simulation stops with an error.
    *
    * Does not fire when in reset (defined as the current implicit reset, e.g. as set by
    * the enclosing `withReset` or Module.reset.
    *
    * May be called outside of a Module (like defined in a function), so
    * functions using assert make the standard Module assumptions (single clock
    * and single reset).
    *
    * @param cond condition, assertion fires (simulation fails) on a rising clock edge when false and reset is not asserted
    * @param message optional chisel Printable type message
    *
    * @note See [[printf.apply(pable:chisel3\.Printable)*]] for documentation on printf using Printables
    * @note currently cannot be used in core Chisel / libraries because macro
    * defs need to be compiled first and the SBT project is not set up to do
    * that
    */
  def apply(
    cond:    Bool,
    message: Printable
  )(
    implicit sourceInfo: SourceInfo
  ): Assert = macro _applyMacroWithPrintableMessage

  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Assert =
    macro _applyMacroWithNoMessage

  import VerificationStatement._

  /** @group VerifPrintMacros */
  def _applyMacroWithInterpolatorCheck(
    c:          blackbox.Context
  )(cond:       c.Tree,
    message:    c.Tree,
    data:       c.Tree*
  )(sourceInfo: c.Tree
  ): c.Tree = {
    import c.universe._
    printf._checkFormatString(c)(message)
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some(_root_.chisel3.Printable.pack($message, ..$data)))($sourceInfo)"
  }

  /** An elaboration-time assertion. Calls the built-in Scala assert function. */
  def apply(cond: Boolean, message: => String): Unit = Predef.assert(cond, message)

  /** An elaboration-time assertion. Calls the built-in Scala assert function. */
  def apply(cond: Boolean): Unit = Predef.assert(cond, "")

  /** Named class for assertions. */
  final class Assert private[chisel3] () extends VerificationStatement

  /** @group VerifPrintMacros */
  def _applyMacroWithStringMessage(
    c:          blackbox.Context
  )(cond:       c.Tree,
    message:    c.Tree,
    data:       c.Tree*
  )(sourceInfo: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
    q"$apply_impl_do($cond, ${getLine(c)},_root_.scala.Some(_root_.chisel3.Printable.pack($message,..$data)))($sourceInfo)"
  }

  /** @group VerifPrintMacros */
  def _applyMacroWithPrintableMessage(
    c:          blackbox.Context
  )(cond:       c.Tree,
    message:    c.Tree
  )(sourceInfo: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some($message))($sourceInfo)"
  }

  /** @group VerifPrintMacros */
  def _applyMacroWithNoMessage(
    c:          blackbox.Context
  )(cond:       c.Tree
  )(sourceInfo: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.None)($sourceInfo)"
  }

  /** This will be removed in Chisel 3.6 in favor of the Printable version
    *
    * @group VerifPrintMacros
    */
  def _applyWithSourceLine(
    cond:    Bool,
    line:    SourceLineInfo,
    message: Option[String],
    data:    Bits*
  )(
    implicit sourceInfo: SourceInfo
  ): Assert = {
    val id = new Assert()
    when(!Module.reset.asBool) {
      failureMessage("Assertion", line, cond, message.map(Printable.pack(_, data: _*)))
      Builder.pushCommand(Verification(id, Formal.Assert, sourceInfo, Module.clock.ref, cond.ref, ""))
    }
    id
  }

  /** @group VerifPrintMacros */
  def _applyWithSourceLinePrintable(
    cond:    Bool,
    line:    SourceLineInfo,
    message: Option[Printable]
  )(
    implicit sourceInfo: SourceInfo
  ): Assert = {
    val id = new Assert()
    message.foreach(Printable.checkScope(_))
    when(!Module.reset.asBool) {
      failureMessage("Assertion", line, cond, message)
      Builder.pushCommand(Verification(id, Formal.Assert, sourceInfo, Module.clock.ref, cond.ref, ""))
    }
    id
  }
}

object assume extends VerifPrintMacrosDoc {

  /** Assumes a condition to be valid in the circuit at all times.
    * Acts like an assertion in simulation and imposes a declarative
    * assumption on the state explored by formal tools.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    *
    * May be called outside of a Module (like defined in a function), so
    * functions using assert make the standard Module assumptions (single clock
    * and single reset).
    *
    * @param cond condition, assertion fires (simulation fails) when false
    * @param message optional format string to print when the assertion fires
    * @param data optional bits to print in the message formatting
    *
    * @note See [[printf.apply(fmt:String* printf]] for format string documentation
    */
  // Macros currently can't take default arguments, so we need two functions to emulate defaults.
  def apply(
    cond:    Bool,
    message: String,
    data:    Bits*
  )(
    implicit sourceInfo: SourceInfo
  ): Assume = macro _applyMacroWithInterpolatorCheck

  /** Assumes a condition to be valid in the circuit at all times.
    * Acts like an assertion in simulation and imposes a declarative
    * assumption on the state explored by formal tools.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    *
    * May be called outside of a Module (like defined in a function), so
    * functions using assert make the standard Module assumptions (single clock
    * and single reset).
    *
    * @param cond condition, assertion fires (simulation fails) when false
    * @param message optional Printable type message when the assertion fires
    *
    * @note See [[printf.apply(pable:chisel3\.Printable)*]] for documentation on printf using Printables
    */
  def apply(
    cond:    Bool,
    message: Printable
  )(
    implicit sourceInfo: SourceInfo
  ): Assume = macro _applyMacroWithPrintableMessage

  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Assume =
    macro _applyMacroWithNoMessage

  /** An elaboration-time assumption. Calls the built-in Scala assume function. */
  def apply(cond: Boolean, message: => String): Unit = Predef.assume(cond, message)

  /** An elaboration-time assumption. Calls the built-in Scala assume function. */
  def apply(cond: Boolean): Unit = Predef.assume(cond, "")

  /** Named class for assumptions. */
  final class Assume private[chisel3] () extends VerificationStatement

  import VerificationStatement._

  /** @group VerifPrintMacros */
  def _applyMacroWithInterpolatorCheck(
    c:          blackbox.Context
  )(cond:       c.Tree,
    message:    c.Tree,
    data:       c.Tree*
  )(sourceInfo: c.Tree
  ): c.Tree = {
    import c.universe._
    printf._checkFormatString(c)(message)
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some(_root_.chisel3.Printable.pack($message, ..$data)))($sourceInfo)"
  }

  /** @group VerifPrintMacros */
  def _applyMacroWithStringMessage(
    c:          blackbox.Context
  )(cond:       c.Tree,
    message:    c.Tree,
    data:       c.Tree*
  )(sourceInfo: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some(_root_.chisel3.Printable.pack($message, ..$data)))($sourceInfo)"
  }

  /** @group VerifPrintMacros */
  def _applyMacroWithPrintableMessage(
    c:          blackbox.Context
  )(cond:       c.Tree,
    message:    c.Tree
  )(sourceInfo: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some($message))($sourceInfo)"
  }

  /** @group VerifPrintMacros */
  def _applyMacroWithNoMessage(
    c:          blackbox.Context
  )(cond:       c.Tree
  )(sourceInfo: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.None)($sourceInfo)"
  }

  /** This will be removed in Chisel 3.6 in favor of the Printable version
    *
    * @group VerifPrintMacros
    */
  def _applyWithSourceLine(
    cond:    Bool,
    line:    SourceLineInfo,
    message: Option[String],
    data:    Bits*
  )(
    implicit sourceInfo: SourceInfo
  ): Assume = {
    val id = new Assume()
    when(!Module.reset.asBool) {
      failureMessage("Assumption", line, cond, message.map(Printable.pack(_, data: _*)))
      Builder.pushCommand(Verification(id, Formal.Assume, sourceInfo, Module.clock.ref, cond.ref, ""))
    }
    id
  }

  /** @group VerifPrintMacros */
  def _applyWithSourceLinePrintable(
    cond:    Bool,
    line:    SourceLineInfo,
    message: Option[Printable]
  )(
    implicit sourceInfo: SourceInfo
  ): Assume = {
    val id = new Assume()
    message.foreach(Printable.checkScope(_))
    when(!Module.reset.asBool) {
      failureMessage("Assumption", line, cond, message)
      Builder.pushCommand(Verification(id, Formal.Assume, sourceInfo, Module.clock.ref, cond.ref, ""))
    }
    id
  }
}

object cover extends VerifPrintMacrosDoc {

  /** Declares a condition to be covered.
    * At ever clock event, a counter is incremented iff the condition is active
    * and reset is inactive.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    *
    * May be called outside of a Module (like defined in a function), so
    * functions using assert make the standard Module assumptions (single clock
    * and single reset).
    *
    * @param cond condition that will be sampled on every clock tick
    * @param message a string describing the cover event
    */
  // Macros currently can't take default arguments, so we need two functions to emulate defaults.
  def apply(cond: Bool, message: String)(implicit sourceInfo: SourceInfo): Cover =
    macro _applyMacroWithMessage
  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Cover =
    macro _applyMacroWithNoMessage

  /** Named class for cover statements. */
  final class Cover private[chisel3] () extends VerificationStatement

  import VerificationStatement._

  /** @group VerifPrintMacros */
  def _applyMacroWithNoMessage(
    c:          blackbox.Context
  )(cond:       c.Tree
  )(sourceInfo: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLine"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.None)($sourceInfo)"
  }

  /** @group VerifPrintMacros */
  def _applyMacroWithMessage(
    c:          blackbox.Context
  )(cond:       c.Tree,
    message:    c.Tree
  )(sourceInfo: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLine"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some($message))($sourceInfo)"
  }

  /** @group VerifPrintMacros */
  def _applyWithSourceLine(
    cond:    Bool,
    line:    SourceLineInfo,
    message: Option[String]
  )(
    implicit sourceInfo: SourceInfo
  ): Cover = {
    val id = new Cover()
    when(!Module.reset.asBool) {
      Builder.pushCommand(Verification(id, Formal.Cover, sourceInfo, Module.clock.ref, cond.ref, ""))
    }
    id
  }
}

object stop {

  /** Terminate execution, indicating success.
    *
    * @param message a string describing why the simulation was stopped
    */
  def apply(message: String = "")(implicit sourceInfo: SourceInfo): Stop = {
    val stp = new Stop()
    when(!Module.reset.asBool) {
      pushCommand(Stop(stp, sourceInfo, Builder.forcedClock.ref, 0))
    }
    stp
  }

  /** Named class for [[stop]]s. */
  final class Stop private[chisel3] () extends VerificationStatement
}

/** Base class for all verification statements: Assert, Assume, Cover, Stop and Printf. */
abstract class VerificationStatement extends NamedComponent {
  _parent.foreach(_.addId(this))
}

/** Helper functions for common functionality required by stop, assert, assume or cover */
private object VerificationStatement {

  type SourceLineInfo = (String, Int, String)

  def getLine(c: blackbox.Context): SourceLineInfo = {
    val p = c.enclosingPosition
    (p.source.file.name, p.line, p.lineContent.trim): @nowarn // suppress, there's no clear replacement
  }

  def failureMessage(
    kind:     String,
    lineInfo: SourceLineInfo,
    cond:     Bool,
    message:  Option[Printable]
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    val (filename, line, content) = lineInfo
    val lineMsg = s"$filename:$line $content".replaceAll("%", "%%")
    val fmt = message match {
      case Some(msg) =>
        p"$kind failed: $msg\n    at $lineMsg\n"
      case None => p"$kind failed\n    at $lineMsg\n"
    }
    when(!cond) {
      printf.printfWithoutReset(fmt)
    }
  }
}
