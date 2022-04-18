// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.reflect.macros.blackbox.Context
import scala.language.experimental.macros
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo

import scala.reflect.macros.blackbox

object assert {

  /** Checks for a condition to be valid in the circuit at all times. If the
    * condition evaluates to false, the circuit simulation stops with an error.
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
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Assert = macro _applyMacroWithMessage
  def apply(cond: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Assert =
    macro _applyMacroWithNoMessage

  /** An elaboration-time assertion. Calls the built-in Scala assert function. */
  def apply(cond: Boolean, message: => String): Unit = Predef.assert(cond, message)

  /** An elaboration-time assertion. Calls the built-in Scala assert function. */
  def apply(cond: Boolean): Unit = Predef.assert(cond, "")

  /** Named class for assertions. */
  final class Assert private[chisel3] () extends VerificationStatement

  import VerificationStatement._

  def _applyMacroWithMessage(
    c:              blackbox.Context
  )(cond:           c.Tree,
    message:        c.Tree,
    data:           c.Tree*
  )(sourceInfo:     c.Tree,
    compileOptions: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLine"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some($message), ..$data)($sourceInfo, $compileOptions)"
  }

  def _applyMacroWithNoMessage(
    c:              blackbox.Context
  )(cond:           c.Tree
  )(sourceInfo:     c.Tree,
    compileOptions: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLine"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.None)($sourceInfo, $compileOptions)"
  }

  /** Used by our macros. Do not call directly! */
  def _applyWithSourceLine(
    cond:    Bool,
    line:    SourceLineInfo,
    message: Option[String],
    data:    Bits*
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Assert = {
    val id = new Assert()
    when(!Module.reset.asBool()) {
      failureMessage("Assertion", line, cond, message, data)
      Builder.pushCommand(Verification(id, Formal.Assert, sourceInfo, Module.clock.ref, cond.ref, ""))
    }
    id
  }
}

object assume {

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
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Assume = macro _applyMacroWithMessage
  def apply(cond: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Assume =
    macro _applyMacroWithNoMessage

  /** An elaboration-time assumption. Calls the built-in Scala assume function. */
  def apply(cond: Boolean, message: => String): Unit = Predef.assume(cond, message)

  /** An elaboration-time assumption. Calls the built-in Scala assume function. */
  def apply(cond: Boolean): Unit = Predef.assume(cond, "")

  /** Named class for assumptions. */
  final class Assume private[chisel3] () extends VerificationStatement

  import VerificationStatement._

  def _applyMacroWithMessage(
    c:              blackbox.Context
  )(cond:           c.Tree,
    message:        c.Tree,
    data:           c.Tree*
  )(sourceInfo:     c.Tree,
    compileOptions: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLine"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some($message), ..$data)($sourceInfo, $compileOptions)"
  }

  def _applyMacroWithNoMessage(
    c:              blackbox.Context
  )(cond:           c.Tree
  )(sourceInfo:     c.Tree,
    compileOptions: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLine"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.None)($sourceInfo, $compileOptions)"
  }

  /** Used by our macros. Do not call directly! */
  def _applyWithSourceLine(
    cond:    Bool,
    line:    SourceLineInfo,
    message: Option[String],
    data:    Bits*
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Assume = {
    val id = new Assume()
    when(!Module.reset.asBool()) {
      failureMessage("Assumption", line, cond, message, data)
      Builder.pushCommand(Verification(id, Formal.Assume, sourceInfo, Module.clock.ref, cond.ref, ""))
    }
    id
  }
}

object cover {

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
  def apply(cond: Bool, message: String)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Cover =
    macro _applyMacroWithMessage
  def apply(cond: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Cover =
    macro _applyMacroWithNoMessage

  /** Named class for cover statements. */
  final class Cover private[chisel3] () extends VerificationStatement

  import VerificationStatement._

  def _applyMacroWithNoMessage(
    c:              blackbox.Context
  )(cond:           c.Tree
  )(sourceInfo:     c.Tree,
    compileOptions: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLine"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.None)($sourceInfo, $compileOptions)"
  }

  def _applyMacroWithMessage(
    c:              blackbox.Context
  )(cond:           c.Tree,
    message:        c.Tree
  )(sourceInfo:     c.Tree,
    compileOptions: c.Tree
  ): c.Tree = {
    import c.universe._
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLine"))
    q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some($message))($sourceInfo, $compileOptions)"
  }

  /** Used by our macros. Do not call directly! */
  def _applyWithSourceLine(
    cond:    Bool,
    line:    SourceLineInfo,
    message: Option[String]
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Cover = {
    val id = new Cover()
    when(!Module.reset.asBool()) {
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
  def apply(message: String = "")(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Stop = {
    val stp = new Stop()
    when(!Module.reset.asBool) {
      pushCommand(Stop(stp, sourceInfo, Builder.forcedClock.ref, 0))
    }
    stp
  }

  /** Terminate execution with a failure code. */
  @deprecated(
    "Non-zero return codes are not well supported. Please use assert(false.B) if you want to indicate a failure.",
    "Chisel 3.5"
  )
  def apply(code: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Stop = {
    val stp = new Stop()
    when(!Module.reset.asBool) {
      pushCommand(Stop(stp, sourceInfo, Builder.forcedClock.ref, code))
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
    (p.source.file.name, p.line, p.lineContent.trim)
  }

  // creates a printf to inform the user of a failed assertion or assumption
  def failureMessage(
    kind:     String,
    lineInfo: SourceLineInfo,
    cond:     Bool,
    message:  Option[String],
    data:     Seq[Bits]
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Unit = {
    val (filename, line, content) = lineInfo
    val lineMsg = s"$filename:$line $content".replaceAll("%", "%%")
    val fmt = message match {
      case Some(msg) =>
        s"$kind failed: $msg\n    at $lineMsg\n"
      case None => s"$kind failed\n    at $lineMsg\n"
    }
    when(!cond) {
      printf.printfWithoutReset(fmt, data: _*)
    }
  }
}
