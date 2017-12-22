// See LICENSE for license details.

package chisel3.core

import scala.reflect.macros.blackbox.Context
import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo

object assert { // scalastyle:ignore object.name
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
  def apply(cond: Bool, message: String, data: Bits*)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit = macro apply_impl_msg_data
  def apply(cond: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit = macro apply_impl

  def apply_impl_msg_data(c: Context)(cond: c.Tree, message: c.Tree, data: c.Tree*)(sourceInfo: c.Tree, compileOptions: c.Tree): c.Tree = {
    import c.universe._
    val p = c.enclosingPosition
    val condStr = s"${p.source.file.name}:${p.line} ${p.lineContent.trim}"
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("apply_impl_do"))
    q"$apply_impl_do($cond, $condStr, _root_.scala.Some($message), ..$data)($sourceInfo, $compileOptions)"
  }

  def apply_impl(c: Context)(cond: c.Tree)(sourceInfo: c.Tree, compileOptions: c.Tree): c.Tree = {
    import c.universe._
    val p = c.enclosingPosition
    val condStr = s"${p.source.file.name}:${p.line} ${p.lineContent.trim}"
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("apply_impl_do"))
    q"$apply_impl_do($cond, $condStr, _root_.scala.None)($sourceInfo, $compileOptions)"
  }

  def apply_impl_do(cond: Bool, line: String, message: Option[String], data: Bits*)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) {
    val escLine = line.replaceAll("%", "%%")
    when (!(cond || Module.reset.toBool)) {
      val fmt = message match {
        case Some(msg) =>
          s"Assertion failed: $msg\n    at $escLine\n"
        case None => s"Assertion failed\n    at $escLine\n"
      }
      printf.printfWithoutReset(fmt, data:_*)
      pushCommand(Stop(sourceInfo, Node(Builder.forcedClock), 1))
    }
  }

  /** An elaboration-time assertion, otherwise the same as the above run-time
    * assertion. */
  def apply(cond: Boolean, message: => String) {
    Predef.assert(cond, message)
  }

  /** A workaround for default-value overloading problems in Scala, just
    * 'assert(cond, "")' */
  def apply(cond: Boolean) {
    Predef.assert(cond, "")
  }
}

object stop { // scalastyle:ignore object.name
  /** Terminate execution with a failure code. */
  def apply(code: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit = {
    when (!Module.reset.toBool) {
      pushCommand(Stop(sourceInfo, Node(Builder.forcedClock), code))
    }
  }

  /** Terminate execution, indicating success. */
  def apply()(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit = {
    stop(0)
  }
}
