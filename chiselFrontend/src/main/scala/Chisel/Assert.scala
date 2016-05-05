// See LICENSE for license details.

package Chisel

import scala.reflect.macros.blackbox.Context
import scala.language.experimental.macros

import internal._
import internal.Builder.pushCommand
import internal.firrtl._
import internal.sourceinfo.SourceInfo

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
    * @param message optional message to print when the assertion fires
    *
    * @note currently cannot be used in core Chisel / libraries because macro
    * defs need to be compiled first and the SBT project is not set up to do
    * that
    */
  // Macros currently can't take default arguments, so we need two functions to emulate defaults.
  def apply(cond: Bool, message: String)(implicit sourceInfo: SourceInfo): Unit = macro apply_impl_msg
  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Unit = macro apply_impl

  def apply_impl_msg(c: Context)(cond: c.Tree, message: c.Tree)(sourceInfo: c.Tree): c.Tree = {
    import c.universe._
    val p = c.enclosingPosition
    val condStr = s"${p.source.file.name}:${p.line} ${p.lineContent.trim}"
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("apply_impl_do"))
    q"$apply_impl_do($cond, $condStr, _root_.scala.Some($message))($sourceInfo)"
  }

  def apply_impl(c: Context)(cond: c.Tree)(sourceInfo: c.Tree): c.Tree = {
    import c.universe._
    val p = c.enclosingPosition
    val condStr = s"${p.source.file.name}:${p.line} ${p.lineContent.trim}"
    val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("apply_impl_do"))
    q"$apply_impl_do($cond, $condStr, _root_.scala.None)($sourceInfo)"
  }

  def apply_impl_do(cond: Bool, line: String, message: Option[String])(implicit sourceInfo: SourceInfo) {
    when (!(cond || Builder.dynamicContext.currentModule.get.reset)) {
      message match {
        case Some(str) => printf.printfWithoutReset(s"Assertion failed: $str\n    at $line\n")
        case None => printf.printfWithoutReset(s"Assertion failed\n    at $line\n")
      }
      pushCommand(Stop(sourceInfo, Node(Builder.dynamicContext.currentModule.get.clock), 1))
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
