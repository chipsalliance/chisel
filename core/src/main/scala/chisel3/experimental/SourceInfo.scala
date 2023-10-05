// SPDX-License-Identifier: Apache-2.0

// This file contains macros for adding source locators at the point of invocation.
//
// This is not part of coreMacros to disallow this macro from being implicitly invoked in Chisel
// frontend (and generating source locators in Chisel core), which is almost certainly a bug.
//
// Note: While these functions and definitions are not private (macros can't be
// private), these are NOT meant to be part of the public API (yet) and no
// forward compatibility guarantees are made.
// A future revision may stabilize the source locator API to allow library
// writers to append source locator information at the point of a library
// function invocation.

package chisel3.experimental

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context
import chisel3.internal.sourceinfo.SourceInfoMacro

/** Abstract base class for generalized source information.
  */
sealed trait SourceInfo {

  /** A prettier toString
    *
    * Make a useful message if SourceInfo is available, nothing otherwise
    */
  def makeMessage(f: String => String): String

  /** The filename for the originating source file, if known */
  def filenameOption: Option[String]
}

sealed trait NoSourceInfo extends SourceInfo {
  def makeMessage(f: String => String): String = ""
  def filenameOption: Option[String] = None
}

/** For when source info can't be generated because of a technical limitation, like for Reg because
  * Scala macros don't support named or default arguments.
  */
case object UnlocatableSourceInfo extends NoSourceInfo

/** For when source info isn't generated because the function is deprecated and we're lazy.
  */
case object DeprecatedSourceInfo extends NoSourceInfo

/** For FIRRTL lines from a Scala source line.
  *
  * @note A column == 0 indicates no column
  */
case class SourceLine(filename: String, line: Int, col: Int) extends SourceInfo {
  def makeMessage(f: String => String): String = f(s"@[${this.serialize}]")
  def filenameOption: Option[String] = Some(filename)

  /** Convert to String for FIRRTL emission */
  def serialize: String = {
    if (col == 0) s"$filename $line" else s"$filename $line:$col"
  }
}

object SourceInfo {
  implicit def materialize: SourceInfo = macro SourceInfoMacro.generate_source_info

  /** Returns the best guess at the first stack frame that belongs to user code.
    */
  private def getUserLineNumber: Option[StackTraceElement] = {
    def isChiselClassname(className: String): Boolean = {
      // List of classpath prefixes that are Chisel internals and should be ignored when looking for user code
      // utils are not part of internals and errors there can be reported
      val chiselPrefixes = Set(
        "java.",
        "scala.",
        "chisel3.",
        "chisel3.internal.",
        "chisel3.experimental.",
        "chisel3.package$" // for some compatibility / deprecated types
      )
      !chiselPrefixes.filter(className.startsWith(_)).isEmpty
    }

    Thread
      .currentThread()
      .getStackTrace
      .toList
      .dropWhile(
        // Get rid of everything in Chisel core
        ste => isChiselClassname(ste.getClassName)
      )
      .headOption
  }

  private[chisel3] def materializeFromStacktrace: SourceInfo =
    getUserLineNumber match {
      case Some(elt) => new SourceLine(elt.getFileName, elt.getLineNumber, 0)
      case None      => UnlocatableSourceInfo
    }
}
