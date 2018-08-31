// See LICENSE for license details.

package chisel3.internal

import scala.collection.mutable.{ArrayBuffer, LinkedHashMap}

import chisel3.core._

class ChiselException(message: String, cause: Throwable) extends Exception(message, cause)

private[chisel3] object throwException {
  def apply(s: String, t: Throwable = null): Nothing =
    throw new ChiselException(s, t)
}

/** Records and reports runtime errors and warnings. */
private[chisel3] class ErrorLog {
  /** Log an error message */
  def error(m: => String): Unit =
    errors += new Error(m, getUserLineNumber)

  /** Log a warning message */
  def warning(m: => String): Unit =
    errors += new Warning(m, getUserLineNumber)

  /** Emit an informational message */
  def info(m: String): Unit =
    println(new Info("[%2.3f] %s".format(elapsedTime/1e3, m), None))  // scalastyle:ignore regex

  /** Log a deprecation warning message */
  def deprecated(m: => String, location: Option[String]): Unit = {
    val sourceLoc = location match {
      case Some(loc) => loc
      case None => getUserLineNumber match {
        case Some(elt: StackTraceElement) => s"${elt.getFileName}:${elt.getLineNumber}"
        case None => "(unknown)"
      }
    }

    val thisEntry = (m, sourceLoc)
    deprecations += ((thisEntry, deprecations.getOrElse(thisEntry, 0) + 1))
  }

  /** Throw an exception if any errors have yet occurred. */
  def checkpoint(): Unit = {
    val depTag = s"[${Console.BLUE}deprecated${Console.RESET}]"
    val warnTag = s"[${Console.YELLOW}warn${Console.RESET}]"
    val errTag = s"[${Console.RED}error${Console.RESET}]"
    deprecations.foreach { case ((message, sourceLoc), count) =>
      println(s"$depTag $sourceLoc ($count calls): $message")
    }
    errors foreach println

    if (!deprecations.isEmpty) {
      println(s"$warnTag ${Console.YELLOW}There were ${deprecations.size} deprecated function(s) used." +
          s" These may stop compiling in a future release - you are encouraged to fix these issues.${Console.RESET}")
      println(s"$warnTag Line numbers for deprecations reported by Chisel may be inaccurate; enable scalac compiler deprecation warnings via either of the following methods:")
      println(s"$warnTag   In the sbt interactive console, enter:")
      println(s"""$warnTag     set scalacOptions in ThisBuild ++= Seq("-unchecked", "-deprecation")""")
      println(s"$warnTag   or, in your build.sbt, add the line:")
      println(s"""$warnTag     scalacOptions := Seq("-unchecked", "-deprecation")""")
    }

    val allErrors = errors.filter(_.isFatal)
    val allWarnings = errors.filter(!_.isFatal)

    if (!allWarnings.isEmpty && !allErrors.isEmpty) {
      println(s"$errTag There were ${Console.RED}${allErrors.size} error(s)${Console.RESET} and ${Console.YELLOW}${allWarnings.size} warning(s)${Console.RESET} during hardware elaboration.")
    } else if (!allWarnings.isEmpty) {
      println(s"$warnTag There were ${Console.YELLOW}${allWarnings.size} warning(s)${Console.RESET} during hardware elaboration.")
    } else if (!allErrors.isEmpty) {
      println(s"$errTag There were ${Console.RED}${allErrors.size} error(s)${Console.RESET} during hardware elaboration.")
    }

    if (!allErrors.isEmpty) {
      throwException("Fatal errors during hardware elaboration")
    } else {
      // No fatal errors, clear accumulated warnings since they've been reported
      errors.clear()
    }
  }

  /** Returns the best guess at the first stack frame that belongs to user code.
    */
  private def getUserLineNumber = {
    def isChiselClassname(className: String): Boolean = {
      // List of classpath prefixes that are Chisel internals and should be ignored when looking for user code
      // utils are not part of internals and errors there can be reported
      val chiselPrefixes = Set(
          "java.",
          "scala.",
          "chisel3.internal.",
          "chisel3.core.",
          "chisel3.package$"  // for some compatibility / deprecated types
          )
      !chiselPrefixes.filter(className.startsWith(_)).isEmpty
    }

    Thread.currentThread().getStackTrace.toList.dropWhile(
          // Get rid of everything in Chisel core
          ste => isChiselClassname(ste.getClassName)
        ).headOption
  }

  private val errors = ArrayBuffer[LogEntry]()
  private val deprecations = LinkedHashMap[(String, String), Int]()

  private val startTime = System.currentTimeMillis
  private def elapsedTime: Long = System.currentTimeMillis - startTime
}

private abstract class LogEntry(msg: => String, line: Option[StackTraceElement]) {
  def isFatal: Boolean = false
  def format: String

  override def toString: String = line match {
    case Some(l) => s"${format} ${l.getFileName}:${l.getLineNumber}: ${msg} in class ${l.getClassName}"
    case None => s"${format} ${msg}"
  }

  protected def tag(name: String, color: String): String =
    s"[${color}${name}${Console.RESET}]"
}

private class Error(msg: => String, line: Option[StackTraceElement]) extends LogEntry(msg, line) {
  override def isFatal: Boolean = true
  def format: String = tag("error", Console.RED)
}

private class Warning(msg: => String, line: Option[StackTraceElement]) extends LogEntry(msg, line) {
  def format: String = tag("warn", Console.YELLOW)
}

private class Info(msg: => String, line: Option[StackTraceElement]) extends LogEntry(msg, line) {
  def format: String = tag("info", Console.MAGENTA)
}
