// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import scala.annotation.tailrec
import scala.collection.mutable.{ArrayBuffer, LinkedHashMap}

class ChiselException(message: String, cause: Throwable = null) extends Exception(message, cause) {

  /** Package names whose stack trace elements should be trimmed when generating a trimmed stack trace */
  val blacklistPackages: Set[String] = Set("chisel3", "scala", "java", "sun", "sbt")

  /** The object name of Chisel's internal `Builder`. Everything stack trace element after this will be trimmed. */
  val builderName: String = chisel3.internal.Builder.getClass.getName

  /** Examine a [[Throwable]], to extract all its causes. Innermost cause is first.
    * @param throwable an exception to examine
    * @return a sequence of all the causes with innermost cause first
    */
  @tailrec
  private def getCauses(throwable: Throwable, acc: Seq[Throwable] = Seq.empty): Seq[Throwable] =
    throwable.getCause() match {
      case null => throwable +: acc
      case a    => getCauses(a, throwable +: acc)
    }

  /** Returns true if an exception contains  */
  private def containsBuilder(throwable: Throwable): Boolean =
    throwable.getStackTrace().collectFirst {
      case ste if ste.getClassName().startsWith(builderName) => throwable
    }.isDefined

  /** Examine this [[ChiselException]] and it's causes for the first [[Throwable]] that contains a stack trace including
    * a stack trace element whose declaring class is the [[builderName]]. If no such element exists, return this
    * [[ChiselException]].
    */
  private lazy val likelyCause: Throwable =
    getCauses(this).collectFirst{ case a if containsBuilder(a) => a }.getOrElse(this)

  /** For an exception, return a stack trace trimmed to user code only
    *
    * This does the following actions:
    *
    *   1. Trims the top of the stack trace while elements match [[blacklistPackages]]
    *   2. Trims the bottom of the stack trace until an element matches [[builderName]]
    *   3. Trims from the [[builderName]] all [[blacklistPackages]]
    *
    * @param throwable the exception whose stack trace should be trimmed
    * @return an array of stack trace elements
    */
  private def trimmedStackTrace(throwable: Throwable): Array[StackTraceElement] = {
    def isBlacklisted(ste: StackTraceElement) = {
      val packageName = ste.getClassName().takeWhile(_ != '.')
      blacklistPackages.contains(packageName)
    }

    val trimmedLeft = throwable.getStackTrace().view.dropWhile(isBlacklisted)
    val trimmedReverse = trimmedLeft.reverse
      .dropWhile(ste => !ste.getClassName.startsWith(builderName))
      .dropWhile(isBlacklisted)
    trimmedReverse.reverse.toArray
  }

  /** trims the top of the stack of elements belonging to [[blacklistPackages]]
    * then trims the bottom elements until it reaches [[builderName]]
    * then continues trimming elements belonging to [[blacklistPackages]]
    */
  @deprecated("This method will be removed in 3.4", "3.3")
  def trimmedStackTrace: Array[StackTraceElement] = trimmedStackTrace(this)

  def chiselStackTrace: String = {
    val trimmed = trimmedStackTrace(likelyCause)

    val sw = new java.io.StringWriter
    sw.write(likelyCause.toString + "\n")
    sw.write("\t...\n")
    trimmed.foreach(ste => sw.write(s"\tat $ste\n"))
    sw.write("\t... (Stack trace trimmed to user code only, rerun with --full-stacktrace if you wish to see the full stack trace)\n")
    sw.toString
  }
}

private[chisel3] object throwException {
  def apply(s: String, t: Throwable = null): Nothing =
    throw new ChiselException(s, t)
}

/** Records and reports runtime errors and warnings. */
private[chisel3] object ErrorLog {
  val depTag = s"[${Console.BLUE}deprecated${Console.RESET}]"
  val warnTag = s"[${Console.YELLOW}warn${Console.RESET}]"
  val errTag = s"[${Console.RED}error${Console.RESET}]"
}

private[chisel3] class ErrorLog {
  /** Log an error message */
  def error(m: => String): Unit =
    errors += new Error(m, getUserLineNumber)

  /** Log a warning message */
  def warning(m: => String): Unit =
    errors += new Warning(m, getUserLineNumber)

  /** Emit an informational message */
  @deprecated("This method will be removed in 3.5", "3.4")
  def info(m: String): Unit =
    println(new Info("[%2.3f] %s".format(elapsedTime/1e3, m), None))

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
    deprecations.foreach { case ((message, sourceLoc), count) =>
      println(s"${ErrorLog.depTag} $sourceLoc ($count calls): $message")
    }
    errors foreach println

    if (!deprecations.isEmpty) {
      println(s"${ErrorLog.warnTag} ${Console.YELLOW}There were ${deprecations.size} deprecated function(s) used." +
          s" These may stop compiling in a future release - you are encouraged to fix these issues.${Console.RESET}")
      println(s"${ErrorLog.warnTag} Line numbers for deprecations reported by Chisel may be inaccurate; enable scalac compiler deprecation warnings via either of the following methods:")
      println(s"${ErrorLog.warnTag}   In the sbt interactive console, enter:")
      println(s"""${ErrorLog.warnTag}     set scalacOptions in ThisBuild ++= Seq("-unchecked", "-deprecation")""")
      println(s"${ErrorLog.warnTag}   or, in your build.sbt, add the line:")
      println(s"""${ErrorLog.warnTag}     scalacOptions := Seq("-unchecked", "-deprecation")""")
    }

    val allErrors = errors.filter(_.isFatal)
    val allWarnings = errors.filter(!_.isFatal)

    if (!allWarnings.isEmpty && !allErrors.isEmpty) {
      println(s"${ErrorLog.errTag} There were ${Console.RED}${allErrors.size} error(s)${Console.RESET} and ${Console.YELLOW}${allWarnings.size} warning(s)${Console.RESET} during hardware elaboration.")
    } else if (!allWarnings.isEmpty) {
      println(s"${ErrorLog.warnTag} There were ${Console.YELLOW}${allWarnings.size} warning(s)${Console.RESET} during hardware elaboration.")
    } else if (!allErrors.isEmpty) {
      println(s"${ErrorLog.errTag} There were ${Console.RED}${allErrors.size} error(s)${Console.RESET} during hardware elaboration.")
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
          "chisel3.",
          "chisel3.internal.",
          "chisel3.experimental.",
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
