// See LICENSE for license details.

package logger

import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{HasShellOptions, ShellOption, StageUtils}

import scopt.OptionParser

/** An annotation associated with a Logger command line option */
sealed trait LoggerOption { this: Annotation => }

/** Describes the verbosity of information to log
  *  - set with `-ll/--log-level`
  *  - if unset, a [[LogLevelAnnotation]] with the default log level will be emitted
  * @param level the level of logging
  */
case class LogLevelAnnotation(globalLogLevel: LogLevel.Value = LogLevel.None) extends NoTargetAnnotation with LoggerOption

object LogLevelAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "log-level",
      toAnnotationSeq = (a: String) => Seq(LogLevelAnnotation(LogLevel(a))),
      helpText = s"Set global logging verbosity (default: ${new LoggerOptions().globalLogLevel}",
      shortOption = Some("ll"),
      helpValueName = Some("{error|warn|info|debug|trace}") ) )

}

/** Describes a mapping of a class to a specific log level
  *  - set with `-cll/--class-log-level`
  * @param name the class name to log
  * @param level the verbosity level
  */
case class ClassLogLevelAnnotation(className: String, level: LogLevel.Value) extends NoTargetAnnotation with LoggerOption

object ClassLogLevelAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[Seq[String]](
      longOption = "class-log-level",
      toAnnotationSeq = (a: Seq[String]) => a.map { aa =>
        val className :: levelName :: _ = aa.split(":").toList
        val level = LogLevel(levelName)
        ClassLogLevelAnnotation(className, level) },
      helpText = "Set per-class logging verbosity",
      shortOption = Some("cll"),
      helpValueName = Some("<FullClassName:{error|warn|info|debug|trace}>...") ) )

}

/** Enables logging to a file (as opposed to STDOUT)
  *  - maps to [[LoggerOptions.logFileName]]
  *  - enabled with `--log-file`
  */
case class LogFileAnnotation(file: Option[String]) extends NoTargetAnnotation with LoggerOption

object LogFileAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "log-file",
      toAnnotationSeq = (a: String) => Seq(LogFileAnnotation(Some(a))),
      helpText = "Log to a file instead of STDOUT",
      helpValueName = Some("<file>") ) )

}

/** Enables class names in log output
  *  - enabled with `-lcn/--log-class-names`
  */
case object LogClassNamesAnnotation extends NoTargetAnnotation with LoggerOption with HasShellOptions {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "log-class-names",
      toAnnotationSeq = (a: Unit) => Seq(LogClassNamesAnnotation),
      helpText = "Show class names and log level in logging output",
      shortOption = Some("lcn") ) )

}
