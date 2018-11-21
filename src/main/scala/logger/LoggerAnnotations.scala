// See LICENSE for license details.

package logger

import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{HasScoptOptions, StageUtils}

import scopt.OptionParser

/** An annotation associated with a Logger command line option */
sealed trait LoggerOption { this: Annotation => }

/** Describes the verbosity of information to log
  *  - set with `-ll/--log-level`
  *  - if unset, a [[LogLevelAnnotation]] with the default log level will be emitted
  * @param level the level of logging
  */
case class LogLevelAnnotation(globalLogLevel: LogLevel.Value = LogLevel.None) extends NoTargetAnnotation with LoggerOption

object LogLevelAnnotation extends HasScoptOptions {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[String]("log-level")
    .abbr("ll")
    .valueName("<Error|Warn|Info|Debug|Trace>")
    .action( (x, c) => LogLevelAnnotation(LogLevel(x)) +: c )
    .validate{ x =>
      lazy val msg = s"$x bad value must be one of error|warn|info|debug|trace"
      if (Array("error", "warn", "info", "debug", "trace").contains(x.toLowerCase)) { p.success      }
      else                                                                          { p.failure(msg) }}
    .unbounded()
    .text(s"Sets the verbosity level of logging, default is ${new LoggerOptions().globalLogLevel}")
}

/** Describes a mapping of a class to a specific log level
  *  - set with `-cll/--class-log-level`
  * @param name the class name to log
  * @param level the verbosity level
  */
case class ClassLogLevelAnnotation(className: String, level: LogLevel.Value) extends NoTargetAnnotation with LoggerOption

object ClassLogLevelAnnotation extends HasScoptOptions {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[Seq[String]]("class-log-level")
    .abbr("cll")
    .valueName("<FullClassName:[Error|Warn|Info|Debug|Trace]>[,...]")
    .action( (x, c) => (x.map { y =>
                          val className :: levelName :: _ = y.split(":").toList
                          val level = LogLevel(levelName)
                          ClassLogLevelAnnotation(className, level) }) ++ c )
    .unbounded()
    .text(s"This defines per-class verbosity of logging")
}

/** Enables logging to a file (as opposed to STDOUT)
  *  - maps to [[LoggerOptions.logFileName]]
  *  - enabled with `--log-file`
  */
case class LogFileAnnotation(file: Option[String]) extends NoTargetAnnotation with LoggerOption

object LogFileAnnotation extends HasScoptOptions {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = {
    p.opt[String]("log-file")
      .action( (x, c) => LogFileAnnotation(Some(x)) +: c )
      .unbounded()
      .text(s"log to the specified file")
  }
}

/** Enables class names in log output
  *  - enabled with `-lcn/--log-class-names`
  */
case object LogClassNamesAnnotation extends NoTargetAnnotation with LoggerOption with HasScoptOptions {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[Unit]("log-class-names")
    .abbr("lcn")
    .action( (x, c) => LogClassNamesAnnotation +: c )
    .unbounded()
    .text(s"shows class names and log level in logging output, useful for target --class-log-level")
}
