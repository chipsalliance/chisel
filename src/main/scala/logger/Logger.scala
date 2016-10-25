// See LICENSE for license details.

package logger

import java.io.{PrintStream, File, FileOutputStream}

import firrtl.ExecutionOptionsManager

/**
  * This provides a facility for a log4scala* type logging system.  Why did we write our own?  Because
  * the canned ones are just to darned hard to turn on, particularly when embedded in a distribution.
  * This one can be turned on programmatically or with the options exposed in the [[firrtl.CommonOptions]]
  * and [[ExecutionOptionsManager]] API's in firrtl.
  * There are 4 main options.
  *  * a simple global option to turn on all in scope (and across threads, might want to fix this)
  *  * turn on specific levels for specific fully qualified class names
  *  * set a file to write things to, default is just to use stdout
  *  * include the class names and level in the output.  This is useful to figure out what
  *  the class names that extend LazyLogger are
  *
  *  This is not overly optimized but does pass the string as () => String to avoid string interpolation
  *  occurring if the the logging level is not sufficiently high. This could be further optimized by playing
  *  with methods
  */
/**
  * The supported log levels, what do they mean? Whatever you want them to.
  */
object LogLevel extends Enumeration {
  val Error, Warn, Info, Debug, Trace = Value
}

/**
  * extend this trait to enable logging in a class you are implementing
  */
trait LazyLogging {
  val logger = new Logger(this.getClass.getName)
}

/**
  * Singleton in control of what is supposed to get logged, how it's to be logged and where it is to be logged
  */
object Logger {
  var globalLevel = LogLevel.Error
  val classLevels = new scala.collection.mutable.HashMap[String, LogLevel.Value]
  var logClassNames = false

  def showMessage(level: LogLevel.Value, className: String, message: => String): Unit = {
    if(globalLevel == level || (classLevels.nonEmpty && classLevels.getOrElse(className, LogLevel.Error) >= level)) {
      if(logClassNames) {
        stream.println(s"[$level:$className] $message")
      }
      else {
        stream.println(message)
      }
    }
  }

  var stream: PrintStream = System.out

  def setOutput(fileName: String): Unit = {
    stream = new PrintStream(new FileOutputStream(new File(fileName)))
  }
  def setConsole(): Unit = {
    stream = Console.out
  }
  def setClassLogLevels(namesToLevel: Map[String, LogLevel.Value]): Unit = {
    classLevels ++= namesToLevel
  }

  def setOptions(optionsManager: ExecutionOptionsManager): Unit = {
    val commonOptions = optionsManager.commonOptions
    globalLevel = commonOptions.globalLogLevel
    setClassLogLevels(commonOptions.classLogLevels)
    if(commonOptions.logToFile) {
      setOutput(commonOptions.getLogFileName(optionsManager))
    }
    logClassNames = commonOptions.logClassNames
  }
}

/**
  * Classes implementing [[LazyLogging]] will have logger of this type
  * @param containerClass  passed in from the LazyLogging trait in order to provide class level logging granularity
  */
class Logger(containerClass: String) {
  def error(message: => String): Unit = {
    Logger.showMessage(LogLevel.Error, containerClass, message)
  }
  def warn(message: => String): Unit = {
    Logger.showMessage(LogLevel.Warn, containerClass, message)
  }
  def info(message: => String): Unit = {
    Logger.showMessage(LogLevel.Info, containerClass, message)
  }
  def debug(message: => String): Unit = {
    Logger.showMessage(LogLevel.Debug, containerClass, message)
  }
  def trace(message: => String): Unit = {
    Logger.showMessage(LogLevel.Trace, containerClass, message)
  }
}
