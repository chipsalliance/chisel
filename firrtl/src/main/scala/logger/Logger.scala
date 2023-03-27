// SPDX-License-Identifier: Apache-2.0

package logger

import java.io.{ByteArrayOutputStream, File, FileOutputStream, PrintStream}

import firrtl.AnnotationSeq
import firrtl.options.Viewer.view
import logger.phases.{AddDefaults, Checks}

import scala.util.DynamicVariable

/**
  * This provides a facility for a log4scala* type logging system.  Why did we write our own?  Because
  * the canned ones are just too darned hard to turn on, particularly when embedded in a distribution.
  * There are 4 main options:
  *  * A simple global option to turn on all in scope (and across threads, might want to fix this)
  *  * Turn on specific levels for specific fully qualified class names
  *  * Set a file to write things to, default is just to use stdout
  *  * Include the class names and level in the output. This is useful for figuring out what
  *    the class names that extend LazyLogger are.
  *
  *  This is not overly optimized but does pass the string as () => String to avoid string interpolation
  *  occurring if the the logging level is not sufficiently high. This could be further optimized by playing
  *  with methods.
  */
/**
  * The supported log levels, what do they mean? Whatever you want them to.
  */
object LogLevel extends Enumeration {
  // None indicates "not set"
  val Error, Warn, Info, Debug, Trace, None = Value

  def default = Warn

  def apply(s: String): LogLevel.Value = s.toLowerCase match {
    case "error" => LogLevel.Error
    case "warn"  => LogLevel.Warn
    case "info"  => LogLevel.Info
    case "debug" => LogLevel.Debug
    case "trace" => LogLevel.Trace
    case level   => throw new Exception(s"Unknown LogLevel '$level'")
  }
}

/**
  * extend this trait to enable logging in a class you are implementing
  */
trait LazyLogging {
  protected val logger = new Logger(this.getClass.getName)
  def getLogger: Logger = logger
}

/**
  * Mutable state of the logging system.  Multiple LoggerStates may be present
  * when used in multi-threaded environments
  */
private class LoggerState {
  var globalLevel = LogLevel.None
  val classLevels = new scala.collection.mutable.HashMap[String, LogLevel.Value]
  val classToLevelCache = new scala.collection.mutable.HashMap[String, LogLevel.Value]
  var logClassNames = false
  var stream:             PrintStream = Console.out
  var fromInvoke:         Boolean = false // this is used to not have invokes re-create run-state
  var stringBufferOption: Option[Logger.OutputCaptor] = None

  override def toString: String = {
    s"gl $globalLevel classLevels ${classLevels.mkString("\n")}"
  }

  /**
    * create a new state object copying the basic values of this state
    * @return new state object
    */
  def copy: LoggerState = {
    val newState = new LoggerState
    newState.globalLevel = this.globalLevel
    newState.classLevels ++= this.classLevels
    newState.stream = this.stream
    newState.logClassNames = this.logClassNames
    newState
  }
}

/**
  * Singleton in control of what is supposed to get logged, how it's to be logged and where it is to be logged
  * We uses a dynamic variable in case multiple threads are used as can be in scalatests
  */
object Logger {
  private val updatableLoggerState = new DynamicVariable[Option[LoggerState]](Some(new LoggerState))
  private def state: LoggerState = {
    updatableLoggerState.value.get
  }

  /**
    * a class for managing capturing logging output in a string buffer
    */
  class OutputCaptor {
    val byteArrayOutputStream = new ByteArrayOutputStream()
    val printStream = new PrintStream(byteArrayOutputStream)

    /**
      * Get logged messages to this captor as a string
      * @return
      */
    def getOutputAsString: String = {
      byteArrayOutputStream.toString
    }

    /**
      * Clear the string buffer
      */
    def clear(): Unit = {
      byteArrayOutputStream.reset()
    }
  }

  /** Set a scope for this logger based on available annotations
    * @param options a sequence annotations
    * @param codeBlock some Scala code over which to define this scope
    * @tparam A return type of the code block
    * @return the original return of the code block
    */
  def makeScope[A](options: AnnotationSeq = Seq.empty)(codeBlock: => A): A = {
    val runState: LoggerState = {
      val newRunState = updatableLoggerState.value.getOrElse(new LoggerState)
      if (newRunState.fromInvoke) {
        newRunState
      } else {
        val forcedNewRunState = new LoggerState
        forcedNewRunState.fromInvoke = true
        forcedNewRunState
      }
    }
    updatableLoggerState.withValue(Some(runState)) {
      setOptions(options)
      codeBlock
    }
  }

  /**
    * Used to test whether a given log statement should generate some logging output.
    * It breaks up a class name into a list of packages.  From this list generate progressively
    * broader names (lopping off from right side) checking for a match
    * @param className  class name that the logging statement came from
    * @param level  the level of the log statement being evaluated
    * @return
    */
  private def testPackageNameMatch(className: String, level: LogLevel.Value): Option[Boolean] = {
    val classLevels = state.classLevels
    if (classLevels.isEmpty) return None

    // If this class name in cache just use that value
    val levelForThisClassName = state.classToLevelCache.getOrElse(
      className, {
        // otherwise break up the class name in to full package path as list and find most specific entry you can
        val packageNameList = className.split("""\.""").toList
        /*
         * start with full class path, lopping off from the tail until nothing left
         */
        def matchPathToFindLevel(packageList: List[String]): LogLevel.Value = {
          if (packageList.isEmpty) {
            LogLevel.None
          } else {
            val partialName = packageList.mkString(".")
            val level = classLevels.getOrElse(
              partialName, {
                matchPathToFindLevel(packageList.reverse.tail.reverse)
              }
            )
            level
          }
        }

        val levelSpecified = matchPathToFindLevel(packageNameList)
        if (levelSpecified != LogLevel.None) {
          state.classToLevelCache(className) = levelSpecified
        }
        levelSpecified
      }
    )

    if (levelForThisClassName != LogLevel.None) {
      Some(levelForThisClassName >= level)
    } else {
      None
    }
  }

  /**
    * Used as the common log routine, for warn, debug etc.  Only calls message if log should be generated
    * Allows lazy evaluation of any string interpolation or function that generates the message itself
    * @note package level supercedes global, which allows one to turn on debug everywhere except for specific classes
    * @param level     level of the called statement
    * @param className class name of statement
    * @param message   a function returning a string with the message
    */
  private def showMessage(level: LogLevel.Value, className: String, message: => String): Unit = {
    def logIt(): Unit = {
      if (state.logClassNames) {
        state.stream.println(s"[$level:$className] $message")
      } else {
        state.stream.println(message)
      }
    }
    testPackageNameMatch(className, level) match {
      case Some(true)  => logIt()
      case Some(false) =>
      case None =>
        if (getGlobalLevel >= level) {
          logIt()
        }
    }
  }

  def getGlobalLevel: LogLevel.Value = state.globalLevel match {
    // None means "not set" so use default in that case
    case LogLevel.None => LogLevel.default
    case other         => other
  }

  /**
    * This resets everything in the current Logger environment, including the destination
    * use this with caution.  Unexpected things can happen
    */
  def reset(): Unit = {
    state.classLevels.clear()
    clearCache()
    state.logClassNames = false
    state.globalLevel = LogLevel.Error
    state.stream = System.out
  }

  /**
    * clears the cache of class names top class specific log levels
    */
  private def clearCache(): Unit = {
    state.classToLevelCache.clear()
  }

  /**
    * This sets the global logging level
    * @param level  The desired global logging level
    */
  def setLevel(level: LogLevel.Value): Unit = {
    state.globalLevel = level
  }

  /**
    * This sets the logging level for a particular class or package
    * The package name must be general to specific.  I.e.
    * package1.package2.class
    * package1.package2
    * package1
    * Will work.
    * package2.class will not work if package2 is within package1
    * @param classOrPackageName The string based class name or
    * @param level  The desired global logging level
    */
  def setLevel(classOrPackageName: String, level: LogLevel.Value): Unit = {
    clearCache()
    state.classLevels(classOrPackageName) = level
  }

  /**
    * Set the log level based on a class type
    * @example {{{ setLevel(classOf[SomeClass], LogLevel.Debug) }}}
    * @param classType Kind of class
    * @param level log level to set
    */
  def setLevel(classType: Class[_ <: LazyLogging], level: LogLevel.Value): Unit = {
    clearCache()
    val name = classType.getCanonicalName
    state.classLevels(name) = level
  }

  /**
    * Clears the logging data in the string capture buffer if it exists
    * @return The logging data if it exists
    */
  def clearStringBuffer(): Unit = {
    state.stringBufferOption match {
      case Some(x) => x.byteArrayOutputStream.reset()
      case None    =>
    }
  }

  /**
    * Set the logging destination to a file name
    * @param fileName destination name
    */
  def setOutput(fileName: String): Unit = {
    state.stream = new PrintStream(new FileOutputStream(new File(fileName)))
  }

  /**
    * Set the logging destination to a print stream
    * @param stream destination stream
    */
  def setOutput(stream: PrintStream): Unit = {
    state.stream = stream
  }

  /**
    * Sets the logging destination to Console.out
    */
  def setConsole(): Unit = {
    state.stream = Console.out
  }

  /**
    * Adds a list of of className, loglevel tuples to the global (dynamicVar)
    * See testPackageNameMatch for a description of how class name matching works
    * @param namesToLevel a list of tuples (class name, log level)
    */
  def setClassLogLevels(namesToLevel: Map[String, LogLevel.Value]): Unit = {
    clearCache()
    state.classLevels ++= namesToLevel
  }

  /** Set logger options based on the content of an [[firrtl.AnnotationSeq AnnotationSeq]]
    * @param inputAnnotations annotation sequence containing logger options
    */
  def setOptions(inputAnnotations: AnnotationSeq): Unit = {
    val annotations =
      Seq(new AddDefaults, Checks)
        .foldLeft(inputAnnotations)((a, p) => p.transform(a))

    val lopts = view[LoggerOptions](annotations)
    state.globalLevel = (state.globalLevel, lopts.globalLogLevel) match {
      case (LogLevel.None, LogLevel.None) => LogLevel.None
      case (x, LogLevel.None)             => x
      case (LogLevel.None, x)             => x
      case (_, x)                         => x
    }
    setClassLogLevels(lopts.classLogLevels)

    if (lopts.logFileName.nonEmpty) {
      setOutput(lopts.logFileName.get)
    }

    state.logClassNames = lopts.logClassNames
  }
}

/**
  * Classes implementing [[LazyLogging]] will have logger of this type
  * @param containerClass  passed in from the LazyLogging trait in order to provide class level logging granularity
  */
class Logger(containerClass: String) {

  /**
    * Log message at Error level
    * @param message message generator to be invoked if level is right
    */
  def error(message: => String): Unit = {
    Logger.showMessage(LogLevel.Error, containerClass, message)
  }

  /**
    * Log message at Warn level
    * @param message message generator to be invoked if level is right
    */
  def warn(message: => String): Unit = {
    Logger.showMessage(LogLevel.Warn, containerClass, message)
  }

  /**
    * Log message at Inof level
    * @param message message generator to be invoked if level is right
    */
  def info(message: => String): Unit = {
    Logger.showMessage(LogLevel.Info, containerClass, message)
  }

  /**
    * Log message at Debug level
    * @param message message generator to be invoked if level is right
    */
  def debug(message: => String): Unit = {
    Logger.showMessage(LogLevel.Debug, containerClass, message)
  }

  /**
    * Log message at Trace level
    * @param message message generator to be invoked if level is right
    */
  def trace(message: => String): Unit = {
    Logger.showMessage(LogLevel.Trace, containerClass, message)
  }
}

/** An exception originating from the Logger
  * @param str an exception message
  * @param cause a reason for the exception
  */
class LoggerException(val str: String, cause: Throwable = null) extends RuntimeException(str, cause)
