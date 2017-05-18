// See LICENSE for license details.

package loggertests

import logger.Logger.OutputCaptor
import logger.{LazyLogging, LogLevel, Logger}
import org.scalatest.{FreeSpec, Matchers, OneInstancePerTest}

object LoggerSpec {
  val ErrorMsg = "message error"
  val WarnMsg = "message warn"
  val InfoMsg = "message info"
  val DebugMsg = "message debug"
  val TraceMsg = "message trace"
}

class Logger1 extends LazyLogging {
  def run(): Unit = {
    logger.error(LoggerSpec.ErrorMsg)
    logger.warn(LoggerSpec.WarnMsg)
    logger.info(LoggerSpec.InfoMsg)
    logger.debug(LoggerSpec.DebugMsg)
    logger.trace(LoggerSpec.TraceMsg)
  }
}

class LogsInfo2 extends LazyLogging {
  def run(): Unit = {
    logger.info("logger2")
  }
}
class LogsInfo3 extends LazyLogging {
  def run(): Unit = {
    logger.info("logger3")
  }
}
class LoggerSpec extends FreeSpec with Matchers with OneInstancePerTest with LazyLogging {
  "Logger is a simple but powerful logging system" - {
    "Following tests show how global level can control logging" - {
      "only error shows up by default" in {
        Logger.makeScope() {
          val captor = new OutputCaptor
          Logger.setOutput(captor.printStream)

          val r1 = new Logger1
          r1.run()
          val messagesLogged = captor.getOutputAsString

          messagesLogged.contains(LoggerSpec.ErrorMsg) should be(true)
          messagesLogged.contains(LoggerSpec.WarnMsg) should be(false)
          messagesLogged.contains(LoggerSpec.InfoMsg) should be(false)
          messagesLogged.contains(LoggerSpec.DebugMsg) should be(false)
          messagesLogged.contains(LoggerSpec.TraceMsg) should be(false)
        }
      }

      "setting level to warn will result in error and warn messages" in {
        Logger.makeScope() {
          val captor = new OutputCaptor
          Logger.setOutput(captor.printStream)
          Logger.setLevel(LogLevel.Warn)

          val r1 = new Logger1
          r1.run()
          val messagesLogged = captor.getOutputAsString

          messagesLogged.contains(LoggerSpec.ErrorMsg) should be(true)
          messagesLogged.contains(LoggerSpec.WarnMsg) should be(true)
          messagesLogged.contains(LoggerSpec.InfoMsg) should be(false)
          messagesLogged.contains(LoggerSpec.DebugMsg) should be(false)
        }
      }
      "setting level to info will result in error, info, and warn messages" in {
        Logger.makeScope() {
          val captor = new OutputCaptor
          Logger.setOutput(captor.printStream)
          Logger.setLevel(LogLevel.Info)

          val r1 = new Logger1
          r1.run()
          val messagesLogged = captor.getOutputAsString

          messagesLogged.contains(LoggerSpec.ErrorMsg) should be(true)
          messagesLogged.contains(LoggerSpec.WarnMsg) should be(true)
          messagesLogged.contains(LoggerSpec.InfoMsg) should be(true)
          messagesLogged.contains(LoggerSpec.DebugMsg) should be(false)
        }
      }
      "setting level to debug will result in error, info, debug, and warn messages" in {
        Logger.makeScope() {
          val captor = new OutputCaptor
          Logger.setOutput(captor.printStream)

          Logger.setLevel(LogLevel.Error)
          Logger.setOutput(captor.printStream)
          Logger.setLevel(LogLevel.Debug)

          val r1 = new Logger1
          r1.run()
          val messagesLogged = captor.getOutputAsString

          messagesLogged.contains(LoggerSpec.ErrorMsg) should be(true)
          messagesLogged.contains(LoggerSpec.WarnMsg) should be(true)
          messagesLogged.contains(LoggerSpec.InfoMsg) should be(true)
          messagesLogged.contains(LoggerSpec.DebugMsg) should be(true)
          messagesLogged.contains(LoggerSpec.TraceMsg) should be(false)
        }
      }
      "setting level to trace will result in error, info, debug, trace, and warn messages" in {
        Logger.makeScope() {
          val captor = new OutputCaptor
          Logger.setOutput(captor.printStream)

          Logger.setLevel(LogLevel.Error)
          Logger.setOutput(captor.printStream)
          Logger.setLevel(LogLevel.Trace)

          val r1 = new Logger1
          r1.run()
          val messagesLogged = captor.getOutputAsString

          messagesLogged.contains(LoggerSpec.ErrorMsg) should be(true)
          messagesLogged.contains(LoggerSpec.WarnMsg) should be(true)
          messagesLogged.contains(LoggerSpec.InfoMsg) should be(true)
          messagesLogged.contains(LoggerSpec.DebugMsg) should be(true)
          messagesLogged.contains(LoggerSpec.TraceMsg) should be(true)
        }
      }
    }
    "the following tests show how logging can be controlled by package and class name" - {
      "only capture output by class name" - {
        "capture logging from LogsInfo2" in {
          Logger.makeScope() {
            val captor = new OutputCaptor
            Logger.setOutput(captor.printStream)

            Logger.setLevel("loggertests.LogsInfo2", LogLevel.Info)

            val r2 = new LogsInfo2
            val r3 = new LogsInfo3
            r3.run()
            r2.run()

            val messagesLogged = captor.getOutputAsString

            messagesLogged.contains("logger3") should be(false)
            messagesLogged.contains("logger2") should be(true)
          }
        }
        "capture logging from LogsInfo2 using class" in {
          Logger.makeScope() {
            val captor = new OutputCaptor
            Logger.setOutput(captor.printStream)

            Logger.setLevel(classOf[LogsInfo2], LogLevel.Info)

            val r2 = new LogsInfo2
            val r3 = new LogsInfo3
            r3.run()
            r2.run()

            val messagesLogged = captor.getOutputAsString

            messagesLogged.contains("logger3") should be(false)
            messagesLogged.contains("logger2") should be(true)
          }
        }
        "capture logging from LogsInfo3" in {
          Logger.makeScope() {
            val captor = new OutputCaptor
            Logger.setOutput(captor.printStream)

            Logger.setLevel("loggertests.LogsInfo3", LogLevel.Info)

            val r2 = new LogsInfo2
            val r3 = new LogsInfo3
            r2.run()
            r3.run()

            val messagesLogged = captor.getOutputAsString

            messagesLogged.contains("logger2") should be(false)
            messagesLogged.contains("logger3") should be(true)
          }
        }
      }
      "log based on package name" - {
        "both log because of package, also showing re-run after change works" in {
          Logger.makeScope() {
            val captor = new OutputCaptor
            Logger.setOutput(captor.printStream)

            Logger.setLevel(LogLevel.Error)
            Logger.setLevel("loggertests", LogLevel.Error)

            val r2 = new LogsInfo2
            val r3 = new LogsInfo3
            r2.run()
            r3.run()

            var messagesLogged = captor.getOutputAsString

            messagesLogged.contains("logger2") should be(false)
            messagesLogged.contains("logger3") should be(false)

            Logger.setLevel("loggertests", LogLevel.Debug)

            r2.run()
            r3.run()

            messagesLogged = captor.getOutputAsString

            messagesLogged.contains("logger2") should be(true)
            messagesLogged.contains("logger3") should be(true)
          }
        }
      }
      "check for false positives" in {
        Logger.makeScope() {
          val captor = new OutputCaptor
          Logger.setOutput(captor.printStream)

          Logger.setLevel("bad-loggertests", LogLevel.Info)

          val r2 = new LogsInfo2
          val r3 = new LogsInfo3
          r2.run()
          r3.run()

          val messagesLogged = captor.getOutputAsString

          messagesLogged.contains("logger2") should be(false)
          messagesLogged.contains("logger3") should be(false)
        }
      }
      "show that class specific level supercedes global level" in {
        Logger.makeScope() {
          val captor = new OutputCaptor
          Logger.setOutput(captor.printStream)


          Logger.setLevel(LogLevel.Info)
          Logger.setLevel("loggertests.LogsInfo2", LogLevel.Error)

          val r2 = new LogsInfo2
          val r3 = new LogsInfo3
          r2.run()
          r3.run()

          val messagesLogged = captor.getOutputAsString

          messagesLogged.contains("logger2") should be(false)
          messagesLogged.contains("logger3") should be(true)
        }
      }
      "Show logging can be set with command options" in {
        val captor = new Logger.OutputCaptor

        Logger.makeScope(Array("--class-log-level", "loggertests.LogsInfo3:info")) {
          Logger.setOutput(captor.printStream)
          val r2 = new LogsInfo2
          val r3 = new LogsInfo3
          r2.run()
          r3.run()

          val messagesLogged = captor.getOutputAsString

          messagesLogged.contains("logger2") should be(false)
          messagesLogged.contains("logger3") should be(true)
        }
      }
      "Show that printstream remains across makeScopes" in {
        Logger.makeScope() {
          val captor = new Logger.OutputCaptor
          Logger.setOutput(captor.printStream)

          logger.error("message 1")
          Logger.makeScope() {
            logger.error("message 2")
          }

          val logText = captor.getOutputAsString
          logText should include ("message 1")
          logText should include ("message 2")
        }
      }
      "Show that nested makeScopes share same state" in {
        Logger.getGlobalLevel should be (LogLevel.None)

        Logger.makeScope() {
          Logger.setLevel(LogLevel.Info)

          Logger.getGlobalLevel should be (LogLevel.Info)

          Logger.makeScope() {
            Logger.getGlobalLevel should be (LogLevel.Info)
          }

          Logger.makeScope() {
            Logger.setLevel(LogLevel.Debug)
            Logger.getGlobalLevel should be (LogLevel.Debug)
          }

          Logger.getGlobalLevel should be (LogLevel.Debug)
        }

        Logger.getGlobalLevel should be (LogLevel.None)
      }

      "Show that first makeScope starts with fresh state" in {
        Logger.getGlobalLevel should be (LogLevel.None)

        Logger.setLevel(LogLevel.Warn)
        Logger.getGlobalLevel should be (LogLevel.Warn)

        Logger.makeScope() {
          Logger.getGlobalLevel should be (LogLevel.None)

          Logger.setLevel(LogLevel.Trace)
          Logger.getGlobalLevel should be (LogLevel.Trace)
        }

        Logger.getGlobalLevel should be (LogLevel.Warn)
      }
    }
  }
}
