// SPDX-License-Identifier: Apache-2.0

package chiselTests

import java.io.{ByteArrayOutputStream, PrintStream}
import logger.{LogLevel, LogLevelAnnotation, Logger}
import firrtl.{annoSeqToSeq, seqToAnnoSeq, AnnotationSeq}

trait LogUtils {

  /** Run some Scala thunk and return STDOUT and STDERR as strings.
    * @param thunk some Scala code
    * @return a tuple containing STDOUT, STDERR, and what the thunk returns
    */
  def grabStdOutErr[T](thunk: => T): (String, String, T) = {
    val stdout, stderr = new ByteArrayOutputStream()
    val ret = scala.Console.withOut(stdout) { scala.Console.withErr(stderr) { thunk } }
    (stdout.toString, stderr.toString, ret)
  }

  /** Run some Scala thunk and return all logged messages as Strings
    * @param thunk some Scala code
    * @return a tuple containing LOGGED, and what the thunk returns
    */
  def grabLog[T](thunk: => T): (String, T) = grabLogLevel(LogLevel.default)(thunk)

  /** Run some Scala thunk and return all logged messages as Strings
    * @param level the log level to use
    * @param thunk some Scala code
    * @return a tuple containing LOGGED, and what the thunk returns
    */
  def grabLogLevel[T](level: LogLevel.Value)(thunk: => T): (String, T) = {
    val baos = new ByteArrayOutputStream()
    val stream = new PrintStream(baos, true, "utf-8")
    val ret = Logger.makeScope(LogLevelAnnotation(level) :: Nil) {
      Logger.setOutput(stream)
      thunk
    }
    (baos.toString, ret)
  }
}
