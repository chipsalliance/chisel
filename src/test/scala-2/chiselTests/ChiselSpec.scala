// SPDX-License-Identifier: Apache-2.0

package chiselTests

import _root_.logger.{LogLevel, LogLevelAnnotation, Logger}
import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, PrintFullStackTraceAnnotation}
import chisel3.testers._
import circt.stage.{CIRCTTarget, CIRCTTargetAnnotation, ChiselStage}
import chisel3.simulator._
import svsim._
import firrtl.annotations.Annotation
import firrtl.ir.Circuit
import firrtl.stage.FirrtlCircuitAnnotation
import firrtl.{AnnotationSeq, EmittedVerilogCircuitAnnotation}
import org.scalacheck._
import org.scalatest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.propspec.AnyPropSpec
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import org.scalactic.source.Position

import java.io.{ByteArrayOutputStream, PrintStream}
import java.security.Permission
import scala.reflect.ClassTag
import java.text.SimpleDateFormat
import java.util.Calendar
import chisel3.reflect.DataMirror

trait Utils {

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
