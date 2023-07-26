// SPDX-License-Identifier: Apache-2.0

package firrtl.testutils

import java.io._
import java.security.Permission
import scala.sys.process._

import logger.{LazyLogging, LogLevel, LogLevelAnnotation}

import org.scalatest._
import org.scalatestplus.scalacheck._

import firrtl._
import firrtl.ir._
import firrtl.Parser.UseInfo
import firrtl.options.Dependency
import firrtl.stage.InfoModeAnnotation
import firrtl.annotations._
import firrtl.transforms.{DontTouchAnnotation, NoDedupAnnotation}
import firrtl.renamemap.MutableRenameMap
import firrtl.util.BackendCompilationUtilities
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.propspec.AnyPropSpec

trait FirrtlMatchers extends Matchers

abstract class FirrtlPropSpec extends AnyPropSpec with ScalaCheckPropertyChecks with LazyLogging

abstract class FirrtlFlatSpec extends AnyFlatSpec with FirrtlMatchers with LazyLogging

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

}
