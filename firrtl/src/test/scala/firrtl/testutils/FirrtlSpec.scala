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

  /** Encodes a System.exit exit code
    * @param status the exit code
    */
  private case class ExitException(status: Int) extends SecurityException(s"Found a sys.exit with code $status")

  /** A security manager that converts calls to System.exit into [[ExitException]]s by explicitly disabling the ability of
    * a thread to actually exit. For more information, see:
    *   - https://docs.oracle.com/javase/tutorial/essential/environment/security.html
    */
  private class ExceptOnExit extends SecurityManager {
    override def checkPermission(perm: Permission): Unit = {}
    override def checkPermission(perm: Permission, context: Object): Unit = {}
    override def checkExit(status: Int): Unit = {
      super.checkExit(status)
      throw ExitException(status)
    }
  }

  /** Encodes a file that some code tries to write to
    * @param the file name
    */
  private case class WriteException(file: String) extends SecurityException(s"Tried to write to file $file")

  /** A security manager that converts writes to any file into [[WriteException]]s.
    */
  private class ExceptOnWrite extends SecurityManager {
    override def checkPermission(perm: Permission): Unit = {}
    override def checkPermission(perm: Permission, context: Object): Unit = {}
    override def checkWrite(file: String): Unit = {
      super.checkWrite(file)
      throw WriteException(file)
    }
  }

  /** Run some Scala code (a thunk) in an environment where all System.exit are caught and returned. This avoids a
    * situation where a test results in something actually exiting and killing the entire test. This is necessary if you
    * want to test a command line program, e.g., the `main` method of [[firrtl.options.Stage Stage]].
    *
    * NOTE: THIS WILL NOT WORK IN SITUATIONS WHERE THE THUNK IS CATCHING ALL [[Exception]]s OR [[Throwable]]s, E.G.,
    * SCOPT. IF THIS IS HAPPENING THIS WILL NOT WORK. REPEAT THIS WILL NOT WORK.
    * @param thunk some Scala code
    * @return either the output of the thunk (`Right[T]`) or an exit code (`Left[Int]`)
    */
  def catchStatus[T](thunk: => T): Either[Int, T] = {
    try {
      System.setSecurityManager(new ExceptOnExit())
      Right(thunk)
    } catch {
      case ExitException(a) => Left(a)
    } finally {
      System.setSecurityManager(null)
    }
  }

  /** Run some Scala code (a thunk) in an environment where file writes are caught and the file that a program tries to
    * write to is returned. This is useful if you want to test that some thunk either tries to write to a specific file
    * or doesn't try to write at all.
    */
  def catchWrites[T](thunk: => T): Either[String, T] = {
    try {
      System.setSecurityManager(new ExceptOnWrite())
      Right(thunk)
    } catch {
      case WriteException(a) => Left(a)
    } finally {
      System.setSecurityManager(null)
    }
  }
}
