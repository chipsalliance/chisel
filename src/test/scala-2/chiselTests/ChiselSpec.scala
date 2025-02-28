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

/** Utilities for writing property-based checks */
trait PropertyUtils extends ScalaCheckPropertyChecks {

  // Constrain the default number of instances generated for every use of forAll.
  implicit override val generatorDrivenConfig: PropertyCheckConfiguration =
    PropertyCheckConfiguration(minSuccessful = 8, minSize = 1, sizeRange = 3)

  // Generator for small positive integers.
  val smallPosInts = Gen.choose(1, 4)

  // Generator for positive (ascending or descending) ranges.
  def posRange: Gen[Range] = for {
    dir <- Gen.oneOf(true, false)
    step <- Gen.choose(1, 3)
    m <- Gen.choose(1, 10)
    n <- Gen.choose(1, 10)
  } yield {
    if (dir) {
      Range(m, (m + n) * step, step)
    } else {
      Range((m + n) * step, m, -step)
    }
  }

  // Generator for widths considered "safe".
  val safeUIntWidth = Gen.choose(1, 30)

  // Generators for integers that fit within "safe" widths.
  val safeUInts = Gen.choose(0, (1 << 30))

  // Generators for vector sizes.
  val vecSizes = Gen.choose(0, 4)

  // Generator for string representing an arbitrary integer.
  val binaryString = for (i <- Arbitrary.arbitrary[Int]) yield "b" + i.toBinaryString

  // Generator for a sequence of Booleans of size n.
  def enSequence(n: Int): Gen[List[Boolean]] = Gen.containerOfN[List, Boolean](n, Gen.oneOf(true, false))

  // Generator which gives a width w and a list (of size n) of numbers up to w bits.
  def safeUIntN(n: Int): Gen[(Int, List[Int])] = for {
    w <- smallPosInts
    i <- Gen.containerOfN[List, Int](n, Gen.choose(0, (1 << w) - 1))
  } yield (w, i)

  // Generator which gives a width w and a numbers up to w bits.
  val safeUInt = for {
    w <- smallPosInts
    i <- Gen.choose(0, (1 << w) - 1)
  } yield (w, i)

  // Generator which gives a width w and a list (of size n) of a pair of numbers up to w bits.
  def safeUIntPairN(n: Int): Gen[(Int, List[(Int, Int)])] = for {
    w <- smallPosInts
    i <- Gen.containerOfN[List, Int](n, Gen.choose(0, (1 << w) - 1))
    j <- Gen.containerOfN[List, Int](n, Gen.choose(0, (1 << w) - 1))
  } yield (w, i.zip(j))

  // Generator which gives a width w and a pair of numbers up to w bits.
  val safeUIntPair = for {
    w <- smallPosInts
    i <- Gen.choose(0, (1 << w) - 1)
    j <- Gen.choose(0, (1 << w) - 1)
  } yield (w, i, j)

}

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
