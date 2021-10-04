// SPDX-License-Identifier: Apache-2.0

package chiselTests.stage

import chisel3._
import chisel3.stage.ChiselMain
import java.io.File

import chisel3.aop.inspecting.{InspectingAspect, InspectorAspect}
import org.scalatest.GivenWhenThen
import org.scalatest.featurespec.AnyFeatureSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inside._
import org.scalatest.EitherValues._

import scala.io.Source
import firrtl.Parser

object ChiselMainSpec {

  /** A module that connects two different types together resulting in an elaboration error */
  class DifferentTypesModule extends RawModule {
    val in = IO(Input(UInt(1.W)))
    val out = IO(Output(SInt(1.W)))
    out := in
  }

  /** A module that connects two of the same types together */
  class SameTypesModule extends Module {
    val in = IO(Input(UInt(1.W)))
    val out = IO(Output(UInt(1.W)))
    out := in
  }

  /** A module that fails a requirement */
  class FailingRequirementModule extends RawModule {
    require(false, "the user wrote a failing requirement")
  }

  /** A module that triggers a Builder.error (as opposed to exception) */
  class BuilderErrorModule extends RawModule {
    val w = Wire(UInt(8.W))
    w(3, -1)
  }
}

case class TestClassAspect() extends InspectorAspect[RawModule] ({
  _: RawModule => println("Ran inspectingAspect")
})

case object TestObjectAspect extends InspectorAspect[RawModule] ({
  _: RawModule => println("Ran inspectingAspect")
})

class ChiselMainSpec extends AnyFeatureSpec with GivenWhenThen with Matchers with chiselTests.Utils {

  import ChiselMainSpec._

  class ChiselMainFixture {
    Given("a Chisel stage")
    val stage = ChiselMain
  }

  class TargetDirectoryFixture(dirName: String) {
    val dir = new File(s"test_run_dir/ChiselStageSpec/$dirName")
    val buildDir = new File(dir + "/build")
    dir.mkdirs()
  }

  case class ChiselMainTest(
    args: Array[String],
    generator: Option[Class[_ <: RawModule]] = None,
    files: Seq[String] = Seq.empty,
    stdout: Seq[Either[String, String]] = Seq.empty,
    stderr: Seq[Either[String, String]] = Seq.empty,
    result: Int = 0,
    fileChecks: Map[String, File => Unit] = Map.empty) {
    def testName: String = "args" + args.mkString("_")
    def argsString: String = args.mkString(" ")
  }

  /** A test of ChiselMain that is going to involve catching an exception.
    * @param args command line arguments (excluding --module) to pass in
    * @param generator the module to build (used to generate --module)
    * @param message snippets of text that should appear (Right) or not appear (Left) in the exception message
    * @param stdout snippets of text that should appear (Right) or not appear (Left) in STDOUT
    * @param stderr snippets of text that should appear (Right) or not appear (Left) in STDERR
    * @param stackTrace snippets of text that should appear (Right) or not appear (Left) in the stack trace
    * @tparam the type of exception that should occur
    */
  case class ChiselMainExceptionTest[A <: Throwable](
    args: Array[String],
    generator: Option[Class[_ <: RawModule]] = None,
    message: Seq[Either[String, String]] = Seq.empty,
    stdout: Seq[Either[String, String]] = Seq.empty,
    stderr: Seq[Either[String, String]] = Seq.empty,
    stackTrace: Seq[Either[String, String]] = Seq.empty
  ) {
    def testName: String = "args" + args.mkString("_")
    def argsString: String = args.mkString(" ")
  }

  def runStageExpectFiles(p: ChiselMainTest): Unit = {
    Scenario(s"""User runs Chisel Stage with '${p.argsString}'""") {
      val f = new ChiselMainFixture
      val td = new TargetDirectoryFixture(p.testName)

      p.files.foreach( f => new File(td.buildDir + s"/$f").delete() )

      When(s"""the user tries to compile with '${p.argsString}'""")
      val module: Array[String] =
        (if (p.generator.nonEmpty) { Array("--module", p.generator.get.getName) }
         else                      { Array.empty[String]                        })
      f.stage.main(Array("-td", td.buildDir.toString) ++ module ++ p.args)
      val (stdout, stderr, result) =
        grabStdOutErr {
          catchStatus {
            f.stage.main(Array("-td", td.buildDir.toString) ++ module ++ p.args)
          }
        }

      p.stdout.foreach {
        case Right(a) =>
          Then(s"""STDOUT should include "$a"""")
          stdout should include (a)
        case Left(a) =>
          Then(s"""STDOUT should not include "$a"""")
          stdout should not include (a)
      }

      p.stderr.foreach {
        case Right(a) =>
          Then(s"""STDERR should include "$a"""")
          stderr should include (a)
        case Left(a) =>
          Then(s"""STDERR should not include "$a"""")
          stderr should not include (a)
      }

      p.result match {
        case 0 =>
          And(s"the exit code should be 0")
          result shouldBe a [Right[_,_]]
        case a =>
          And(s"the exit code should be $a")
          result shouldBe (Left(a))
      }

      p.files.foreach { f =>
        And(s"file '$f' should be emitted in the target directory")
        val out = new File(td.buildDir + s"/$f")
        out should (exist)
        p.fileChecks.get(f).map(_(out))
      }
    }
  }

  /** Run a ChiselMainExceptionTest and verify that all the properties it spells out hold.
    * @param p the test to run
    * @tparam the type of the exception to catch (you shouldn't have to explicitly provide this)
    */
  def runStageExpectException[A <: Throwable: scala.reflect.ClassTag](p: ChiselMainExceptionTest[A]): Unit = {
    Scenario(s"""User runs Chisel Stage with '${p.argsString}'""") {
      val f = new ChiselMainFixture
      val td = new TargetDirectoryFixture(p.testName)

      When(s"""the user tries to compile with '${p.argsString}'""")
      val module: Array[String] =
        (if (p.generator.nonEmpty) { Array("--module", p.generator.get.getName) }
         else                      { Array.empty[String]                        })
      val (stdout, stderr, result) =
        grabStdOutErr {
          catchStatus {
            intercept[A] {
              f.stage.main(Array("-td", td.buildDir.toString) ++ module ++ p.args)
            }
          }
        }

      Then("the expected exception was thrown")
      result should be a ('right)
      val exception = result.right.get
      info(s"""  - Exception was a "${exception.getClass.getName}"""")

      val message = exception.getMessage
      p.message.foreach {
        case Right(a) =>
          Then(s"""STDOUT should include "$a"""")
          message should include (a)
        case Left(a) =>
          Then(s"""STDOUT should not include "$a"""")
          message should not include (a)
      }

      p.stdout.foreach {
        case Right(a) =>
          Then(s"""STDOUT should include "$a"""")
          stdout should include (a)
        case Left(a) =>
          Then(s"""STDOUT should not include "$a"""")
          stdout should not include (a)
      }

      p.stderr.foreach {
        case Right(a) =>
          Then(s"""STDERR should include "$a"""")
          stderr should include (a)
        case Left(a) =>
          Then(s"""STDERR should not include "$a"""")
          stderr should not include (a)
      }

      val stackTraceString = exception.getStackTrace.mkString("\n")
      p.stackTrace.foreach {
        case Left(a) =>
          And(s"""the stack does not include "$a"""")
          stackTraceString should not include (a)
        case Right(a) =>
          And(s"""the stack trace includes "$a"""")
          stackTraceString should include (a)
      }

    }
  }

  info("As a Chisel user")
  info("I compile a design")
  Feature("show elaborating message") {
    runStageExpectFiles(
      ChiselMainTest(args = Array("-X", "high"),
        generator = Some(classOf[SameTypesModule])
      )
    )
  }

  info("I screw up and compile some bad code")
  Feature("Stack trace trimming of ChiselException") {
    Seq(
      ChiselMainExceptionTest[chisel3.internal.ChiselException](
        args = Array("-X", "low"),
        generator = Some(classOf[DifferentTypesModule]),
        stackTrace = Seq(Left("java"), Right(classOf[DifferentTypesModule].getName))
      ),
      ChiselMainExceptionTest[chisel3.internal.ChiselException](
        args = Array("-X", "low", "--full-stacktrace"),
        generator = Some(classOf[DifferentTypesModule]),
        stackTrace = Seq(Right("java"), Right(classOf[DifferentTypesModule].getName))
      )
    ).foreach(runStageExpectException)
  }
  Feature("Stack trace trimming of user exceptions") {
    Seq(
      ChiselMainExceptionTest[java.lang.IllegalArgumentException](
        args = Array("-X", "low"),
        generator = Some(classOf[FailingRequirementModule]),
        stackTrace = Seq(Right(classOf[FailingRequirementModule].getName), Left("java"))
      ),
      ChiselMainExceptionTest[java.lang.IllegalArgumentException](
        args = Array("-X", "low", "--full-stacktrace"),
        generator = Some(classOf[FailingRequirementModule]),
        stackTrace = Seq(Right(classOf[FailingRequirementModule].getName), Right("java"))
      )
    ).foreach(runStageExpectException)
  }
  Feature("Stack trace trimming and Builder.error errors") {
    Seq(
      ChiselMainExceptionTest[chisel3.internal.ChiselException](
        args = Array("-X", "low"),
        generator = Some(classOf[BuilderErrorModule]),
        message = Seq(Right("Fatal errors during hardware elaboration")),
        stdout = Seq(Right("ChiselMainSpec.scala:43: Invalid bit range (3,-1) in class chiselTests.stage.ChiselMainSpec$BuilderErrorModule"))
      )
    ).foreach(runStageExpectException)
  }

  Feature("Specifying a custom output file") {
    runStageExpectFiles(ChiselMainTest(
      args = Array("--chisel-output-file", "Foo", "--no-run-firrtl"),
      generator = Some(classOf[SameTypesModule]),
      files = Seq("Foo.fir"),
      fileChecks = Map(
        "Foo.fir" -> { file =>
          And("It should be valid FIRRTL")
          Parser.parse(Source.fromFile(file).mkString)
        }
      )
    ))
    runStageExpectFiles(ChiselMainTest(
      args = Array("--chisel-output-file", "Foo.pb", "--no-run-firrtl"),
      generator = Some(classOf[SameTypesModule]),
      files = Seq("Foo.pb"),
      fileChecks = Map(
        "Foo.pb" -> { file =>
          And("It should be valid ProtoBuf")
          firrtl.proto.FromProto.fromFile(file.toString)
        }
      )
    ))
  }

  info("As an aspect writer")
  info("I write an aspect")
  Feature("Running aspects via the command line") {
    Seq(
      ChiselMainTest(args = Array( "-X", "high", "--with-aspect", "chiselTests.stage.TestClassAspect" ),
        generator = Some(classOf[SameTypesModule]),
        stdout = Seq(Right("Ran inspectingAspect"))),
      ChiselMainTest(args = Array( "-X", "high", "--with-aspect", "chiselTests.stage.TestObjectAspect" ),
        generator = Some(classOf[SameTypesModule]),
        stdout = Seq(Right("Ran inspectingAspect")))
    ).foreach(runStageExpectFiles)
  }

}
