// SPDX-License-Identifier: Apache-2.0

package chiselTests.stage

import chisel3._
import chisel3.stage.ChiselMain
import java.io.File

import chisel3.aop.inspecting.{InspectingAspect, InspectorAspect}
import org.scalatest.GivenWhenThen
import org.scalatest.featurespec.AnyFeatureSpec
import org.scalatest.matchers.should.Matchers

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
  class SameTypesModule extends MultiIOModule {
    val in = IO(Input(UInt(1.W)))
    val out = IO(Output(UInt(1.W)))
    out := in
  }

  /** A module that fails a requirement */
  class FailingRequirementModule extends RawModule {
    require(false)
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
    stdout: Option[String] = None,
    stderr: Option[String] = None,
    result: Int = 0,
    fileChecks: Map[String, File => Unit] = Map.empty) {
    def testName: String = "args" + args.mkString("_")
    def argsString: String = args.mkString(" ")
  }

  def runStageExpectFiles(p: ChiselMainTest): Unit = {
    Scenario(s"""User runs Chisel Stage with '${p.argsString}'""") {
      val f = new ChiselMainFixture
      val td = new TargetDirectoryFixture(p.testName)

      p.files.foreach( f => new File(td.buildDir + s"/$f").delete() )

      When(s"""the user tries to compile with '${p.argsString}'""")
      val (stdout, stderr, result) =
        grabStdOutErr {
          catchStatus {
            val module: Array[String] = Array("foo") ++
              (if (p.generator.nonEmpty) { Array("--module", p.generator.get.getName) }
               else                      { Array.empty[String]                        })
            f.stage.main(Array("-td", td.buildDir.toString) ++ module ++ p.args)
          }
        }

      p.stdout match {
        case Some(a) =>
          Then(s"""STDOUT should include "$a"""")
          stdout should include (a)
        case None =>
          Then(s"nothing should print to STDOUT")
          stdout should be (empty)
      }

      p.stderr match {
        case Some(a) =>
          And(s"""STDERR should include "$a"""")
          stderr should include (a)
        case None =>
          And(s"nothing should print to STDERR")
          stderr should be (empty)
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

  info("As a Chisel user")
  info("I compile a design")
  Feature("show elaborating message") {
    runStageExpectFiles(
      ChiselMainTest(args = Array("-X", "high"),
        generator = Some(classOf[SameTypesModule]),
        stdout = Some("Done elaborating.")
      )
    )
  }

  info("I screw up and compile some bad code")
  Feature("Stack trace trimming") {
    Seq(
      ChiselMainTest(args = Array("-X", "low"),
                     generator = Some(classOf[DifferentTypesModule]),
                     stdout = Some("Stack trace trimmed to user code only"),
                     result = 1),
      ChiselMainTest(args = Array("-X", "high", "--full-stacktrace"),
                     generator = Some(classOf[DifferentTypesModule]),
                     stdout = Some("org.scalatest"),
                     result = 1)
    ).foreach(runStageExpectFiles)
  }
  Feature("Report properly trimmed stack traces") {
    Seq(
      ChiselMainTest(args = Array("-X", "low"),
                     generator = Some(classOf[FailingRequirementModule]),
                     stdout = Some("requirement failed"),
                     result = 1),
      ChiselMainTest(args = Array("-X", "low", "--full-stacktrace"),
                     generator = Some(classOf[FailingRequirementModule]),
                     stdout = Some("chisel3.internal.ChiselException"),
                     result = 1)
    ).foreach(runStageExpectFiles)
  }
  Feature("Builder.error source locator") {
    Seq(
      ChiselMainTest(args = Array("-X", "none"),
        generator = Some(classOf[BuilderErrorModule]),
        stdout = Some("ChiselMainSpec.scala:41: Invalid bit range (3,-1) in class chiselTests.stage.ChiselMainSpec$BuilderErrorModule"),
        result = 1)
    ).foreach(runStageExpectFiles)
  }

  Feature("Specifying a custom output file") {
    runStageExpectFiles(ChiselMainTest(
      args = Array("--chisel-output-file", "Foo", "--no-run-firrtl"),
      generator = Some(classOf[SameTypesModule]),
      stdout = Some(""),
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
      stdout = Some(""),
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
        stdout = Some("Ran inspectingAspect")),
      ChiselMainTest(args = Array( "-X", "high", "--with-aspect", "chiselTests.stage.TestObjectAspect" ),
        generator = Some(classOf[SameTypesModule]),
        stdout = Some("Ran inspectingAspect"))
    ).foreach(runStageExpectFiles)
  }

}
