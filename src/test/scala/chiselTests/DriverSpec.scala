// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import firrtl.{FirrtlExecutionSuccess, HasFirrtlExecutionOptions, FIRRTLException}
import firrtl.annotations.Annotation
import firrtl.options.ExecutionOptionsManager
import org.scalacheck.Test.Failed
import org.scalatest.{FreeSpec, Matchers, Succeeded}

class DummyModule extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(1.W))
    val out = Output(UInt(1.W))
  })
  io.out := io.in
}

class DriverSpec extends FreeSpec with Matchers {

  val name = "DummyModule"
  def filesShouldExist(exts: Seq[String])(implicit dir: String): Unit = exts
    .foreach{ ext =>
      val dummyOutput = new File(dir, name + "." + ext)
      dummyOutput.exists() should be (true)
      dummyOutput.delete() }
  def filesShouldNotExist(exts: Seq[String])(implicit dir: String): Unit = exts
    .foreach{ ext =>
      val dummyOutput = new File(dir, name + "." + ext)
      dummyOutput.exists should be (false) }

  "Driver's execute methods are used to run chisel and firrtl" - {
    "options can be picked up from the command line with no args" in {
      // NOTE: Since we don't provide any arguments (notably, "--target-dir"),
      //  the generated files will be created in the current directory.
      implicit val targetDir = "."
      Driver.execute(Array.empty[String], () => new DummyModule) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          filesShouldExist(List("anno.json", "fir", "v"))
          Succeeded
        case _ =>
          Failed
      }
    }

    "options can be picked up from the command line setting top name" in {
      implicit val targetDir = "test_run_dir"
      Driver.execute(Array(
                       "-tn", name,
                       "-td", targetDir,
                       "--compiler", "low"), () => new DummyModule) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          filesShouldExist(List("anno.json", "fir", "lo.fir"))
          filesShouldNotExist(List("v", "hi.fir", "mid.fir"))
          Succeeded
        case _ =>
          Failed
      }
    }

    "--dont-save-chirrtl should disable CHIRRTL emission" in {
      implicit val targetDir = "test_run_dir"
      val args = Array(
        "--dont-save-chirrtl",
        "--compiler", "middle",
        "--target-dir", targetDir)
      Driver.execute(args, () => new DummyModule) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          filesShouldExist(List("anno.json", "mid.fir"))
          filesShouldNotExist(List("fir", "v", "hi.fir", "lo.fir"))
          Succeeded
        case _ =>
          Failed
      }
    }

    "--dont-save-annotations should disable annotation emission" in {
      implicit val targetDir = "test_run_dir"
      val args = Array(
        "--dont-save-annotations",
        "--compiler", "high",
        "--target-dir", targetDir)
      Driver.execute(args, () => new DummyModule) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          filesShouldExist(List("hi.fir"))
          filesShouldNotExist(List("v", "lo.fir", "mid.fir", "anno.json"))
          Succeeded
        case _ =>
          Failed
      }
    }

    "--no-run-firrtl should emit CHIRRTL and not FIRRTL or Verilog" in {
      implicit val targetDir = "test_run_dir"
      val args = Array(
        "--no-run-firrtl",
        "--compiler", "verilog",
        "--target-dir", targetDir)
      Driver.execute(args, () => new DummyModule) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          filesShouldExist(List("anno.json", "fir"))
          filesShouldNotExist(List("v", "hi.fir", "lo.fir", "mid.fir"))
          Succeeded
        case _ =>
          Failed
      }
    }

    "deprecated execute method still works" in {
      implicit val targetDir = "test_run_dir"
      val optionsManager = new ExecutionOptionsManager ("test") with HasFirrtlExecutionOptions
          with HasChiselExecutionOptions
      val args = Array( "-tn", name,
                        "-td", targetDir )
      Driver.execute(args, () => new DummyModule) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          filesShouldExist(List("anno.json", "fir", "v"))
          filesShouldNotExist(List("hi.fir", "lo.fir", "mid.fir"))
          Succeeded
        case _ =>
          Failed
      }
    }

    "execute returns a chisel execution result" in {
      val targetDir = "test_run_dir"
      val args = Array("--dut", "chiselTests.DummyModule", "--compiler", "low", "--target-dir", targetDir)
      val result = Driver.execute(args, Seq[Annotation]())
      result shouldBe a[ChiselExecutionSuccess]
      val successResult = result.asInstanceOf[ChiselExecutionSuccess]
      successResult.emitted should include ("circuit DummyModule")
      val dummyOutput = new File(targetDir, "DummyModule.lo.fir")
      dummyOutput.exists() should be(true)
      dummyOutput.delete()
    }
  }

  "Invalid options should be caught when" - {
    def shouldExceptOnOptionsOrAnnotations(name: String, args: Array[String], annos: Seq[Annotation]) {
      name in {
        info("via annotations")
        a [ChiselOptionsException] should be thrownBy (Driver.execute(Array[String](), Seq.fill(2)(annos).flatten))
        info("via arguments")
        a [ChiselOptionsException] should be thrownBy (Driver.execute(Array.fill(2)(args).flatten, Seq[Annotation]()))
        info("via arguments and annotations")
        a [ChiselOptionsException] should be thrownBy (Driver.execute(args, annos))
      }
    }

    "no Chisel circuit is specified" in {
      a [ChiselException] should be thrownBy (Driver.execute(Array[String](), Seq[Annotation]()))
    }

    shouldExceptOnOptionsOrAnnotations("multiple Chisel circuits are specified",
                                       Array("--dut", "chiselTests.DummyModule"),
                                       Seq(ChiselCircuitAnnotation(() => new DummyModule)))
  }
}
