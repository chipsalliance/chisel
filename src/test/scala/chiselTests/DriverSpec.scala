// See LICENSE for license details.

package chiselTests

import java.io.File

import chisel3._
import firrtl.FirrtlExecutionSuccess
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
  "Driver's execute methods are used to run chisel and firrtl" - {
    "options can be picked up from comand line with no args" in {
      // NOTE: Since we don't provide any arguments (notably, "--target-dir"),
      //  the generated files will be created in the current directory.
      val  targetDir = "."
      Driver.execute(Array.empty[String], () => new DummyModule) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          val exts = List("anno.json", "fir", "v")
          for (ext <- exts) {
            val dummyOutput = new File(targetDir, "DummyModule" + "." + ext)
            dummyOutput.exists() should be(true)
            dummyOutput.delete()
          }
          Succeeded
        case _ =>
          Failed
      }
    }

    "options can be picked up from comand line setting top name" in {
      val  targetDir = "local-build"
      Driver.execute(Array("-tn", "dm", "-td", targetDir), () => new DummyModule) match {
        case ChiselExecutionSuccess(_, _, Some(_: FirrtlExecutionSuccess)) =>
          val exts = List("anno.json", "fir", "v")
          for (ext <- exts) {
            val dummyOutput = new File(targetDir, "dm" + "." + ext)
            dummyOutput.exists() should be(true)
            dummyOutput.delete()
          }
          Succeeded
        case _ =>
          Failed
      }

    }
    "execute returns a chisel execution result" in {
      val targetDir = "test_run_dir"
      val args = Array("--compiler", "low", "--target-dir", targetDir)
      val result = Driver.execute(args, () => new DummyModule)
      result shouldBe a[ChiselExecutionSuccess]
      val successResult = result.asInstanceOf[ChiselExecutionSuccess]
      successResult.emitted should include ("circuit DummyModule")
      val dummyOutput = new File(targetDir, "DummyModule.lo.fir")
      dummyOutput.exists() should be(true)
      dummyOutput.delete()
    }
  }
}
