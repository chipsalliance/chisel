// SPDX-License-Identifier: Apache-2.0

package chiselTests

import java.io.File

import chisel3._
import firrtl.FirrtlExecutionSuccess
import org.scalacheck.Test.Failed
import org.scalatest.Succeeded
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class DummyModule extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(1.W))
    val out = Output(UInt(1.W))
  })
  io.out := io.in
}

class UserErrorModule extends chisel3.MultiIOModule {
  val in = IO(Input(UInt(1.W)))
  val out = IO(Output(SInt(1.W)))
  assert(false, "The user wrote an assertion that failed")
  out := in
}

class DriverSpec extends AnyFreeSpec with Matchers with chiselTests.Utils {
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
            info(s"${dummyOutput.toString} exists")
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
            info(s"${dummyOutput.toString} exists")
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

      info("Driver returned a ChiselExecutionSuccess")
      val result = Driver.execute(args, () => new DummyModule)
      result shouldBe a[ChiselExecutionSuccess]

      info("emitted circuit included 'circuit DummyModule'")
      val successResult = result.asInstanceOf[ChiselExecutionSuccess]
      successResult.emitted should include ("circuit DummyModule")

      val dummyOutput = new File(targetDir, "DummyModule.lo.fir")
      info(s"${dummyOutput.toString} exists")
      dummyOutput.exists() should be(true)
      dummyOutput.delete()
    }

    "user errors show a trimmed stack trace" in {
      val targetDir = "test_run_dir"
      val args = Array("--compiler", "low", "--target-dir", targetDir)

      val (stdout, stderr, result) = grabStdOutErr { Driver.execute(args, () => new UserErrorModule) }

      info("""stdout should include "--full-stacktrace"""")
      stdout should include ("Stack trace trimmed to user code only")

      info("stdout does not include FIRRTL information")
      stdout should not include ("firrtl.")

      info("Driver returned a ChiselExecutionFailure")
      result shouldBe a [ChiselExecutionFailure]
    }
  }
}
