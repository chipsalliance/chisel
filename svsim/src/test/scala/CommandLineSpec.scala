// SPDX-License-Identifier: Apache-2.0

package svsim.test

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import svsim.CommonCompilationSettings

class CommandLineSpec extends AnyFlatSpec with Matchers {

  case class OptionTest(name: String, settings: svsim.verilator.Backend.CompilationSettings, string: Seq[String])

  behavior of ("Verilator options")

  val verilatorBackend = new svsim.verilator.Backend(executablePath = "foo")

  val parallelismTests = {
    import svsim.verilator.Backend.CompilationSettings
    import svsim.verilator.Backend.CompilationSettings.Parallelism
    Seq(
      OptionTest("default", CompilationSettings(), Seq("-j", "0")),
      OptionTest(
        "uniform parallelism",
        CompilationSettings(parallelism = Some(Parallelism.Uniform.default.withNum(2))),
        Seq("-j", "2")
      ),
      OptionTest(
        "only build parallelism",
        CompilationSettings(parallelism = Some(Parallelism.Different.default.withBuild(Some(4)))),
        Seq("--build-jobs", "4")
      ),
      OptionTest(
        "only verilate parallelism",
        CompilationSettings(parallelism = Some(Parallelism.Different.default.withVerilate(Some(8)))),
        Seq("--verilate-jobs", "8")
      ),
      OptionTest(
        "different build and verilate parallelism",
        CompilationSettings(parallelism =
          Some(Parallelism.Different.default.withBuild(Some(16)).withVerilate(Some(32)))
        ),
        Seq("--build-jobs", "16", "--verilate-jobs", "32")
      )
    )
  }

  parallelismTests.foreach { case OptionTest(name, settings, expected) =>
    it should s"support $name behavior" in {
      verilatorBackend
        .generateParameters("bar", "baz", Seq.empty, CommonCompilationSettings(), settings)
        .compilerInvocation
        .arguments should contain inOrderElementsOf (expected)
    }
  }

}
