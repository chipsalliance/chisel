// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator.scalatest

import chisel3._
import chisel3.simulator.PeekPokeAPI.FailedExpectationException
import chisel3.simulator.{ChiselSettings, ChiselSim, HasSimulator, MacroText}
import chisel3.testing.HasTestingDirectory
import chisel3.testing.scalatest.{FileCheck, TestingDirectory}
import java.nio.file.FileSystems
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers
import scala.reflect.io.Directory

class ChiselSimSpec extends AnyFunSpec with Matchers with ChiselSim with FileCheck with TestingDirectory {

  describe("scalatest.ChiselSim") {

    it("should work correctly for poke and expect") {
      class Foo extends RawModule {
        val a = IO(Input(Bool()))
        val b = IO(Output(Bool()))

        b :<= !a
      }

      info("poke and expect work")
      simulateRaw(new Foo) { foo =>
        foo.a.poke(true.B)
        foo.b.expect(false.B)

        foo.a.poke(false.B)
        foo.b.expect(true.B)
      }

      info("an expect throws an exception")
      intercept[FailedExpectationException[_]] {
        simulateRaw(new Foo) { foo =>
          foo.a.poke(false.B)
          foo.b.expect(false.B)
        }
      }
    }

    it("should error if an expect fails") {
      val message = intercept[Exception] {
        simulate {
          new Module {
            val a = IO(Output(Bool()))
            a :<= false.B
          }
        } { _.a.expect(true.B) }
      }.getMessage
      message.fileCheck() {
        """|CHECK:      Failed Expectation
           |CHECK-NEXT: ---
           |CHECK-NEXT: Observed value: '0'
           |CHECK-NEXT: Expected value: '1'
           |CHECK:      ---
           |""".stripMargin
      }
    }

    it("should error if a chisel3.assert fires during the simulation") {
      class Foo extends Module {
        chisel3.assert(false.B, "foo assertion")
      }

      val message = intercept[Exception] {
        simulate(new Foo) { foo =>
          foo.clock.step(4)
        }
      }.getMessage

      message.fileCheck()(
        """|CHECK:      One or more assertions failed during Chiselsim simulation
           |CHECK-NEXT: ---
           |CHECK-NEXT: The following assertion failures were extracted from the log file:
           |CHECK:      lineNo  line
           |CHECK-NEXT: ---
           |CHECK-NEXT:      0  [5] %Error:
           |CHECK:      For more information, see the complete log file:
           |CHECK:        build/chiselsim/ChiselSimSpec/scalatest.ChiselSim/should-error-if-a-chisel3.assert-fires-during-the-simulation/workdir-verilator/simulation-log.txt
           |CHECK-NEXT: ---
           |""".stripMargin
      )
    }

    it("should error if an ltl.AssertProperty fires during the simulation") {
      class Foo extends Module {
        ltl.AssertProperty(false.B)
      }

      val message = intercept[Exception] {
        simulate(new Foo) { foo =>
          foo.clock.step(4)
        }
      }.getMessage

      message.fileCheck()(
        """|CHECK:      One or more assertions failed during Chiselsim simulation
           |CHECK-NEXT: ---
           |CHECK-NEXT: The following assertion failures were extracted from the log file:
           |CHECK:      lineNo  line
           |CHECK-NEXT: ---
           |CHECK-NEXT:      0  [5] %Error:
           |CHECK:      For more information, see the complete log file:
           |CHECK:        build/chiselsim/ChiselSimSpec/scalatest.ChiselSim/should-error-if-an-ltl.AssertProperty-fires-during-the-simulation/workdir-verilator/simulation-log.txt
           |CHECK-NEXT: ---
           |""".stripMargin
      )
    }

    it("should allow for customization of macros during simulation") {
      class Foo extends RawModule {
        val a, b, c = IO(Input(Bool()))
      }

      val chiselSettings = ChiselSettings
        .defaultRaw[Foo]
        .copy(
          assertVerboseCond = Some(MacroText.Signal(_.a)),
          printfCond = Some(MacroText.Signal(_.b)),
          stopCond = Some(MacroText.NotSignal(_.c))
        )

      simulateRaw(new Foo, chiselSettings = chiselSettings) { _ => }

      io.Source
        .fromFile(
          FileSystems
            .getDefault()
            .getPath(implicitly[HasTestingDirectory].getDirectory.toString, "workdir-verilator", "Makefile")
            .toFile
        )
        .mkString
        .fileCheck()(
          """|CHECK:      '+define+ASSERT_VERBOSE_COND=svsimTestbench.a'
             |CHECK-NEXT: '+define+PRINTF_COND=svsimTestbench.b'
             |CHECK-NEXT: '+define+STOP_COND=!svsimTestbench.c'
             |""".stripMargin
        )
    }

    it("should allow for a user to customize the build directory") {
      class Foo extends Module {
        stop()
      }

      /** An implementation that always writes to the subdirectory "test_run_dir/<class-name>/foo/" */
      implicit val fooDirectory = new HasTestingDirectory {
        override def getDirectory =
          FileSystems.getDefault().getPath("test_run_dir", "foo")
      }

      val directory = Directory(FileSystems.getDefault().getPath("test_run_dir", "foo").toFile())
      directory.deleteRecursively()

      simulate(new Foo()) { _ => }(
        hasSimulator = implicitly[HasSimulator],
        testingDirectory = fooDirectory,
        implicitly,
        implicitly
      )

      info(s"found expected directory: '$directory'")
      assert(directory.exists)
      assert(directory.isDirectory)

      val allFiles = directory.deepFiles.toSeq.map(_.toString).toSet
      for (
        file <- Seq(
          "test_run_dir/foo/workdir-verilator/Makefile",
          "test_run_dir/foo/primary-sources/Foo.sv"
        )
      ) {
        info(s"found expected file: '$file'")
        allFiles should contain(file)
      }
    }

    it("should dump a waveform when enableWaves is used") {

      implicit val verilator = HasSimulator.simulators
        .verilator(verilatorSettings =
          svsim.verilator.Backend.CompilationSettings(
            traceStyle =
              Some(svsim.verilator.Backend.CompilationSettings.TraceStyle.Vcd(traceUnderscore = true, "trace.vcd"))
          )
        )

      class Foo extends Module {
        stop()
      }

      val vcdFile = FileSystems
        .getDefault()
        .getPath(implicitly[HasTestingDirectory].getDirectory.toString, "workdir-verilator", "trace.vcd")
        .toFile

      vcdFile.delete

      simulateRaw(new Foo) { _ =>
        enableWaves()
      }

      info(s"$vcdFile exists")
      vcdFile should (exist)
    }
  }

}
