// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator.scalatest

import chisel3._
import chisel3.simulator.PeekPokeAPI.FailedExpectationException
import chisel3.simulator.{ChiselSettings, ChiselSim, HasTestingDirectory, MacroText}
import chisel3.simulator.scalatest.WithTestingDirectory
import chiselTests.FileCheck
import java.nio.file.FileSystems
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class ChiselSimSpec extends AnyFunSpec with Matchers with ChiselSim with FileCheck with WithTestingDirectory {

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

    it("should error if a chisel3.assert fires during the simulation") {
      class Foo extends Module {
        chisel3.assert(false.B, "foo assertion")
      }

      val message = intercept[Exception] {
        simulate(new Foo) { foo =>
          foo.clock.step(4)
        }
      }.getMessage

      fileCheckString(message)(
        """|CHECK:      One or more assertions failed during Chiselsim simulation
           |CHECK-NEXT: ---
           |CHECK-NEXT: The following assertion failures were extracted from the log file:
           |CHECK:      lineNo  line
           |CHECK-NEXT: ---
           |CHECK-NEXT:      0  [5] %Error:
           |CHECK:      For more information, see the complete log file:
           |CHECK:        build/ChiselSimSpec/scalatest.ChiselSim/should-error-if-a-chisel3.assert-fires-during-the-simulation/workdir-verilator/simulation-log.txt
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

      fileCheckString(message)(
        """|CHECK:      One or more assertions failed during Chiselsim simulation
           |CHECK-NEXT: ---
           |CHECK-NEXT: The following assertion failures were extracted from the log file:
           |CHECK:      lineNo  line
           |CHECK-NEXT: ---
           |CHECK-NEXT:      0  [5] %Error:
           |CHECK:      For more information, see the complete log file:
           |CHECK:        build/ChiselSimSpec/scalatest.ChiselSim/should-error-if-an-ltl.AssertProperty-fires-during-the-simulation/workdir-verilator/simulation-log.txt
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

      fileCheckString(
        io.Source
          .fromFile(
            FileSystems
              .getDefault()
              .getPath(implicitly[HasTestingDirectory].getDirectory.toString, "workdir-verilator", "Makefile")
              .toFile
          )
          .mkString
      )(
        """|CHECK:      '+define+ASSERT_VERBOSE_COND=svsimTestbench.a'
           |CHECK-NEXT: '+define+PRINTF_COND=svsimTestbench.b'
           |CHECK-NEXT: '+define+STOP_COND=!svsimTestbench.c'
           |""".stripMargin
      )
    }
  }

}
