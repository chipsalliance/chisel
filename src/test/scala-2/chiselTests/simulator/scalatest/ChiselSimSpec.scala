// SPDX-License-Identifier: Apache-2.0

package chiselTests.simulator.scalatest

import chisel3._
import chisel3.simulator.PeekPokeAPI.FailedExpectationException
import chisel3.simulator.ChiselSim
import chisel3.simulator.scalatest.WithTestingDirectory
import chiselTests.FileCheck
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class ChiselSimSpec extends AnyFunSpec with Matchers with ChiselSim with FileCheck {

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
           |CHECK-NEXT:   - {{.+}} foo assertion
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
           |CHECK-NEXT:   - {{.+$}}
           |CHECK-NEXT: ---
           |""".stripMargin
      )
    }
  }

}
