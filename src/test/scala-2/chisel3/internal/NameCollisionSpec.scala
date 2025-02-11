// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage

class NameCollisionSpec extends ChiselFlatSpec with Utils {
  behavior.of("Builder")

  it should "error on duplicated names with a correct message" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          val foo, bar = IO(Input(UInt(8.W)))
          val out = IO(Output(UInt(8.W)))

          // Rename both inputs to the same name
          foo.suggestName("same")
          bar.suggestName("same")

          out := foo + bar
        },
        Array("--throw-on-first-error")
      )
    }).getMessage should include(
      "Attempted to name NameCollisionSpec_Anon.same: IO[UInt<8>] with a duplicated name 'same'."
    )
  }

  it should "error on sanitization resulting in duplicated names with a helpful message" in {
    // Case one: An unsanitary name that results in a collision with an existing name once sanitized
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          val foo, bar = IO(Input(UInt(8.W)))
          val out = IO(Output(UInt(8.W)))

          // Seed an initial name with no necessary sanitization
          foo.suggestName("unsanitary")
          // Seed an additional name which, when sanitized, collides with the previous name
          bar.suggestName("unsanitary-")

          out := foo + bar
        },
        Array("--throw-on-first-error")
      )
    }).getMessage should include(
      "Attempted to name NameCollisionSpec_Anon.unsanitary-: IO[UInt<8>] with an unsanitary name 'unsanitary-'"
    )

    // Case two: An unsanitary name which does not collide with any names once sanitized. No error is raised
    ChiselStage.emitCHIRRTL(
      new Module {
        val foo, bar = IO(Input(UInt(8.W)))
        val out = IO(Output(UInt(8.W)))

        out.suggestName("unsanitary-")

        out := foo + bar
      },
      Array("--throw-on-first-error")
    )
  }

  it should "error on nameless ports being assigned default names" in {
    ((the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.emitCHIRRTL(
        new Module {
          // Write to an output port that isn't assigned to a val, and so doesn't get prefixed
          IO(Output(UInt(8.W))) := 123.U
        },
        Array("--throw-on-first-error")
      )
    }).getMessage should include).regex("Assign .+ to a val, or explicitly call suggestName to seed a unique name")
  }
}
