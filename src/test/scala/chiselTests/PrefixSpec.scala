// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage.emitCHIRRTL

class PrefixSpec extends ChiselFlatSpec with ChiselRunners with Utils with MatchesAndOmits {

  behavior.of("withPrefix")

  it should "do something" in {
    class Foo extends RawModule {
      val a = Wire(Bool())
    }

    class Top extends RawModule {
      val foo = Module(new Foo)

      withModulePrefix("Pref") {
        val pref_foo = Module(new Foo)
      }
    }

    val chirrtl = emitCHIRRTL(new Top)
    println(chirrtl)
    matchesAndOmits(chirrtl)(
      "module Foo :",
      "  wire a : UInt<1>",
      "module Pref_Foo :",
      "  wire a : UInt<1>",
      "module Top :",
      "  inst foo of Foo",
      "  inst pref_foo of Pref_Foo",
    )()
  }
}
