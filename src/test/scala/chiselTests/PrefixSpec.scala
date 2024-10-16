// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage.emitCHIRRTL

class PrefixSpec extends ChiselFlatSpec with ChiselRunners with Utils with MatchesAndOmits {

  behavior.of("withPrefix")

  it should "prefix modules in a withModulePrefix block, but not outside" in {
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

  it should "Allow nested module prefixes" in {
    class Bar extends RawModule {
      val a = Wire(Bool())
    }

    class Foo extends RawModule {
      withModulePrefix("Inner") {
        val bar = Module(new Bar)
      }
    }

    class Top extends RawModule {
      withModulePrefix("Outer") {
        val foo = Module(new Foo)
      }
    }

    val chirrtl = emitCHIRRTL(new Top)
    println(chirrtl)
    matchesAndOmits(chirrtl)(
      "module Outer_Inner_Bar :",
      "  wire a : UInt<1>",
      "module Outer_Foo",
      "  inst bar of Outer_Inner_Bar",
      "module Top :",
      "  inst foo of Outer_Foo",
    )()
  }
}
