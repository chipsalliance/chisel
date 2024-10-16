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

    val lines = """
      {
        "class":"chisel3.ModulePrefixAnnotation",
        "target":"~Top|Pref_Foo",
        "prefix":"Pref_"
      }

      module Foo :
        wire a : UInt<1>
      module Pref_Foo :
        wire a : UInt<1>
      module Top :
        inst foo of Foo
        inst pref_foo of Pref_Foo
        """.linesIterator.map(_.trim).toSeq
    matchesAndOmits(chirrtl)(lines: _*)()
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

    val lines =
      """circuit Top :%[[
      [[
        {
          "class":"chisel3.ModulePrefixAnnotation",
          "target":"~Top|Outer_Inner_Bar",
          "prefix":"Outer_Inner_"
        },
        {
          "class":"chisel3.ModulePrefixAnnotation",
          "target":"~Top|Outer_Foo",
          "prefix":"Outer_"
        }
      ]]
      module Outer_Inner_Bar :
        wire a : UInt<1>
      module Outer_Foo
        inst bar of Outer_Inner_Bar
      module Top :
        inst foo of Outer_Foo
        """.linesIterator.map(_.trim).toSeq

    matchesAndOmits(chirrtl)(lines: _*)()
  }
}
