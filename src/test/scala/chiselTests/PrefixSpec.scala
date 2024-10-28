// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance, Instantiate}
import circt.stage.ChiselStage.emitCHIRRTL

class PrefixSpec extends ChiselFlatSpec with ChiselRunners with Utils with MatchesAndOmits {

  behavior.of("withModulePrefix")

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

  it should "Instnantiate should create distinct module definitions when instantiated with distinct prefixes" in {
    class Top extends Module {
      val width = 8
      val in = IO(Input(UInt(width.W)))
      val out = IO(Output(UInt(width.W)))

      val foo_inst = withModulePrefix("Foo") {
        Instantiate(new AddOne(width))
      }

      val bar_inst = withModulePrefix("Bar") {
        Instantiate(new AddOne(width))
      }

      // np: no prefix
      val np_inst = Instantiate(new AddOne(width))

      foo_inst.in := in
      bar_inst.in := foo_inst.out
      out := bar_inst.out
      np_inst.in := in
    }

    val chirrtl = emitCHIRRTL(new Top)
    // println(chirrtl)

    val lines = """
      module Foo_AddOne :
      module Bar_AddOne :
      module AddOne :
      public module Top :
        inst np_inst of AddOne
        inst foo_inst of Foo_AddOne
        inst bar_inst of Bar_AddOne
        """.linesIterator.map(_.trim).toSeq

    matchesAndOmits(chirrtl)(lines: _*)("AddOne_1")
  }

  it should "Instantiate should reference the same module definitions when instantiated with the same prefix" in {
    class Top extends Module {
      val width = 8
      val in = IO(Input(UInt(width.W)))
      val out = IO(Output(UInt(width.W)))
      val foo_inst1 = withModulePrefix("Foo") {
        Instantiate(new AddOne(width))
      }

      val foo_inst2 = withModulePrefix("Foo") {
        Instantiate(new AddOne(width))
      }

      foo_inst1.in := in
      foo_inst2.in := in
      out := foo_inst1.out
    }

    val chirrtl = emitCHIRRTL(new Top)
    // println(chirrtl)

    val lines = """
      module Foo_AddOne :
      public module Top :
        inst foo_inst1 of Foo_AddOne
        inst foo_inst2 of Foo_AddOne
        """.linesIterator.map(_.trim).toSeq

    matchesAndOmits(chirrtl)(lines: _*)("AddOne_1", "Bar_AddOne")
  }
}

// This has to be defined at the top-level because @instantiable doesn't work when nested.
@instantiable
class AddOne(width: Int) extends Module {
  @public val in = IO(Input(UInt(width.W)))
  @public val out = IO(Output(UInt(width.W)))
  out := in + 1.U
}
