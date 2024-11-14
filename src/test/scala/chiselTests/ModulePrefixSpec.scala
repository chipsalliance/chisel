// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.ExtModule
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance, Instantiate}
import circt.stage.ChiselStage.emitCHIRRTL
import circt.stage.ChiselStage
import chisel3.util.SRAM

object ModulePrefixSpec {
  // This has to be defined at the top-level because @instantiable doesn't work when nested.
  @instantiable
  class AddOne(width: Int) extends Module {
    @public val in = IO(Input(UInt(width.W)))
    @public val out = IO(Output(UInt(width.W)))
    out := in + 1.U
  }
}

class ModulePrefixSpec extends ChiselFlatSpec with ChiselRunners with Utils with MatchesAndOmits {
  import ModulePrefixSpec._
  behavior.of("withModulePrefix")

  it should "prefix modules in a withModulePrefix block, but not outside" in {
    class Foo extends RawModule {
      val a = Wire(Bool())
    }

    class Top extends RawModule {
      val foo = Module(new Foo)

      val pref_foo = withModulePrefix("Pref") { Module(new Foo) }
    }

    val chirrtl = emitCHIRRTL(new Top)

    val lines = """
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
      """
      module Outer_Inner_Bar :
        wire a : UInt<1>
      module Outer_Foo
        inst bar of Outer_Inner_Bar
      module Top :
        inst foo of Outer_Foo
        """.linesIterator.map(_.trim).toSeq

    matchesAndOmits(chirrtl)(lines: _*)()
  }

  it should "Instantiate should create distinct module definitions when instantiated with distinct prefixes" in {
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

    val lines = """
      module Foo_AddOne :
      public module Top :
        inst foo_inst1 of Foo_AddOne
        inst foo_inst2 of Foo_AddOne
        """.linesIterator.map(_.trim).toSeq

    matchesAndOmits(chirrtl)(lines: _*)("AddOne_1", "Bar_AddOne")
  }

  it should "Memories work" in {
    class Top extends Module {
      val io = IO(new Bundle {
        val enable = Input(Bool())
        val write = Input(Bool())
        val addr = Input(UInt(10.W))
        val dataIn = Input(UInt(8.W))
        val dataOut = Output(UInt(8.W))
      })

      val smem = withModulePrefix("Foo") {
        SyncReadMem(1024, UInt(8.W))
      }

      val cmem = withModulePrefix("Bar") {
        Mem(1024, UInt(8.W))
      }

      val sram = withModulePrefix("Baz") {
        SRAM(1024, UInt(8.W), 1, 1, 0)
      }

      smem.write(io.addr, io.dataIn)
      io.dataOut := smem.read(io.addr, io.enable)
    }

    val chirrtl = emitCHIRRTL(new Top)

    val lines = """
      {
        "class":"chisel3.ModulePrefixAnnotation",
        "target":"~Top|Top>smem",
        "prefix":"Foo_"
      },
      {
        "class":"chisel3.ModulePrefixAnnotation",
        "target":"~Top|Top>cmem",
        "prefix":"Bar_"
      },
      {
        "class":"chisel3.ModulePrefixAnnotation",
        "target":"~Top|Top>sram_sram",
        "prefix":"Baz_"
      }
        """.linesIterator.map(_.trim).toSeq

    matchesAndOmits(chirrtl)(lines: _*)()
  }

  it should "Definitions that appear within withModulePrefix get prefixed" in {
    class Top extends Module {
      val dfn = withModulePrefix("Foo") {
        Definition(new AddOne(8))
      }

      val addone = Instance(dfn)
    }

    val chirrtl = emitCHIRRTL(new Top)

    val lines = """
  module Foo_AddOne
  module Top
        """.linesIterator.map(_.trim).toSeq

    matchesAndOmits(chirrtl)(lines: _*)()
  }

  it should "allow definitions to be instantiated within a withModulePrefix block without prefixing it" in {
    class Child(defn: Definition[AddOne]) extends Module {
      val addone = Instance(defn)
    }

    class Top extends Module {
      val defn = Definition(new AddOne(8))

      val child = withModulePrefix("Foo") {
        Module(new Child(defn))
      }
    }

    val chirrtl = emitCHIRRTL(new Top)

    val lines = """
    module AddOne
    module Foo_Child
    public module Top
        """.linesIterator.map(_.trim).toSeq
  }

  it should "withModulePrefix does not automatically affect ExtModules" in {
    class Sub extends ExtModule {}

    class Top extends Module {
      val sub_foo = withModulePrefix("Foo") { Module(new Sub) }
    }

    val chirrtl = emitCHIRRTL(new Top)

    val lines = """
      extmodule Sub
        defname = Sub
      module Top
        """.linesIterator.map(_.trim).toSeq

    matchesAndOmits(chirrtl)(lines: _*)()
  }

  it should "Using modulePrefix to force the name of an extmodule" in {
    class Sub extends ExtModule {
      override def desiredName = modulePrefix + "Sub"
    }

    class Top extends Module {
      val sub_foo = withModulePrefix("Foo") { Module(new Sub) }
    }

    val chirrtl = emitCHIRRTL(new Top)

    val lines = """
      extmodule Foo_Sub
        defname = Foo_Sub
      module Top
        """.linesIterator.map(_.trim).toSeq

    matchesAndOmits(chirrtl)(lines: _*)()
  }
}
