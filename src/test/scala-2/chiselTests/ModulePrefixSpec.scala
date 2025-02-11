// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.ExtModule
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance, Instantiate}
import circt.stage.ChiselStage.emitCHIRRTL
import circt.stage.ChiselStage
import chisel3.util.SRAM
import firrtl.transforms.DedupGroupAnnotation

object ModulePrefixSpec {
  // This has to be defined at the top-level because @instantiable doesn't work when nested.
  @instantiable
  class AddOne(width: Int) extends Module {
    @public val in = IO(Input(UInt(width.W)))
    @public val out = IO(Output(UInt(width.W)))
    out := in + 1.U
  }
}

class ModulePrefixSpec extends ChiselFlatSpec with ChiselRunners with Utils with FileCheck {
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

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: module Foo :
         |CHECK:         wire a : UInt<1>
         |CHECK-LABEL: module Pref_Foo :
         |CHECK:         wire a : UInt<1>
         |CHECK-LABEL: module Top :
         |CHECK:         inst foo of Foo
         |CHECK-NEXT:    inst pref_foo of Pref_Foo
         |""".stripMargin
    )
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

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: module Outer_Inner_Bar :
         |CHECK:         wire a : UInt<1>
         |CHECK-LABEL: module Outer_Foo
         |CHECK:         inst bar of Outer_Inner_Bar
         |CHECK-LABEL: module Top :
         |CHECK:         inst foo of Outer_Foo
         |""".stripMargin
    )
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

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: module Foo_AddOne :
         |CHECK-LABEL: module Bar_AddOne :
         |CHECK-LABEL: module AddOne :
         |CHECK-LABEL: public module Top :
         |CHECK:         inst foo_inst of Foo_AddOne
         |CHECK:         inst bar_inst of Bar_AddOne
         |CHECK:         inst np_inst of AddOne
         |""".stripMargin
    )
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

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: module Foo_AddOne :
         |CHECK-LABEL: public module Top :
         |CHECK:         inst foo_inst1 of Foo_AddOne
         |CHECK:         inst foo_inst2 of Foo_AddOne
         |""".stripMargin
    )
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

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK:      {
         |CHECK-NEXT:   "class":"chisel3.ModulePrefixAnnotation",
         |CHECK-NEXT:   "target":"~Top|Top>smem",
         |CHECK-NEXT:   "prefix":"Foo_"
         |CHECK-NEXT: },
         |CHECK-NEXT: {
         |CHECK-NEXT:   "class":"chisel3.ModulePrefixAnnotation",
         |CHECK-NEXT:   "target":"~Top|Top>cmem",
         |CHECK-NEXT:   "prefix":"Bar_"
         |CHECK-NEXT: },
         |CHECK-NEXT: {
         |CHECK-NEXT:   "class":"chisel3.ModulePrefixAnnotation",
         |CHECK-NEXT:   "target":"~Top|Top>sram_sram",
         |CHECK-NEXT:   "prefix":"Baz_"
         |CHECK-NEXT: }
         |""".stripMargin
    )
  }

  it should "Definitions that appear within withModulePrefix get prefixed" in {
    class Top extends Module {
      val dfn = withModulePrefix("Foo") {
        Definition(new AddOne(8))
      }

      val addone = Instance(dfn)
    }

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK: module Foo_AddOne
         |CHECK: module Top
         |""".stripMargin
    )
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

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK: module AddOne
         |CHECK: module Foo_Child
         |CHECK: public module Top
         |""".stripMargin
    )
  }

  it should "withModulePrefix does not automatically affect ExtModules" in {
    class Sub extends ExtModule {}

    class Top extends Module {
      val sub_foo = withModulePrefix("Foo") { Module(new Sub) }
    }

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: extmodule Sub
         |CHECK:         defname = Sub
         |CHECK-LABEL: module Top
         |""".stripMargin
    )
  }

  it should "Using modulePrefix to force the name of an extmodule" in {
    class Sub extends ExtModule {
      override def desiredName = modulePrefix + "Sub"
    }

    class Top extends Module {
      val sub_foo = withModulePrefix("Foo") { Module(new Sub) }
    }

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: extmodule Foo_Sub
         |CHECK:  defname = Foo_Sub
         |CHECK:module Top
         |""".stripMargin
    )
  }

  it should "support omitting the separator" in {
    class Foo extends Module
    class Top extends Module {
      val foo = withModulePrefix("Prefix", false) {
        Module(new Foo)
      }
    }

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: module PrefixFoo :
         |CHECK-LABEL: module Top :
         |CHECK:         inst foo of PrefixFoo
         |""".stripMargin
    )
  }

  behavior.of("BaseModule.localModulePrefix")

  it should "set the prefix for a Module and its children" in {

    class Foo extends RawModule
    class Bar extends RawModule

    class Top extends RawModule {
      override def localModulePrefix = Some("Prefix")
      val foo = Module(new Foo)
      val bar = Module(new Bar)
    }

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: module Prefix_Foo :
         |CHECK-LABEL: module Prefix_Bar :
         |CHECK-LABEL: module Prefix_Top :
         |CHECK:         inst foo of Prefix_Foo
         |CHECK:         inst bar of Prefix_Bar
      """.stripMargin
    )
  }

  it should "set the prefix for a Module's children but not the Module itself if localModulePrefixAppliesToSelf is false" in {

    class Foo extends RawModule
    class Bar extends RawModule

    class Top extends RawModule {
      override def localModulePrefix = Some("Prefix")
      override def localModulePrefixAppliesToSelf = false
      val foo = Module(new Foo)
      val bar = Module(new Bar)
    }

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: module Prefix_Foo :
         |CHECK-LABEL: module Prefix_Bar :
         |CHECK-LABEL: module Top :
         |CHECK:         inst foo of Prefix_Foo
         |CHECK:         inst bar of Prefix_Bar
         |""".stripMargin
    )
  }

  it should "compose with withModulePrefix" in {

    class Foo extends RawModule {
      override def localModulePrefix = Some("Inner")
    }
    class Bar extends RawModule

    class Top extends RawModule {
      override def localModulePrefix = Some("Outer")
      val f1 = Module(new Foo)
      withModulePrefix("Prefix") {
        val f2 = Module(new Foo)
        val bar = Module(new Bar)
      }
    }

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: module Outer_Inner_Foo :
         |CHECK-LABEL: module Outer_Prefix_Inner_Foo :
         |CHECK-LABEL: module Outer_Prefix_Bar :
         |CHECK-LABEL: module Outer_Top :
         |CHECK:         inst f1 of Outer_Inner_Foo
         |CHECK:         inst f2 of Outer_Prefix_Inner_Foo
         |CHECK:         inst bar of Outer_Prefix_Bar
         |""".stripMargin
    )
  }

  it should "omit the prefix if localModulePrefixUseSeparator is false" in {
    class Foo extends RawModule {
      override def localModulePrefix = Some("Inner")
      override def localModulePrefixUseSeparator = false
    }
    class Top extends RawModule {
      override def localModulePrefix = Some("Outer")
      override def localModulePrefixUseSeparator = false
      override def localModulePrefixAppliesToSelf = false
      val foo = Module(new Foo)
    }

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: module OuterInnerFoo :
         |CHECK-LABEL: module Top :
         |CHECK:         inst foo of OuterInnerFoo
         |""".stripMargin
    )
  }

  it should "support mixing and matching of separator omission" in {
    class Foo extends RawModule {
      override def localModulePrefix = Some("Inner")
    }
    class Bar extends RawModule {
      override def localModulePrefix = Some("Middle")
      val foo = Module(new Foo)
    }
    class Top extends RawModule {
      override def localModulePrefix = Some("Outer")
      override def localModulePrefixUseSeparator = false
      val bar = Module(new Bar)
    }

    generateFirrtlAndFileCheck(new Top)(
      """|CHECK-LABEL: module OuterMiddle_Inner_Foo :
         |CHECK-LABEL: module OuterMiddle_Bar :
         |CHECK:         inst foo of OuterMiddle_Inner_Foo
         |CHECK-LABEL: module OuterTop :
         |CHECK:         inst bar of OuterMiddle_Bar
         |""".stripMargin
    )
  }

  behavior.of("Module prefixes")

  it should "affect the dedup group" in {
    class Foo extends RawModule
    class Bar extends RawModule {
      override def localModulePrefix = Some("Outer")
      val foo = withModulePrefix("Inner") {
        Module(new Foo)
      }
    }
    class Top extends RawModule {
      val bar = Module(new Bar)
    }
    val (_, annos) = getFirrtlAndAnnos(new Top)
    val dedupGroups = annos.collect { case DedupGroupAnnotation(target, group) =>
      target.module -> group
    }
    dedupGroups should be(Seq("Outer_Inner_Foo" -> "Outer_Inner_Foo", "Outer_Bar" -> "Outer_Bar", "Top" -> "Top"))
  }

}
