// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.probe
import chisel3.testers._
import chisel3.util.experimental.BoringUtils

class BoringUtilsTapSpec extends ChiselFlatSpec with ChiselRunners with Utils with MatchesAndOmits {
  "Ready-only tap" should "work downwards from parent to child" in {
    class Foo extends RawModule {
      val internalWire = Wire(Bool())
    }
    class Top extends RawModule {
      val foo = Module(new Foo())
      val outProbe = IO(probe.Probe(Bool()))
      val out = IO(Bool())
      probe.define(outProbe, BoringUtils.tap(foo.internalWire))
      out := BoringUtils.tapAndRead(foo.internalWire)
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
    matchesAndOmits(chirrtl)(
      "module Foo :",
      "output bore : Probe<UInt<1>>",
      "output out_bore : Probe<UInt<1>>",
      "define bore = probe(internalWire)",
      "define out_bore = probe(internalWire)",
      "module Top :",
      "define outProbe = foo.bore",
      "connect out, read(foo.out_bore)"
    )()
  }

  it should "work downwards from grandparent to grandchild" in {
    class Bar extends RawModule {
      val internalWire = Wire(Bool())
    }
    class Foo extends RawModule {
      val bar = Module(new Bar)
    }
    class Top extends RawModule {
      val foo = Module(new Foo)
      val out = IO(Bool())
      out := BoringUtils.tapAndRead(foo.bar.internalWire)
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
    matchesAndOmits(chirrtl)(
      "module Bar :",
      "output out_bore : Probe<UInt<1>>",
      "define out_bore = probe(internalWire)",
      "module Foo :",
      "output out_bore : Probe<UInt<1>>",
      "define out_bore = bar.out_bore",
      "module Top :",
      "connect out, read(foo.out_bore)"
    )()
  }

  it should "work upwards from child to parent" in {
    class Foo(parentData: Data) extends RawModule {
      val outProbe = IO(probe.Probe(Bool()))
      val out = IO(Bool())
      probe.define(outProbe, BoringUtils.tap(parentData))
      out := BoringUtils.tapAndRead(parentData)
      out := probe.read(outProbe)
    }
    class Top extends RawModule {
      val parentWire = Wire(Bool())
      val foo = Module(new Foo(parentWire))
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top, Array("--full-stacktrace"))
    matchesAndOmits(chirrtl)(
      "module Foo :",
      "output outProbe : Probe<UInt<1>>",
      "input bore : UInt<1>",
      "input out_bore : UInt<1>",
      "define outProbe = probe(bore)",
      "connect out, out_bore",
      "connect out, read(outProbe)",
      "module Top :",
      "connect foo.bore, parentWire",
      "connect foo.out_bore, parentWire"
    )()
  }

  it should "work upwards from grandchild to grandparent" in {
    class Bar(grandParentData: Data) extends RawModule {
      val out = IO(Bool())
      out := BoringUtils.tapAndRead(grandParentData)
    }
    class Foo(parentData: Data) extends RawModule {
      val bar = Module(new Bar(parentData))
    }
    class Top extends RawModule {
      val parentWire = Wire(Bool())
      val foo = Module(new Foo(parentWire))
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
    matchesAndOmits(chirrtl)(
      "module Bar :",
      "input out_bore : UInt<1>",
      "connect out, out_bore",
      "module Foo :",
      "input out_bore : UInt<1>",
      "connect bar.out_bore, out_bore",
      "module Top :",
      "connect foo.out_bore, parentWire"
    )()
  }

  it should "work from child to its sibling" in {
    class Bar extends RawModule {
      val a = Wire(Bool())
    }
    class Baz(_a: Bool) extends RawModule {
      val b = Wire(Bool())
      b := BoringUtils.tapAndRead(_a)
    }
    class Top extends RawModule {
      val bar = Module(new Bar)
      val baz = Module(new Baz(bar.a))
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
    matchesAndOmits(chirrtl)(
      "module Bar :",
      "output b_bore : Probe<UInt<1>>",
      "define b_bore = probe(a)",
      "module Baz :",
      "input b_bore : UInt<1>",
      "connect b, b_bore",
      "module Top :",
      "connect baz.b_bore, read(bar.b_bore)"
    )()
  }

  it should "work from child to sibling at different levels" in {
    class Bar extends RawModule {
      val a = Wire(Bool())
    }
    class Baz(_a: Bool) extends RawModule {
      val b = Wire(Bool())
      b := BoringUtils.tapAndRead(_a)
    }
    class Foo(_a: Bool) extends RawModule {
      val baz = Module(new Baz(_a))
    }
    class Top extends RawModule {
      val bar = Module(new Bar)
      val foo = Module(new Foo(bar.a))
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
    matchesAndOmits(chirrtl)(
      "module Bar :",
      "output b_bore : Probe<UInt<1>>",
      "define b_bore = probe(a)",
      "module Baz :",
      "input b_bore : UInt<1>",
      "connect b, b_bore",
      "module Foo :",
      "input b_bore : UInt<1>",
      "connect baz.b_bore, b_bore",
      "module Top :",
      "connect foo.b_bore, read(bar.b_bore)"
    )()
  }

  "Writable tap" should "work downwards from grandparent to grandchild" in {
    class Bar extends RawModule {
      val internalWire = Wire(Bool())
    }
    class Foo extends RawModule {
      val bar = Module(new Bar)
    }
    class Top extends RawModule {
      val foo = Module(new Foo)
      val out = IO(Bool())
      out := probe.read(BoringUtils.rwTap(foo.bar.internalWire))
      probe.forceInitial(BoringUtils.rwTap(foo.bar.internalWire), false.B)
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)
    matchesAndOmits(chirrtl)(
      "module Bar :",
      "output out_bore : RWProbe<UInt<1>>",
      "define out_bore = rwprobe(internalWire)",
      "module Foo :",
      "output out_bore : RWProbe<UInt<1>>",
      "define out_bore = bar.out_bore",
      "module Top :",
      "connect out, read(foo.out_bore)",
      "force_initial(foo.bore, UInt<1>(0h0))"
    )()
  }

  it should "not work upwards child to parent" in {
    class Foo(parentData: Data) extends RawModule {
      val outProbe = IO(probe.RWProbe(Bool()))
      probe.define(outProbe, BoringUtils.rwTap(parentData))
    }
    class Top extends RawModule {
      val parentWire = Wire(Bool())
      val foo = Module(new Foo(parentWire))
    }
    val e = intercept[Exception] {
      circt.stage.ChiselStage.emitCHIRRTL(new Top, Array("--throw-on-first-error"))
    }
    e.getMessage should include("Cannot drill writable probes upwards.")
  }

  it should "not work from child to sibling at different levels" in {
    class Bar extends RawModule {
      val a = Wire(Bool())
    }
    class Baz(_a: Bool) extends RawModule {
      val b = Output(probe.RWProbe(Bool()))
      b := BoringUtils.rwTap(_a)
    }
    class Foo(_a: Bool) extends RawModule {
      val baz = Module(new Baz(_a))
    }
    class Top extends RawModule {
      val bar = Module(new Bar)
      val foo = Module(new Foo(bar.a))
    }
    val e = intercept[Exception] {
      circt.stage.ChiselStage.emitCHIRRTL(new Top, Array("--throw-on-first-error"))
    }
    e.getMessage should include("Cannot drill writable probes upwards.")
  }

  it should "work when tapping an element within a Bundle" in {
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(
      new RawModule {
        class Child() extends RawModule {
          val b = Wire(new Bundle {
            val x = Bool()
          })
        }

        val child = Module(new Child())
        val outRWProbe = IO(probe.RWProbe(Bool()))
        probe.define(outRWProbe, BoringUtils.rwTap(child.b.x))
      }
    )
    matchesAndOmits(chirrtl)(
      "wire b : { x : UInt<1>}",
      "define bore = rwprobe(b.x)",
      "define outRWProbe = child.bore"
    )()
  }

  it should "work when tapping an element within a Vec" in {
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(
      new RawModule {
        class Child() extends RawModule {
          val b = Wire(Vec(4, Bool()))
        }

        val child = Module(new Child())
        val outRWProbe = IO(probe.RWProbe(Bool()))
        probe.define(outRWProbe, BoringUtils.rwTap(child.b(2)))
      }
    )
    matchesAndOmits(chirrtl)(
      "wire b : UInt<1>[4]",
      "define bore = rwprobe(b[2])",
      "define outRWProbe = child.bore"
    )()
  }

}
