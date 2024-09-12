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

  // This test requires ability to identify what region to add commands to,
  // *after* building them.  This is not yet supported.
  ignore should "work downwards from grandparent to grandchild through when" in {
    class Bar extends RawModule {
      val internalWire = Wire(Bool())
    }
    class Foo extends RawModule {
      when(true.B) {
        val bar = Module(new Bar)
      }
    }
    class Top extends RawModule {
      val foo = Module(new Foo)
      val out = IO(Bool())
      out := DontCare

      when(true.B) {
        val w = WireInit(
          Bool(),
          BoringUtils.tapAndRead((chisel3.aop.Select.collectDeep(foo) { case b: Bar => b }).head.internalWire)
        )
      }
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)

    // The define should be at the end of the when block.
    matchesAndOmits(chirrtl)(
      "    when UInt<1>(0h1) :",
      "      inst bar of Bar",
      "      define w_bore = bar.w_bore"
    )()

    // Check is valid FIRRTL.
    circt.stage.ChiselStage.emitFIRRTLDialect(new Top)
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

  it should "work upwards from grandchild to grandparent through when" in {
    class Bar(grandParentData: Data) extends RawModule {
      val out = IO(Bool())
      out := BoringUtils.tapAndRead(grandParentData)
    }
    class Foo(parentData: Data) extends RawModule {
      when(true.B) {
        val bar = Module(new Bar(parentData))
      }
    }
    class Top extends RawModule {
      val parentWire = Wire(Bool())
      parentWire := DontCare
      val foo = Module(new Foo(parentWire))
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)

    // The connect should be at the end of the when block.
    matchesAndOmits(chirrtl)(
      "    when UInt<1>(0h1) :",
      "      inst bar of Bar",
      "      connect bar.out_bore, out_bore"
    )()

    // Check is valid FIRRTL.
    circt.stage.ChiselStage.emitFIRRTLDialect(new Top)
  }

  it should "work upwards from grandchild to grandparent into layer" in {
    object TestLayer extends layer.Layer(layer.LayerConfig.Extract())
    class Bar(grandParentData: Data) extends RawModule {
      val out = IO(Bool())
      out := BoringUtils.tapAndRead(grandParentData)
    }
    class Foo(parentData: Data) extends RawModule {
      layer.block(TestLayer) {
        val bar = Module(new Bar(parentData))
      }
    }
    class Top extends RawModule {
      val parentWire = Wire(Bool())
      parentWire := DontCare
      val foo = Module(new Foo(parentWire))
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top)

    // The connect should be at the end of the layerblock.
    matchesAndOmits(chirrtl)(
      "    layerblock TestLayer :",
      "      inst bar of Bar",
      "      connect bar.out_bore, out_bore"
    )()

    // Check is valid FIRRTL and builds to SV.
    circt.stage.ChiselStage.emitSystemVerilog(new Top)
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

  it should "work for identity views" in {
    import chisel3.experimental.dataview._
    class Foo extends RawModule {
      private val internalWire = Wire(Bool())
      val view = internalWire.viewAs[Bool]
    }
    class Top extends RawModule {
      val foo = Module(new Foo)
      val outProbe = IO(probe.Probe(Bool()))
      val out = IO(Bool())
      probe.define(outProbe, BoringUtils.tap(foo.view))
      out := BoringUtils.tapAndRead(foo.view)
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

  it should "NOT work [yet] for non-identity views" in {
    import chisel3.experimental.dataview._
    class MyBundle extends Bundle {
      val a = Bool()
      val b = Bool()
    }
    object MyBundle {
      implicit val view: DataView[(Bool, Bool), MyBundle] = DataView(
        _ => new MyBundle,
        _._1 -> _.a,
        _._2 -> _.b
      )
    }
    class Foo extends RawModule {
      private val w1, w2 = Wire(Bool())
      val view = (w1, w2).viewAs[MyBundle]
    }
    class Top extends RawModule {
      val foo = Module(new Foo)
      val out = IO(new MyBundle)
      val outProbe = IO(probe.Probe(new MyBundle))
      probe.define(outProbe, BoringUtils.tap(foo.view))
      out := BoringUtils.tapAndRead(foo.view)
    }
    val e = the[ChiselException] thrownBy circt.stage.ChiselStage.emitCHIRRTL(new Top)
    e.getMessage should include("BoringUtils currently only support identity views")
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
        class MiniBundle extends Bundle {
          val x = Bool()
        }
        class Child() extends RawModule {
          val b = Wire(new MiniBundle)
        }

        val child = Module(new Child())

        // directly tap Bundle element
        val outRWProbe = IO(probe.RWProbe(Bool()))
        probe.define(outRWProbe, BoringUtils.rwTap(child.b.x))

        // tap Bundle, then access element
        val outRWBundleProbe = IO(probe.RWProbe(new MiniBundle))
        val outElem = IO(probe.RWProbe(Bool()))
        probe.define(outRWBundleProbe, BoringUtils.rwTap(child.b))
        probe.define(outElem, outRWBundleProbe.x)
      }
    )
    matchesAndOmits(chirrtl)(
      "wire b : { x : UInt<1>}",
      "define bore = rwprobe(b.x)",
      "define bore_1 = rwprobe(b)",
      "define outRWProbe = child.bore",
      "define outRWBundleProbe = child.bore_1",
      "define outElem = outRWBundleProbe.x"
    )()
  }

  it should "work when tapping an element within a Vec" in {
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(
      new RawModule {
        class Child() extends RawModule {
          val b = Wire(Vec(4, Bool()))
        }

        val child = Module(new Child())

        // directly tap Vec element
        val outRWProbe = IO(probe.RWProbe(Bool()))
        probe.define(outRWProbe, BoringUtils.rwTap(child.b(2)))

        // tap Vec, then access element
        val outRWVecProbe = IO(probe.RWProbe(Vec(4, Bool())))
        val outElem = IO(probe.RWProbe(Bool()))
        probe.define(outRWVecProbe, BoringUtils.rwTap(child.b))
        probe.define(outElem, outRWVecProbe(1))
      }
    )
    matchesAndOmits(chirrtl)(
      "wire b : UInt<1>[4]",
      "define bore = rwprobe(b[2])",
      "define bore_1 = rwprobe(b)",
      "define outRWProbe = child.bore",
      "define outRWVecProbe = child.bore_1",
      "define outElem = outRWVecProbe[1]"
    )()
  }

  it should "work when rw-tapping IO, as rwprobe() from inside module" in {
    class Foo extends RawModule {
      class InOutBundle extends Bundle {
        val in = Flipped(Bool())
        val out = Bool()
      }
      class Child() extends RawModule {
        val v = IO(Vec(2, new InOutBundle))
        v(0).out := v(0).in
        v(1).out := v(1).in
      }

      val inputs = IO(Flipped(Vec(2, Bool())))
      val child = Module(new Child())
      child.v(0).in := inputs(0)
      child.v(1).in := inputs(1)

      // Directly rwTap field of bundle within vector.
      val outV_0_out = IO(probe.RWProbe(Bool()))
      probe.define(outV_0_out, BoringUtils.rwTap(child.v(0).out))

      // Also rwTap flipped field (input port).
      val outV_1_in = IO(probe.RWProbe(Bool()))
      probe.define(outV_1_in, BoringUtils.rwTap(child.v(1).in))
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Foo)
    matchesAndOmits(chirrtl)(
      // Child
      "output v : { flip in : UInt<1>, out : UInt<1>}[2]",
      "define bore = rwprobe(v[0].out)",
      "define bore_1 = rwprobe(v[1].in)",
      // Forwarding probes out from instantiating module.
      "define outV_0_out = child.bore",
      "define outV_1_in = child.bore_1"
    )()
    // Send through firtool and lightly check output.
    // Bit fragile across firtool versions.
    val sv = circt.stage.ChiselStage.emitSystemVerilog(new Foo)
    matchesAndOmits(sv)(
      // Child ports.
      "module Child(",
      "input  v_0_in,",
      "output v_0_out,",
      "input  v_1_in",
      // Instantiation.
      "Child child (",
      ".v_0_in  (inputs_0),", // Alive because feeds outV_0_out probe.
      ".v_0_out (", // rwprobe target.
      ".v_1_in  (inputs_1)", // rwprobe target.
      // Ref ABI.  Names of internal signals are subject to change.
      "`define ref_Foo_outV_0_out child.v_0_out",
      "`define ref_Foo_outV_1_in child.v_1_in"
    )("v_1_out")
  }

  it should "work when tapping IO, as probe() from outside module" in {
    class Foo extends RawModule {
      class InOutBundle extends Bundle {
        val in = Flipped(Bool())
        val out = Bool()
      }
      class Child() extends RawModule {
        val v = IO(Vec(2, new InOutBundle))
        v(0).out := v(0).in
        v(1).out := v(1).in
      }

      val inputs = IO(Flipped(Vec(2, Bool())))
      val child = Module(new Child())
      child.v(0).in := inputs(0)
      child.v(1).in := inputs(1)

      // Directly tap entire vector of bundles.
      val outProbeForChildVec = IO(probe.Probe(Vec(2, new InOutBundle)))
      probe.define(outProbeForChildVec, BoringUtils.tap(child.v))

      // Also tap specific leaf.
      val outV_1_in = IO(probe.Probe(Bool()))
      probe.define(outV_1_in, BoringUtils.tap(child.v(1).in))

      // Index through probe of aggregate to sibling leaf.
      val outV_1_out_refsub = IO(probe.Probe(Bool()))
      probe.define(outV_1_out_refsub, outProbeForChildVec(1).out)
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Foo)
    matchesAndOmits(chirrtl)(
      // Child port.
      "output v : { flip in : UInt<1>, out : UInt<1>}[2]",
      // Probes in terms of child instance ports.
      "define outProbeForChildVec = probe(child.v)",
      "define outV_1_in = probe(child.v[1].in)",
      "define outV_1_out_refsub = outProbeForChildVec[1].out"
    )("define bore")

    // Send through firtool but don't inspect output.
    // Read-only probes only ensure they'll read same as in input FIRRTL,
    // and so there may be significant churn here.
    // Simulation should always read same values.
    circt.stage.ChiselStage.emitSystemVerilog(new Foo)
  }

  it should "work with D/I" in {
    import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}
    @instantiable trait FooInterface {
      @public val tapTarget: Bool = IO(probe.RWProbe(Bool()))
    }
    class Foo extends RawModule with FooInterface {
      val internalWire = Wire(Bool())
      internalWire := DontCare

      probe.define(tapTarget, BoringUtils.rwTap(internalWire))
    }
    class Top(fooDef: Definition[Foo]) extends RawModule {
      val fooInstA = Instance(fooDef)
      val fooInstB = Instance(fooDef)

      probe.forceInitial(fooInstA.tapTarget, true.B)

      val outProbe = IO(probe.RWProbe(Bool()))
      probe.define(outProbe, fooInstB.tapTarget)
    }
    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top(Definition(new Foo)), Array("--full-stacktrace"))
    matchesAndOmits(chirrtl)(
      "module Foo :",
      "output tapTarget : RWProbe<UInt<1>>",
      "define tapTarget = rwprobe(internalWire)",
      "module Top :",
      "force_initial(fooInstA.tapTarget, UInt<1>(0h1))",
      "define outProbe = fooInstB.tapTarget"
    )()

    // Check that firtool also passes
    val verilog = circt.stage.ChiselStage.emitSystemVerilog(new Top(Definition(new Foo)))
  }

  it should "work with DecoupledIO in a hierarchy" in {
    import chisel3.util.{Decoupled, DecoupledIO}
    class Bar() extends RawModule {
      val decoupledThing = Wire(Decoupled(Bool()))
      decoupledThing := DontCare
    }
    class Foo() extends RawModule {
      val bar = Module(new Bar())
    }
    class FakeView(foo: Foo) extends RawModule {
      val decoupledThing = Wire(DecoupledIO(Bool()))
      decoupledThing := BoringUtils.tapAndRead(foo.bar.decoupledThing)
    }
    class Top() extends RawModule {
      val foo = Module(new Foo())
      val fakeView = Module(new FakeView(foo))
    }

    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Top(), Array("--full-stacktrace"))
    matchesAndOmits(chirrtl)(
      "module Bar :",
      "output decoupledThing_bore : Probe<{ ready : UInt<1>, valid : UInt<1>, bits : UInt<1>}>",
      "define decoupledThing_bore = probe(decoupledThing)",
      "module Foo :",
      "output decoupledThing_bore : Probe<{ ready : UInt<1>, valid : UInt<1>, bits : UInt<1>}>",
      "define decoupledThing_bore = bar.decoupledThing_bore",
      "module FakeView :",
      "input decoupledThing_bore : { ready : UInt<1>, valid : UInt<1>, bits : UInt<1>}",
      "module Top :",
      "connect fakeView.decoupledThing_bore, read(foo.decoupledThing_bore)"
    )()

    // Check that firtool also passes
    val verilog = circt.stage.ChiselStage.emitSystemVerilog(new Top())
  }

  it should "work with DecoupledIO" in {
    import chisel3.util.{Decoupled, DecoupledIO}
    class Bar(b: DecoupledIO[Bool]) extends RawModule {
      BoringUtils.tapAndRead(b)
    }

    class Foo extends RawModule {
      val a = WireInit(DecoupledIO(Bool()), DontCare)
      val bar = Module(new Bar(a))
    }

    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Foo, Array("--full-stacktrace"))

    matchesAndOmits(chirrtl)(
      "module Bar :",
      "input bore : { ready : UInt<1>, valid : UInt<1>, bits : UInt<1>}",
      "module Foo :",
      "wire a : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<1>}",
      "connect bar.bore, read(probe(a))"
    )()

    // Check that firtool also passes
    val verilog = circt.stage.ChiselStage.emitSystemVerilog(new Foo)
  }

  it should "work with DecoupledIO locally" in {
    import chisel3.util.{Decoupled, DecoupledIO}
    class Foo extends RawModule {
      val a = WireInit(DecoupledIO(Bool()), DontCare)
      val b = BoringUtils.tapAndRead(a)
      assert(chisel3.reflect.DataMirror.isFullyAligned(b), "tapAndRead should always return passive data")
    }

    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Foo, Array("--full-stacktrace"))

    matchesAndOmits(chirrtl)(
      "module Foo :",
      "wire a : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<1>}",
      "wire b : { ready : UInt<1>, valid : UInt<1>, bits : UInt<1>}",
      "connect b.bits, a.bits",
      "connect b.valid, a.valid",
      "connect b.ready, a.ready"
    )()

    // Check that firtool also passes
    val verilog = circt.stage.ChiselStage.emitSystemVerilog(new Foo)
  }

  it should "allow tapping a probe" in {
    class Bar extends RawModule {
      val a = IO(probe.Probe(Bool()))
    }
    class Foo extends RawModule {
      val b = IO(probe.Probe(Bool()))
      val bar = Module(new Bar)
      probe.define(b, BoringUtils.tap(bar.a))
    }

    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Foo)

    matchesAndOmits(chirrtl)(
      "define b = bar.a"
    )()
  }

  it should "work for identity views" in {
    import chisel3.experimental.dataview._
    class Bar extends RawModule {
      private val internalWire = Wire(Bool())
      val view = internalWire.viewAs[Bool]
    }
    class Foo extends RawModule {
      val bar = Module(new Bar)
    }
    class Top extends RawModule {
      val foo = Module(new Foo)
      val out = IO(Bool())
      out := probe.read(BoringUtils.rwTap(foo.bar.view))
      probe.forceInitial(BoringUtils.rwTap(foo.bar.view), false.B)
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

  it should "NOT work [yet] for non-identity views" in {
    import chisel3.experimental.dataview._
    class MyBundle extends Bundle {
      val a = Bool()
      val b = Bool()
    }
    object MyBundle {
      implicit val view: DataView[(Bool, Bool), MyBundle] = DataView(
        _ => new MyBundle,
        _._1 -> _.a,
        _._2 -> _.b
      )
    }
    class Bar extends RawModule {
      private val w1, w2 = Wire(Bool())
      val view = (w1, w2).viewAs[MyBundle]
    }
    class Foo extends RawModule {
      val bar = Module(new Bar)
    }
    class Top extends RawModule {
      val foo = Module(new Foo)
      val out = IO(Bool())
      out := probe.read(BoringUtils.rwTap(foo.bar.view))
      probe.forceInitial(BoringUtils.rwTap(foo.bar.view), false.B)
    }
    val e = the[ChiselException] thrownBy circt.stage.ChiselStage.emitCHIRRTL(new Top)
    e.getMessage should include("BoringUtils currently only support identity views")
  }
}
