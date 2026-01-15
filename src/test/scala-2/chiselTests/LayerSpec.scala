// SPDX-License-Identifier: Apache-2.0

package chiselTests

import _root_.circt.stage.ChiselStage
import chisel3._
import chisel3.experimental.hierarchy.core.{Definition, Instance}
import chisel3.experimental.hierarchy.instantiable
import chisel3.ltl.AssertProperty
import chisel3.ltl.Sequence._
import chisel3.probe.{define, Probe, ProbeValue}
import chisel3.reflect.DataMirror.internal.chiselTypeClone
import chisel3.simulator.{LayerControl, Settings}
import chisel3.simulator.scalatest.ChiselSim
import chisel3.testing.scalatest.FileCheck
import java.nio.file.{FileSystems, Paths}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LayerSpec extends AnyFlatSpec with Matchers with FileCheck with ChiselSim {

  val sep: String = FileSystems.getDefault().getSeparator()

  object A extends layer.Layer(layer.LayerConfig.Extract()) {
    object B extends layer.Layer(layer.LayerConfig.Extract())
  }

  object C extends layer.Layer(layer.LayerConfig.Extract())

  "Layers" should "allow for creation of a layer and nested layers" in {

    class Foo extends RawModule {
      val a = IO(Input(Bool()))

      layer.block(A) {
        val w = WireInit(a)
        dontTouch(w)
        layer.block(A.B) {
          val x = WireInit(w)
          dontTouch(x)
        }
      }

      val y = Wire(Bool())
      y := DontCare
    }

    info("CHIRRTL emission looks correct")
    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      s"""|CHECK:      layer A, bind, "A" :
          |CHECK-NEXT:   layer B, bind, "A${sep}B" :
          |
          |CHECK:      module Foo :
          |CHECK:        layerblock A :
          |CHECK-NEXT:     wire w : UInt<1>
          |CHECK:          layerblock B :
          |CHECK-NEXT:       wire x : UInt<1>
          |CHECK:        wire y : UInt<1>
          |""".stripMargin
    }
  }

  they should "create parent layer blocks automatically" in {

    class Foo extends RawModule {
      layer.block(A.B) {}
      layer.block(C) {
        layer.block(C) {}
      }
    }

    val check =
      ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
        """|CHECK: layerblock A :
           |CHECK:   layerblock B :
           |CHECK:     layerblock C :
           |""".stripMargin
      }
  }

  they should "respect the 'skipIfAlreadyInBlock' parameter" in {
    class Foo extends RawModule {
      layer.block(A, skipIfAlreadyInBlock = true) {
        // This will fail to compile if `skipIfAlreadyInBlock=false`.
        layer.block(C, skipIfAlreadyInBlock = true) {}
      }
    }

    val check =
      ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
        """|CHECK:     layerblock A :
           |CHECK-NOT:   layerblock C :
           |""".stripMargin
      }
  }

  they should "respect the 'skipIfLayersEnabled' parameter" in {
    class Foo extends RawModule {
      layer.enable(A)
      layer.block(A.B, skipIfLayersEnabled = true) {}
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck()("CHECK-NOT: layerblock")
  }

  they should "create no layer blocks when wrapped in 'elideBlocks'" in {
    class Foo extends RawModule {
      layer.elideBlocks {
        layer.block(A.B) {}
      }
    }
    ChiselStage.emitCHIRRTL(new Foo).fileCheck()("CHECK-NOT: layerblock")
  }

  they should "not get confused if `elideBlocks` is called multiple times" in {
    class Foo extends RawModule {
      layer.elideBlocks {
        layer.elideBlocks {}
        layer.block(A.B) {}
      }
    }
    ChiselStage.emitCHIRRTL(new Foo).fileCheck()("CHECK-NOT: layerblock")
  }

  they should "generate valid CHIRRTL when module instantiated under layer block has layer blocks" in {
    object A extends layer.Layer(layer.LayerConfig.Inline) {
      object B extends layer.Layer(layer.LayerConfig.Inline)
    }
    class Bar extends RawModule {
      layer.block(A.B) {
        val w = WireInit(Bool(), true.B)
      }
    }

    class Foo extends RawModule {
      layer.block(A) {
        val bar = Module(new Bar())
      }
    }

    // Check the generated CHIRRTL only.
    // Layer-under-module-under-layer is rejected by firtool presently.
    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:      module Bar :
         |CHECK:        layerblock A :
         |CHECK-NEXT:     layerblock B :
         |
         |CHECK:      module Foo :
         |CHECK:        layerblock A :
         |CHECK-NEXT:     inst bar of Bar
         |""".stripMargin
    }
  }

  they should "allow for defines to layer-colored probes" in {

    class Foo extends RawModule {
      val a, b, c, d, e = IO(Input(Bool()))
      val x = IO(Output(Probe(Bool(), A)))
      val y = IO(Output(Probe(Bool(), A.B)))
      define(x, ProbeValue(a))
      define(y, ProbeValue(b))
      layer.block(A) {
        define(x, ProbeValue(c))
        define(y, ProbeValue(d))
        layer.block(A.B) {
          define(y, ProbeValue(e))
        }
      }
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:      module Foo :
         |CHECK:        define x = probe(a)
         |CHECK-NEXT:   define y = probe(b)
         |CHECK:        layerblock A :
         |CHECK-NEXT:     define x = probe(c)
         |CHECK-NEXT:     define y = probe(d)
         |CHECK:          layerblock B :
         |CHECK-NEXT:       define y = probe(e)
         |""".stripMargin
    }
  }

  they should "allow for defines to layer-colored probes without layer blocks" in {

    class Foo extends RawModule {
      val a, b = IO(Input(Bool()))
      val x = IO(Output(Probe(Bool(), A)))
      val y = IO(Output(Probe(Bool(), A.B)))
      define(x, ProbeValue(a))
      define(y, ProbeValue(b))
    }

    ChiselStage.elaborate(new Foo)
  }

  they should "allow for defines to layer-colored probes regardless of enabled layers" in {

    class Foo extends RawModule {
      val a, b = IO(Input(Bool()))
      val x = IO(Output(Probe(Bool(), A)))
      val y = IO(Output(Probe(Bool(), A.B)))
      layer.enable(C)
      define(x, ProbeValue(a))
      define(y, ProbeValue(b))
    }

    ChiselStage.elaborate(new Foo)
  }

  they should "be enabled with a function" in {

    class Foo extends RawModule {
      layer.enable(A.B)
      layer.enable(C)
      // This should be a no-op.
      layer.enable(layer.Layer.root)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:      layer A, bind
         |CHECK-NEXT:   layer B, bind
         |CHECK-NEXT: layer C, bind
         |
         |CHECK: module Foo enablelayer A.B enablelayer C :
         |""".stripMargin
    }

  }

  they should ", when enabled, not propagate to child modules" in {

    class Bar extends RawModule

    class Foo extends RawModule {
      layer.enable(A)
      val bar = Module(new Bar)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: module Bar :
         |CHECK: module Foo enablelayer A :
         |""".stripMargin
    }

  }

  they should "work correctly with Definition/Instance" in {

    @instantiable
    class Bar extends Module {
      layer.block(A) {}
    }

    class Foo extends Module {
      private val bar = Instance(Definition(new Bar))
    }

    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Foo)
    chirrtl should include("layer A")
  }

  they should "emit the output directory when present" in {
    object LayerWithDefaultOutputDir extends layer.Layer(layer.LayerConfig.Extract()) {
      object SublayerWithDefaultOutputDir extends layer.Layer(layer.LayerConfig.Extract()) {}
      object SublayerWithCustomOutputDir
          extends layer.Layer(layer.LayerConfig.Extract(layer.CustomOutputDir(Paths.get("myOtherOutputDir")))) {}
      object SublayerWithNoOutputDir extends layer.Layer(layer.LayerConfig.Extract(layer.NoOutputDir)) {}
    }

    object LayerWithCustomOutputDir
        extends layer.Layer(layer.LayerConfig.Extract(layer.CustomOutputDir(Paths.get("myOutputDir")))) {
      object SublayerWithDefaultOutputDir extends layer.Layer(layer.LayerConfig.Extract()) {}
      object SublayerWithCustomOutputDir
          extends layer.Layer(layer.LayerConfig.Extract(layer.CustomOutputDir(Paths.get("myOtherOutputDir")))) {}
      object SublayerWithNoOutputDir extends layer.Layer(layer.LayerConfig.Extract(layer.NoOutputDir)) {}
    }

    object LayerWithNoOutputDir extends layer.Layer(layer.LayerConfig.Extract(layer.NoOutputDir)) {
      object SublayerWithDefaultOutputDir extends layer.Layer(layer.LayerConfig.Extract()) {}
      object SublayerWithCustomOutputDir
          extends layer.Layer(layer.LayerConfig.Extract(layer.CustomOutputDir(Paths.get("myOtherOutputDir")))) {}
      object SublayerWithNoOutputDir extends layer.Layer(layer.LayerConfig.Extract(layer.NoOutputDir)) {}
    }

    class Foo extends RawModule {
      layer.block(LayerWithDefaultOutputDir) {
        layer.block(LayerWithDefaultOutputDir.SublayerWithDefaultOutputDir) {}
        layer.block(LayerWithDefaultOutputDir.SublayerWithCustomOutputDir) {}
        layer.block(LayerWithDefaultOutputDir.SublayerWithNoOutputDir) {}
      }

      layer.block(LayerWithCustomOutputDir) {
        layer.block(LayerWithCustomOutputDir.SublayerWithDefaultOutputDir) {}
        layer.block(LayerWithCustomOutputDir.SublayerWithCustomOutputDir) {}
        layer.block(LayerWithCustomOutputDir.SublayerWithNoOutputDir) {}
      }

      layer.block(LayerWithNoOutputDir) {
        layer.block(LayerWithNoOutputDir.SublayerWithDefaultOutputDir) {}
        layer.block(LayerWithNoOutputDir.SublayerWithCustomOutputDir) {}
        layer.block(LayerWithNoOutputDir.SublayerWithNoOutputDir) {}
      }
    }

    def decl(name: String, dirs: List[String]): String = {
      val dirsStr = if (dirs.nonEmpty) s""", "${dirs.mkString(sep)}"""" else ""
      s"layer $name, bind$dirsStr :"
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      s"""|CHECK:      circuit Foo :
          |CHECK:        layer LayerWithDefaultOutputDir, bind, "LayerWithDefaultOutputDir" :
          |CHECK-NEXT:     layer SublayerWithDefaultOutputDir, bind, "LayerWithDefaultOutputDir${sep}SublayerWithDefaultOutputDir" :
          |CHECK-NEXT:     layer SublayerWithCustomOutputDir, bind, "myOtherOutputDir" :
          |CHECK-NEXT:     layer SublayerWithNoOutputDir, bind :
          |CHECK-NEXT:   layer LayerWithCustomOutputDir, bind, "myOutputDir" :
          |CHECK-NEXT:     layer SublayerWithDefaultOutputDir, bind, "myOutputDir${sep}SublayerWithDefaultOutputDir"
          |CHECK-NEXT:     layer SublayerWithCustomOutputDir, bind, "myOtherOutputDir" :
          |CHECK-NEXT:     layer SublayerWithNoOutputDir, bind :
          |CHECK-NEXT:   layer LayerWithNoOutputDir, bind :
          |CHECK-NEXT:     layer SublayerWithDefaultOutputDir, bind, "SublayerWithDefaultOutputDir" :
          |CHECK-NEXT:     layer SublayerWithCustomOutputDir, bind, "myOtherOutputDir" :
          |CHECK-NEXT:     layer SublayerWithNoOutputDir, bind :
          |""".stripMargin
    }
  }

  they should "allow manually overriding the parent layer" in {

    implicit val Parent = layers.Verification
    object ExpensiveAsserts extends layer.Layer(layer.LayerConfig.Extract())

    class Foo extends RawModule {
      layer.addLayer(ExpensiveAsserts)
    }

    ChiselStage.emitCHIRRTL(new Foo) should include("""layer ExpensiveAsserts, bind, "verification/ExpensiveAsserts"""")
  }

  "Values returned by layer blocks" should "be layer-colored wires" in {
    class Foo extends RawModule {
      val out = IO {
        Output {
          new Bundle {
            val a = Probe(UInt(1.W), A)
            val b = Probe(UInt(2.W), A.B)
            val c = Probe(UInt(3.W), A.B)
            val d = Probe(UInt(4.W), A)
          }
        }
      }

      // A single layer block
      val a = layer.block(A) {
        Wire(UInt(1.W))
      }
      define(out.a, a)

      // Auto-creation of parent layer blocks
      val b = layer.block(A.B) {
        Wire(UInt(2.W))
      }
      define(out.b, b)

      // Nested layer blocks
      val c = layer.block(A) {
        layer.block(A.B) {
          Wire(UInt(3.W))
        }
      }
      define(out.c, c)

      // Return of ProbeValue
      val d = layer.block(A) {
        ProbeValue(Wire(UInt(4.W)))
      }
      define(out.d, d)

      // No layers created.  Check all generator code paths.
      layer.block(A) {
        // Path 1: `skipIfAlreadyInBlock` set
        val e = layer.block(C, skipIfAlreadyInBlock = true) {
          Wire(UInt(5.W))
        }
        // Path 2: requested layer is the current layer
        val f = layer.block(A) {
          Wire(UInt(6.W))
        }
      }
      // Path 3: `skipIfLayersEnabled` is set and a layer is enabled
      layer.enable(C)
      val g = layer.block(C, skipIfLayersEnabled = true) {
        Wire(UInt(7.W))
      }
      // Path 4: the `elideBlocks` API is used
      val h = layer.elideBlocks {
        layer.block(A) {
          Wire(UInt(8.W))
        }
      }

      // Return non-`Data` as `Unit`.
      val i = layer.block(A) {
        42
      }
      assert(i.getClass() == classOf[Unit])

      // Return `Unit` as `Unit`.
      val j = layer.block(A) { () }
      assert(j.getClass() == classOf[Unit])
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      s"""|CHECK:      module Foo
          |CHECK:        wire a : Probe<UInt<1>, A>
          |CHECK-NEXT:   layerblock A :
          |CHECK-NEXT:     wire [[a:.*]] : UInt<1>
          |CHECK-NEXT:     define a = probe([[a]])
          |CHECK:        define out.a = a
          |
          |CHECK:        wire b : Probe<UInt<2>, A.B>
          |CHECK-NEXT:   layerblock A :
          |CHECK-NEXT:     layerblock B :
          |CHECK-NEXT:       wire [[b:.*]] : UInt<2>
          |CHECK-NEXT:       define b = probe([[b]])
          |CHECK:        define out.b = b
          |
          |CHECK:        wire c : Probe<UInt<3>, A.B>
          |CHECK-NEXT:   layerblock A :
          |CHECK-NEXT:     wire [[c0:.*]] : Probe<UInt<3>, A.B>
          |CHECK-NEXT:     layerblock B :
          |CHECK-NEXT:       wire [[c:.*]] : UInt<3>
          |CHECK-NEXT:       define [[c0]] = probe([[c]])
          |CHECK:          define c = [[c0]]
          |CHECK:        define out.c = c
          |
          |CHECK:        wire d : Probe<UInt<4>, A>
          |CHECK-NEXT:   layerblock A :
          |CHECK-NEXT:     wire [[d:.*]] : UInt<4>
          |CHECK-NEXT:     define d = probe([[d]])
          |CHECK:        define out.d = d
          |
          |CHECK:        layerblock A :
          |CHECK-NEXT:     wire e : UInt<5>
          |CHECK-NEXT:     wire f : UInt<6>
          |CHECK-NOT:    layerblock
          |CHECK:        wire g : UInt<7>
          |CHECK-NOT:    layerblock
          |CHECK:        wire h : UInt<8>
          |""".stripMargin
    }
  }

  they should "require compatible color if returning a layer-colored wire" in {
    class Foo extends RawModule {

      val a = layer.block(A) {
        Wire(Probe(Bool(), C))
      }

    }

    intercept[ChiselException] { ChiselStage.elaborate(new Foo, Array("--throw-on-first-error")) }
      .getMessage() should include("cannot return probe of color 'C' from a layer block associated with layer 'A'")
  }

  "addLayer API" should "add a layer to the output CHIRRTL even if no layer block references that layer" in {
    class Foo extends RawModule {
      layer.addLayer(A)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:     layer A
         |CHECK-NOT:   layerblock
         |""".stripMargin
    }
  }

  "Default Layers" should "always be emitted" in {
    class Foo extends RawModule {}
    val chirrtl = ChiselStage.emitCHIRRTL(new Foo)

    info("default layers are emitted")
    chirrtl.fileCheck() {
      s"""|CHECK:      layer Verification, bind, "verification" :
          |CHECK-NEXT:   layer Assert, bind, "verification${sep}assert" :
          |CHECK-NEXT:     layer Temporal, inline :
          |CHECK-NEXT:   layer Assume, bind, "verification${sep}assume" :
          |CHECK-NEXT:     layer Temporal, inline :
          |CHECK-NEXT:   layer Cover, bind, "verification${sep}cover" :
          |CHECK-NEXT:     layer Temporal, inline :
          |""".stripMargin
    }

    info("user-defined layers are not emitted if not used")
    (chirrtl should not).include("layer B")
  }

  they should "include a Temporal inline layers for Assert, Assume, and Cover" in {
    class Foo extends RawModule {
      layer.block(layers.Verification.Assert.Temporal) {
        val a = Wire(UInt(1.W))
      }
      layer.block(layers.Verification.Assume.Temporal) {
        val a = Wire(UInt(2.W))
      }
      layer.block(layers.Verification.Cover.Temporal) {
        val a = Wire(UInt(3.W))
      }
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      s"""|CHECK:      layer Verification, {{.*}} :
          |CHECK-NEXT:   layer Assert, {{.*}} :
          |CHECK-NEXT:     layer Temporal, inline :
          |CHECK-NEXT:   layer Assume, {{.*}} :
          |CHECK-NEXT:     layer Temporal, inline :
          |CHECK-NEXT:   layer Cover, {{.*}} :
          |CHECK-NEXT:     layer Temporal, inline :
          |""".stripMargin
    }
  }

  "Layers error checking" should "require that the current layer is an ancestor of the desired layer" in {

    class Foo extends RawModule {
      layer.block(A.B) {
        layer.block(C) {
          val a = Wire(Bool())
        }
      }
    }

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new Foo) }.getMessage() should include(
      "a layerblock associated with layer 'C' cannot be created under a layerblock of non-ancestor layer 'A.B'"
    )

  }

  it should "check that a define from inside a layerblock is to a legal layer-colored probe" in {
    class Foo extends RawModule {
      val a = IO(Output(Probe(Bool(), A)))
      layer.block(C) {
        val b = Wire(Bool())
        define(a, ProbeValue(b))
      }
    }
    intercept[ChiselException] { ChiselStage.emitCHIRRTL(new Foo, Array("--throw-on-first-error")) }
      .getMessage() should include(
      "Cannot define 'Foo.a: IO[Probe[A]<Bool>]' from colors {'C'} since at least one of these is NOT enabled when 'Foo.a: IO[Probe[A]<Bool>]' is enabled"
    )
  }

  it should "not consider enabled layers in error" in {
    class Foo extends RawModule {
      layer.enable(A)
      val a = IO(Output(Probe(Bool(), A)))
      layer.block(C) {
        val b = Wire(Bool())
        define(a, ProbeValue(b))
      }
    }

    intercept[ChiselException] { ChiselStage.elaborate(new Foo, Array("--throw-on-first-error")) }
      .getMessage() should include(
      "Cannot define 'Foo.a: IO[Probe[A]<Bool>]' from colors {'C'} since at least one of these is NOT enabled when 'Foo.a: IO[Probe[A]<Bool>]' is enabled"
    )
  }

  "Inline layers" should "generate expected FIRRTL and SystemVerilog" in {
    object A extends layer.Layer(layer.LayerConfig.Extract()) {
      object B extends layer.Layer(layer.LayerConfig.Inline) {
        object C extends layer.Layer(layer.LayerConfig.Inline)
      }
    }

    class Foo extends Module {
      val a = IO(Input(UInt(2.W)))

      layer.block(A) {
        AssertProperty(a > 0.U, "foo")
        layer.block(A.B) {
          AssertProperty(a > 1.U, "bar")
          layer.block(A.B.C) {
            AssertProperty(a > 2.U, "baz")
          }
        }
      }
    }

    info("FIRRTL okay")
    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:      layer A, bind
         |CHECK-NEXT:   layer B, inline :
         |CHECK-NEXT:     layer C, inline :
         |""".stripMargin
    }

    info("SystemVerilog okay")
    val verilog = ChiselStage.emitSystemVerilog(
      new Foo,
      firtoolOpts = Array(
        "-disable-all-randomization",
        "-enable-layers=Verification,Verification.Assert,Verification.Assume,Verification.Cover"
      )
    )
    verilog.fileCheck() {
      """|CHECK:      module Foo(
         |CHECK-NOT:    assert property
         |
         |CHECK:      module Foo_A(
         |CHECK-NOT:    `ifdef
         |CHECK:        foo: assert property
         |CHECK:        `ifdef layer$A$B
         |CHECK-NEXT:     bar: assert property
         |CHECK-NEXT:     `ifdef layer$A$B$C
         |CHECK-NEXT:       baz: assert property
         |CHECK-NEXT:     `endif
         |CHECK-NEXT:   `endif""".stripMargin
    }
  }

  they should "be ignored when choosing default output directories" in {
    object LayerWithDefaultOutputDir extends layer.Layer(layer.LayerConfig.Extract()) {
      object InlineSublayer extends layer.Layer(layer.LayerConfig.Inline) {
        object SublayerWithDefaultOutputDir extends layer.Layer(layer.LayerConfig.Extract()) {}
      }
    }

    object LayerWithCustomOutputDir
        extends layer.Layer(layer.LayerConfig.Extract(layer.CustomOutputDir(Paths.get("myOutputDir")))) {
      object InlineSublayer extends layer.Layer(layer.LayerConfig.Inline) {
        object SublayerWithDefaultOutputDir extends layer.Layer(layer.LayerConfig.Extract()) {}
      }
    }

    object LayerWithNoOutputDir extends layer.Layer(layer.LayerConfig.Extract(layer.NoOutputDir)) {
      object InlineSublayer extends layer.Layer(layer.LayerConfig.Inline) {
        object SublayerWithDefaultOutputDir extends layer.Layer(layer.LayerConfig.Extract()) {}
      }
    }

    class Foo extends RawModule {
      layer.block(LayerWithDefaultOutputDir) {
        layer.block(LayerWithDefaultOutputDir.InlineSublayer) {
          layer.block(LayerWithDefaultOutputDir.InlineSublayer.SublayerWithDefaultOutputDir) {}
        }
      }

      layer.block(LayerWithCustomOutputDir) {
        layer.block(LayerWithCustomOutputDir.InlineSublayer) {
          layer.block(LayerWithCustomOutputDir.InlineSublayer.SublayerWithDefaultOutputDir) {}
        }
      }

      layer.block(LayerWithNoOutputDir) {
        layer.block(LayerWithNoOutputDir.InlineSublayer) {
          layer.block(LayerWithNoOutputDir.InlineSublayer.SublayerWithDefaultOutputDir) {}
        }
      }
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      s"""|CHECK:      layer LayerWithDefaultOutputDir, bind, "LayerWithDefaultOutputDir" :
          |CHECK-NEXT:   layer InlineSublayer, inline :
          |CHECK-NEXT:     layer SublayerWithDefaultOutputDir, bind, "LayerWithDefaultOutputDir${sep}SublayerWithDefaultOutputDir" :
          |CHECK-NEXT: layer LayerWithCustomOutputDir, bind, "myOutputDir" :
          |CHECK-NEXT:   layer InlineSublayer, inline :
          |CHECK-NEXT:     layer SublayerWithDefaultOutputDir, bind, "myOutputDir${sep}SublayerWithDefaultOutputDir" :
          |CHECK-NEXT: layer LayerWithNoOutputDir, bind :
          |CHECK-NEXT:   layer InlineSublayer, inline :
          |CHECK-NEXT:     layer SublayerWithDefaultOutputDir, bind, "SublayerWithDefaultOutputDir" :
          |""".stripMargin
    }
  }

  "The Temporal layer" should "allow guarding s_eventually in Verilator" in {
    class Foo extends Module {
      val a, b = IO(Input(Bool()))

      layer.block(layers.Verification.Assert.Temporal) {
        AssertProperty(a |-> b.eventually)
      }

    }

    info("Verilator errors if s_eventually is included")
    intercept[Exception] {
      simulate(new Foo)(_ => {})
    }.getMessage.fileCheck() {
      """|CHECK: %Error-UNSUPPORTED
         |CHECK: s_eventually
         |""".stripMargin
    }

    info("disabling the temporal layer allows compilation to complete")
    simulate(
      new Foo,
      settings = Settings.default.copy(verilogLayers =
        LayerControl.Enable(layers.Verification.Assert, layers.Verification.Assume, layers.Verification.Cover)
      )
    )(_ => {})
  }

}
