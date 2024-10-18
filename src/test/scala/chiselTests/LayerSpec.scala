// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chiselTests.{ChiselFlatSpec, MatchesAndOmits, Utils}
import _root_.circt.stage.ChiselStage

class LayerSpec extends ChiselFlatSpec with Utils with MatchesAndOmits {

  "Layers" should "allow for creation of a layer and nested layers" in {

    object A extends layer.Layer(layer.Convention.Bind) {
      object B extends layer.Layer(layer.Convention.Bind)
    }

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

    val chirrtl = ChiselStage.emitCHIRRTL(new Foo, Array("--full-stacktrace"))

    info("CHIRRTL emission looks coorect")
    // TODO: Switch to FileCheck for this testing.  This is going to miss all sorts of ordering issues.
    matchesAndOmits(chirrtl)(
      "declgroup A, bind :",
      "declgroup B, bind :",
      "group A :",
      "wire w : UInt<1>",
      "group B :",
      "wire x : UInt<1>",
      "wire y : UInt<1>"
    )()
  }

  "Layers error checking" should "require that a nested layer definition matches its declaration nesting" in {

    object A extends layer.Layer(layer.Convention.Bind) {
      object B extends layer.Layer(layer.Convention.Bind)
    }

    class Foo extends RawModule {
      layer.block(A.B) {
        val a = Wire(Bool())
      }
    }

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new Foo) }.getMessage() should include(
      "nested layer 'B' must be wrapped in parent layer 'A'"
    )

  }

<<<<<<< HEAD
=======
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

    intercept[ChiselException] { ChiselStage.convert(new Foo, Array("--throw-on-first-error")) }
      .getMessage() should include(
      "Cannot define 'Foo.a: IO[Probe[A]<Bool>]' from colors {'C'} since at least one of these is NOT enabled when 'Foo.a: IO[Probe[A]<Bool>]' is enabled"
    )
  }

  "Inline layers" should "generated expected FIRRTL" in {
    object A extends layer.Layer(layer.LayerConfig.Inline) {
      object B extends layer.Layer(layer.LayerConfig.Inline)
    }

    class Foo extends RawModule {
      layer.block(A) {
        layer.block(A.B) {}
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "layer A, inline :",
      "layer B, inline :"
    )()
  }

  "Inline layers" should "be ignored when choosing default output directories" in {
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

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      bindLayer("LayerWithDefaultOutputDir", List("LayerWithDefaultOutputDir")),
      inlineLayer("InlineSublayer"),
      bindLayer("SublayerWithDefaultOutputDir", List("LayerWithDefaultOutputDir", "SublayerWithDefaultOutputDir")),
      bindLayer("LayerWithCustomOutputDir", List("myOutputDir")),
      inlineLayer("InlineSublayer"),
      bindLayer("SublayerWithDefaultOutputDir", List("myOutputDir", "SublayerWithDefaultOutputDir")),
      bindLayer("LayerWithNoOutputDir", List()),
      inlineLayer("InlineSublayer"),
      bindLayer("SublayerWithDefaultOutputDir", List("SublayerWithDefaultOutputDir"))
    )()
  }
>>>>>>> 8c718b2b6 (Add Probes to .toString Data methods (#4478))
}
