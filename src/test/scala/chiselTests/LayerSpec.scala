// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.probe.{define, Probe, ProbeValue}
import chiselTests.{ChiselFlatSpec, MatchesAndOmits, Utils}
import _root_.circt.stage.ChiselStage

class LayerSpec extends ChiselFlatSpec with Utils with MatchesAndOmits {

  object A extends layer.Layer(layer.Convention.Bind) {
    object B extends layer.Layer(layer.Convention.Bind)
  }

  object C extends layer.Layer(layer.Convention.Bind)

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

    val chirrtl = ChiselStage.emitCHIRRTL(new Foo, Array("--full-stacktrace"))

    info("CHIRRTL emission looks coorect")
    // TODO: Switch to FileCheck for this testing.  This is going to miss all sorts of ordering issues.
    matchesAndOmits(chirrtl)(
      "layer A, bind :",
      "layer B, bind :",
      "layerblock A :",
      "wire w : UInt<1>",
      "layerblock B :",
      "wire x : UInt<1>",
      "wire y : UInt<1>"
    )()
  }

  it should "allow for defines to layer-colored probes" in {

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

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "define x = probe(a)",
      "define y = probe(b)",
      "define x = probe(c)",
      "define y = probe(d)",
      "define y = probe(e)"
    )()
  }

  they should "be enabled with a trait" in {

    class Foo extends RawModule {
      layer.enable(A.B)
      layer.enable(C)
      // This should be a no-op.
      layer.enable(layer.Layer.root)
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "layer A, bind",
      "module Foo enablelayer A.B enablelayer C :"
    )()

  }

  they should "allow to define additional layer-colored probes using enables" in {

    class Foo extends RawModule {
      // Without this enable, this circuit is illegal because `C` is NOT enabled
      // when `A` is enabled.  Cf. tests checking errors of this later.
      layer.enable(A)
      val a = IO(Output(Probe(Bool(), A)))
      layer.block(C) {
        val b = Wire(Bool())
        define(a, ProbeValue(b))
      }
    }

    ChiselStage.convert(new Foo)
  }

  "Layers error checking" should "require that a nested layer definition matches its declaration nesting" in {

    class Foo extends RawModule {
      layer.block(A.B) {
        val a = Wire(Bool())
      }
    }

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new Foo) }.getMessage() should include(
      "nested layer 'B' must be wrapped in parent layer 'A'"
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
      "Cannot define 'Foo.a: IO[Bool]' from colors {'C'} since at least one of these is NOT enabled when 'Foo.a: IO[Bool]' is enabled"
    )
  }

}
