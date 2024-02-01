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

  it should "allow for defines in a layer to drive layer-colored probes" in {

    class Foo extends RawModule {
      val in = IO(Input(Bool()))
      val a = IO(Output(Probe(Bool(), A)))
      val b = IO(Output(Probe(Bool(), A.B)))
      layer.block(A) {
        define(a, ProbeValue(in))
        layer.block(A.B) {
          define(b, ProbeValue(in))
        }
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "define a = probe(in)",
      "define b = probe(in)"
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
      "module Foo enablelayer A.B enablelayer C :"
    )()

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

  it should "check that a define to a layer-colored probe must be inside a layerblock" in {

    class Foo extends RawModule {
      val a = IO(Output(Probe(Bool(), A)))
      val b = Wire(Bool())
      define(a, ProbeValue(b))
    }

    intercept[ChiselException] { ChiselStage.emitCHIRRTL(new Foo, Array("--throw-on-first-error")) }
      .getMessage() should include("Cannot define 'Foo.a: IO[Bool]' from outside a layerblock")

  }

  it should "check that a define from inside a layerblock is to a legal layer-colored probe" in {
    class Foo extends RawModule {
      val a = IO(Output(Probe(Bool(), A.B)))
      layer.block(A) {
        val b = Wire(Bool())
        define(a, ProbeValue(b))
      }
    }
    intercept[ChiselException] { ChiselStage.emitCHIRRTL(new Foo, Array("--throw-on-first-error")) }
      .getMessage() should include("Cannot define 'Foo.a: IO[Bool]' from a layerblock associated with layer A")
  }

}
