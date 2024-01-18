// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.probe.{Probe, ProbeValue, define}
import chiselTests.{ChiselFlatSpec, MatchesAndOmits, Utils}
import _root_.circt.stage.ChiselStage

class LayerSpec extends ChiselFlatSpec with Utils with MatchesAndOmits {

    object A extends layer.Layer(layer.Convention.Bind) {
      object B extends layer.Layer(layer.Convention.Bind)
    }

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
      "declgroup A, bind :",
      "declgroup B, bind :",
      "group A :",
      "wire w : UInt<1>",
      "group B :",
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

}
