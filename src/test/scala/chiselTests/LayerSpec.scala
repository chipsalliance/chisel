// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.hierarchy.core.{Definition, Instance}
import chisel3.experimental.hierarchy.instantiable
import chisel3.probe.{define, Probe, ProbeValue}
import chiselTests.{ChiselFlatSpec, MatchesAndOmits, Utils}
import java.nio.file.{FileSystems, Paths}
import _root_.circt.stage.ChiselStage

class LayerSpec extends ChiselFlatSpec with Utils with MatchesAndOmits {

  val sep: String = FileSystems.getDefault().getSeparator()

  def bindLayer(name: String, dirs: List[String]): String = {
    val dirsStr = if (dirs.nonEmpty) s""", "${dirs.mkString(sep)}"""" else ""
    s"layer $name, bind$dirsStr :"
  }

  def inlineLayer(name: String): String = {
    s"layer $name, inline :"
  }

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

    val chirrtl = ChiselStage.emitCHIRRTL(new Foo, Array("--full-stacktrace"))

    info("CHIRRTL emission looks correct")
    // TODO: Switch to FileCheck for this testing.  This is going to miss all sorts of ordering issues.
    matchesAndOmits(chirrtl)(
      "layer A, bind, \"A\" :",
      "layer B, bind, \"A" ++ sep ++ "B\" :",
      "layerblock A :",
      "wire w : UInt<1>",
      "layerblock B :",
      "wire x : UInt<1>",
      "wire y : UInt<1>"
    )()
  }

  they should "create parent layer blocks automatically" in {

    class Foo extends RawModule {
      layer.block(A.B) {}
      layer.block(C) {
        layer.block(C) {}
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "layerblock A :",
      "layerblock B :",
      "layerblock C :"
    )()
  }

  they should "respect the 'skipIfAlreadyInBlock' parameter" in {
    class Foo extends RawModule {
      layer.block(A, skipIfAlreadyInBlock = true) {
        // This will fail to compile if `skipIfAlreadyInBlock=false`.
        layer.block(C, skipIfAlreadyInBlock = true) {}
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "layerblock A"
    )("layerblock C")
  }

  they should "respect the 'skipIfLayersEnabled' parameter" in {
    class Foo extends RawModule {
      layer.enable(A)
      layer.block(A.B, skipIfLayersEnabled = true) {}
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))()("layerblock")
  }

  they should "create no layer blocks when wrapped in 'elideBlocks'" in {
    class Foo extends RawModule {
      layer.elideBlocks {
        layer.block(A.B) {}
      }
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))()("layerblock")
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
    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      // Whitespace shows nested under another block.
      "      layerblock B : "
    )()
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

  they should "work correctly with Definition/Instance" in {

    @instantiable
    class Bar extends Module {
      layer.block(A) {}
    }

    class Foo extends Module {
      private val bar = Instance(Definition(new Bar))
    }

    val chirrtl = circt.stage.ChiselStage.emitCHIRRTL(new Foo)
    matchesAndOmits(chirrtl)("layer A")()
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

    val text = ChiselStage.emitCHIRRTL(new Foo)
    matchesAndOmits(text)(
      bindLayer("LayerWithDefaultOutputDir", List("LayerWithDefaultOutputDir")),
      bindLayer("SublayerWithDefaultOutputDir", List("LayerWithDefaultOutputDir", "SublayerWithDefaultOutputDir")),
      bindLayer("SublayerWithCustomOutputDir", List("myOtherOutputDir")),
      bindLayer("SublayerWithNoOutputDir", List()),
      bindLayer("LayerWithCustomOutputDir", List("myOutputDir")),
      bindLayer("SublayerWithDefaultOutputDir", List("myOutputDir", "SublayerWithDefaultOutputDir")),
      bindLayer("SublayerWithCustomOutputDir", List("myOtherOutputDir")),
      bindLayer("SublayerWithNoOutputDir", List()),
      bindLayer("LayerWithNoOutputDir", List()),
      bindLayer("SublayerWithDefaultOutputDir", List("SublayerWithDefaultOutputDir")),
      bindLayer("SublayerWithCustomOutputDir", List("myOtherOutputDir")),
      bindLayer("SublayerWithNoOutputDir", List())
    )()
  }

  they should "allow manually overriding the parent layer" in {

    implicit val Parent = layers.Verification
    object ExpensiveAsserts extends layer.Layer(layer.LayerConfig.Extract())

    class Foo extends RawModule {
      layer.addLayer(ExpensiveAsserts)
    }

    ChiselStage.emitCHIRRTL(new Foo) should include("""layer ExpensiveAsserts, bind, "Verification/ExpensiveAsserts"""")
  }

  "addLayer API" should "add a layer to the output CHIRRTL even if no layer block references that layer" in {
    class Foo extends RawModule {
      layer.addLayer(A)
    }

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))("layer A")("layer block")
  }

  "Default Layers" should "always be emitted in CHIRRTL (whereas non-default layers are optionally emitted)" in {
    class Foo extends RawModule {}

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "layer Verification",
      "layer Assert",
      "layer Assume",
      "layer Cover"
    )(
      "layer B"
    )
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
      "Cannot define 'Foo.a: IO[Bool]' from colors {'C'} since at least one of these is NOT enabled when 'Foo.a: IO[Bool]' is enabled"
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
}
