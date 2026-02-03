// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.reflect.DataMirror
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

// Avoid collisions with regular BlackBox tests by putting ExtModule blackboxes
// in their own scope.
package extmoduletests {

  class BlackBoxInverter extends ExtModule {
    val in = IO(Input(Bool()))
    val out = IO(Output(Bool()))

    addResource("/chisel3/BlackBoxInverter.v")
  }

  class BlackBoxPassthrough extends ExtModule {
    val in = IO(Input(Bool()))
    val out = IO(Output(Bool()))

    addResource("/chisel3/BlackBoxPassthrough.v")
  }
}

class ExtModuleTester extends Module {
  val blackBoxPos = Module(new extmoduletests.BlackBoxInverter)
  val blackBoxNeg = Module(new extmoduletests.BlackBoxInverter)

  blackBoxPos.in := 1.U
  blackBoxNeg.in := 0.U

  assert(blackBoxNeg.out === 1.U)
  assert(blackBoxPos.out === 0.U)
  stop()
}

/** Instantiate multiple BlackBoxes with similar interfaces but different
  * functionality. Used to detect failures in BlackBox naming and module
  * deduplication.
  */

class MultiExtModuleTester extends Module {
  val blackBoxInvPos = Module(new extmoduletests.BlackBoxInverter)
  val blackBoxInvNeg = Module(new extmoduletests.BlackBoxInverter)
  val blackBoxPassPos = Module(new extmoduletests.BlackBoxPassthrough)
  val blackBoxPassNeg = Module(new extmoduletests.BlackBoxPassthrough)

  blackBoxInvPos.in := 1.U
  blackBoxInvNeg.in := 0.U
  blackBoxPassPos.in := 1.U
  blackBoxPassNeg.in := 0.U

  assert(blackBoxInvNeg.out === 1.U)
  assert(blackBoxInvPos.out === 0.U)
  assert(blackBoxPassNeg.out === 0.U)
  assert(blackBoxPassPos.out === 1.U)
  stop()
}

class ExtModuleWithSuggestName extends ExtModule {
  val in = IO(Input(UInt(8.W)))
  in.suggestName("foo")
  val out = IO(Output(UInt(8.W)))
}

class ExtModuleWithSuggestNameTester extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))
  val inst = Module(new ExtModuleWithSuggestName)
  inst.in := in
  out := inst.out
}

class SimpleIOBundle extends Bundle {
  val in = Input(UInt(8.W))
  val out = Output(UInt(8.W))
}

class ExtModuleWithFlatIO extends ExtModule {
  val badIO = FlatIO(new SimpleIOBundle)
}

class ExtModuleWithFlatIOTester extends Module {
  val io = IO(new SimpleIOBundle)
  val inst = Module(new ExtModuleWithFlatIO)
  io <> inst.badIO
}

class ExtModuleInvalidatedTester extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))
  val inst = Module(new ExtModule {
    val in = IO(Input(UInt(8.W)))
    val out = IO(Output(UInt(8.W)))
  })
  inst.in := in
  out := inst.out
}

class ExtModuleSpec extends AnyFlatSpec with Matchers with ChiselSim with FileCheck {
  "A ExtModule inverter" should "work" in {
    simulate(new ExtModuleTester)(RunUntilFinished(3))
  }
  "Multiple ExtModules" should "work" in {
    simulate(new MultiExtModuleTester)(RunUntilFinished(3))
  }
  "DataMirror.modulePorts" should "work with ExtModule" in {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {})
      val m = Module(new extmoduletests.BlackBoxPassthrough)
      assert(DataMirror.modulePorts(m) == Seq("in" -> m.in, "out" -> m.out))
    })
  }

  behavior.of("ExtModule")

  it should "work with .suggestName (aka it should not require reflection for naming)" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new ExtModuleWithSuggestNameTester)
    chirrtl should include("input foo : UInt<8>")
    chirrtl should include("connect inst.foo, in")
  }

  it should "work with FlatIO" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new ExtModuleWithFlatIOTester)
    chirrtl should include("connect io.out, inst.out")
    chirrtl should include("connect inst.in, io.in")
    chirrtl shouldNot include("badIO")
  }

  it should "not have invalidated ports in a chisel3._ context" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new ExtModuleInvalidatedTester)
    chirrtl shouldNot include("invalidater inst.in")
    chirrtl shouldNot include("invalidate inst.out")
  }

  it should "sort the verilog output of their param map by param key" in {

    class ParameterizedBlackBox(m: Map[String, Param]) extends ExtModule(m) {
      val io = IO(new Bundle {
        val out = Output(Clock())
        val in = Input(Clock())
      })
    }

    class Top(m: Map[String, Param]) extends Module {
      val io = IO(new Bundle {})
      val pbb = Module(new ParameterizedBlackBox(m))
      pbb.io.in := clock
    }

    val sixteenParams = ('a' until 'p').map { key => key.toString -> IntParam(1) }

    def splitAndStrip(verilog: String): Array[String] = verilog.split("\n").map(_.dropWhile(_.isWhitespace))

    getVerilogString(new Top(Map())) should include("ParameterizedBlackBox pbb")
    getVerilogString(new Top(Map("a" -> IntParam(1)))) should include(".a(1)")

    // check that both param orders are the same
    (splitAndStrip(getVerilogString(new Top(Map("a" -> IntParam(1), "b" -> IntParam(1))))) should contain).allOf(
      ".a(1),",
      ".b(1)"
    )
    (splitAndStrip(getVerilogString(new Top(Map("b" -> IntParam(1), "a" -> IntParam(1))))) should contain).allOf(
      ".a(1),",
      ".b(1)"
    )

    // check that both param orders are the same, note that verilog output does a newline when more params are present
    (splitAndStrip(getVerilogString(new Top(sixteenParams.toMap))) should contain).allOf(
      ".a(1),",
      ".b(1),",
      ".c(1),",
      ".d(1),",
      ".e(1),",
      ".f(1),",
      ".g(1),",
      ".h(1),",
      ".i(1),",
      ".j(1),",
      ".k(1),",
      ".l(1),",
      ".m(1),",
      ".n(1),",
      ".o(1)"
    )
    (splitAndStrip(getVerilogString(new Top(sixteenParams.reverse.toMap))) should contain).allOf(
      ".a(1),",
      ".b(1),",
      ".c(1),",
      ".d(1),",
      ".e(1),",
      ".f(1),",
      ".g(1),",
      ".h(1),",
      ".i(1),",
      ".j(1),",
      ".k(1),",
      ".l(1),",
      ".m(1),",
      ".n(1),",
      ".o(1)"
    )
  }

  it should "emit FIRRTL knownlayer syntax if they have known layers" in {

    object A extends layer.Layer(layer.LayerConfig.Extract())

    // No known layers
    class Bar extends ExtModule(knownLayers = Seq.empty)
    // Single known layer, built-in
    class Baz extends ExtModule(knownLayers = Seq(layers.Verification))
    // Multiple known layers
    class Qux extends ExtModule(knownLayers = Seq(layers.Verification, layers.Verification.Assert))
    // Single known layer, user-defined and should be added to the circuit
    class Quz extends ExtModule(knownLayers = Seq(A))

    class Foo extends Module {
      private val bar = Module(new Bar)
      private val baz = Module(new Baz)
      private val qux = Module(new Qux)
      private val quz = Module(new Quz)
    }

    info("emitted CHIRRTL looks correct")
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK: layer A,
           |CHECK: extmodule Bar :
           |CHECK: extmodule Baz knownlayer Verification :
           |CHECK: extmodule Qux knownlayer Verification, Verification.Assert :
           |CHECK: extmodule Quz knownlayer A :
           |""".stripMargin
      )

    info("CIRCT compilation doesn't error")
    ChiselStage.emitSystemVerilog(new Foo)
  }

  it should "allow updates to knownLayers via adding layer-colored probe ports or via addLayer" in {

    class Bar extends ExtModule {
      val a = IO(Output(probe.Probe(Bool(), layers.Verification)))
    }

    object A extends layer.Layer(layer.LayerConfig.Extract())

    class Baz extends ExtModule(knownLayers = Seq(A)) {
      val a = IO(Output(probe.Probe(Bool(), layers.Verification)))
    }

    class Qux extends ExtModule {
      layer.addLayer(A)
    }

    class Foo extends Module {
      private val bar = Module(new Bar)
      private val baz = Module(new Baz)
      private val qux = Module(new Qux)
    }

    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK: layer Verification,
           |CHECK: extmodule Bar knownlayer Verification :
           |CHECK: extmodule Baz knownlayer A, Verification :
           |CHECK: extmodule Qux knownlayer A :
           |""".stripMargin
      )

  }

  it should "have source locator information on ports" in {
    class Bar extends ExtModule {
      val a = IO(Output(Bool()))
    }

    class Foo extends Module {
      val a = IO(Output(Bool()))

      private val bar = Module(new Bar)
      a :<= bar.a
    }

    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK: extmodule Bar :
           |CHECK:   output a : UInt<1> @[{{.+}}ExtModule.scala {{[0-9]+}}:{{[0-9]+}}]
           |""".stripMargin
      )
  }

  it should "emit require statements" in {
    class Bar extends ExtModule(requirements = Seq.empty)
    class Baz extends ExtModule(requirements = Seq("libdv"))
    class Qux extends ExtModule(requirements = Seq("libdv", "vcs>=202505"))

    class Foo extends Module {
      private val bar = Module(new Bar)
      private val baz = Module(new Baz)
      private val qux = Module(new Qux)
    }

    info("emitted CHIRRTL looks correct")
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK: extmodule Bar :
           |CHECK: extmodule Baz requires "libdv" :
           |CHECK: extmodule Qux requires "libdv", "vcs>=202505" :
           |""".stripMargin
      )
  }

  it should "emit both knownlayer and requires when both are specified" in {
    object A extends layer.Layer(layer.LayerConfig.Extract())

    class Bar extends ExtModule(knownLayers = Seq(A), requirements = Seq("libdv"))

    class Foo extends Module {
      private val bar = Module(new Bar)
    }

    info("emitted CHIRRTL looks correct")
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK: layer A,
           |CHECK: extmodule Bar knownlayer A requires "libdv" :
           |""".stripMargin
      )
  }

  it should "escape special characters in requirements" in {
    class Bar extends ExtModule(requirements = Seq("\"quoted\"", "back\\slash"))

    class Foo extends Module {
      private val bar = Module(new Bar)
    }

    info("emitted CHIRRTL looks correct")
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        """|CHECK: extmodule Bar requires "\"quoted\"", "back\\slash" :
           |""".stripMargin
      )
  }

}
