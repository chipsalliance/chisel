// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.experimental.{fromDoubleToDoubleParam, fromIntToIntParam, fromStringToStringParam}
import chisel3.reflect.DataMirror
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testing.scalatest.FileCheck
import chisel3.util._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class BlackBoxInverter extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  })

  addResource("/chisel3/BlackBoxInverter.v")
}

// Due to the removal of "val io", this technically works
// This style is discouraged, please use "val io"
class BlackBoxInverterSuggestName extends BlackBox with HasBlackBoxResource {
  override def desiredName: String = "BlackBoxInverter"
  val foo = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  }).suggestName("io")

  addResource("/chisel3/BlackBoxInverter.v")
}

class BlackBoxPassthrough extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  })

  addResource("/chisel3/BlackBoxPassthrough.v")
}

// Test Flip on top-level IO
class BlackBoxPassthrough2 extends BlackBox with HasBlackBoxResource {
  val io = IO(Flipped(new Bundle() {
    val in = Output(Bool())
    val out = Input(Bool())
  }))

  addResource("/chisel3/BlackBoxPassthrough2.v")
}

class BlackBoxRegister extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle() {
    val clock = Input(Clock())
    val in = Input(Bool())
    val out = Output(Bool())
  })

  addResource("/chisel3/BlackBoxRegister.v")
}

class BlackBoxTester extends Module {
  val blackBoxPos = Module(new BlackBoxInverter)
  val blackBoxNeg = Module(new BlackBoxInverter)

  blackBoxPos.io.in := 1.U
  blackBoxNeg.io.in := 0.U

  assert(blackBoxNeg.io.out === 1.U)
  assert(blackBoxPos.io.out === 0.U)
  stop()
}

class BlackBoxTesterSuggestName extends Module {
  val blackBoxPos = Module(new BlackBoxInverterSuggestName)
  val blackBoxNeg = Module(new BlackBoxInverterSuggestName)

  blackBoxPos.foo.in := 1.U
  blackBoxNeg.foo.in := 0.U

  assert(blackBoxNeg.foo.out === 1.U)
  assert(blackBoxPos.foo.out === 0.U)
  stop()
}

class BlackBoxFlipTester extends Module {
  val blackBox = Module(new BlackBoxPassthrough2)

  blackBox.io.in := 1.U
  assert(blackBox.io.out === 1.U)
  stop()
}

/** Instantiate multiple BlackBoxes with similar interfaces but different
  * functionality. Used to detect failures in BlackBox naming and module
  * deduplication.
  */

class MultiBlackBoxTester extends Module {
  val blackBoxInvPos = Module(new BlackBoxInverter)
  val blackBoxInvNeg = Module(new BlackBoxInverter)
  val blackBoxPassPos = Module(new BlackBoxPassthrough)
  val blackBoxPassNeg = Module(new BlackBoxPassthrough)

  blackBoxInvPos.io.in := 1.U
  blackBoxInvNeg.io.in := 0.U
  blackBoxPassPos.io.in := 1.U
  blackBoxPassNeg.io.in := 0.U

  assert(blackBoxInvNeg.io.out === 1.U)
  assert(blackBoxInvPos.io.out === 0.U)
  assert(blackBoxPassNeg.io.out === 0.U)
  assert(blackBoxPassPos.io.out === 1.U)
  stop()
}

class BlackBoxWithClockTester extends Module {
  val blackBox = Module(new BlackBoxRegister)
  val model = Reg(Bool())

  val (cycles, end) = Counter(true.B, 15)

  val impetus = cycles(0)
  blackBox.io.clock := clock
  blackBox.io.in := impetus
  model := impetus

  when(cycles > 0.U) {
    assert(blackBox.io.out === model)
  }
  when(end) { stop() }
}

class BlackBoxConstant(value: Int)
    extends BlackBox(Map("VALUE" -> value, "WIDTH" -> log2Ceil(value + 1)))
    with HasBlackBoxResource {
  require(value >= 0, "value must be a UInt!")
  val io = IO(new Bundle {
    val out = Output(UInt(log2Ceil(value + 1).W))
  })

  addResource("/chisel3/BlackBoxConstant.v")
}

class BlackBoxStringParam(str: String) extends BlackBox(Map("STRING" -> str)) with HasBlackBoxResource {
  val io = IO(new Bundle {
    val out = UInt(32.W)
  })

  addResource("/chisel3/BlackBoxStringParam.v")
}

class BlackBoxRealParam(dbl: Double) extends BlackBox(Map("REAL" -> dbl)) with HasBlackBoxResource {
  val io = IO(new Bundle {
    val out = UInt(64.W)
  })

  addResource("/chisel3/BlackBoxRealParam.v")
}

class BlackBoxTypeParam(w: Int, raw: String) extends BlackBox(Map("T" -> RawParam(raw))) with HasBlackBoxResource {
  val io = IO(new Bundle {
    val out = UInt(w.W)
  })

  addResource("/chisel3/BlackBoxTypeParam.v")
}

class BlackBoxNoIO extends BlackBox with HasBlackBoxResource {
  // Whoops! typo
  val ioo = IO(new Bundle {
    val out = Output(UInt(8.W))
  })

  addResource("/chisel3/BlackBoxNoIO.v")
}

class BlackBoxUIntIO extends BlackBox with HasBlackBoxResource {
  val io = IO(Output(UInt(8.W)))

  addResource("/chisel3/BlackBoxUIntIO.v")
}

class SimplerBlackBoxWithParamsTester extends Module {
  val blackBoxTypeParamBit = Module(new BlackBoxTypeParam(1, "bit"))
  val blackBoxTypeParamWord = Module(new BlackBoxTypeParam(32, "bit [31:0]"))

  val (cycles, end) = Counter(true.B, 4)

  assert(blackBoxTypeParamBit.io.out === 1.U)
  assert(blackBoxTypeParamWord.io.out === "hdeadbeef".U(32.W))

  when(end) { stop() }
}

class BlackBoxWithParamsTester extends Module {
  val blackBoxOne = Module(new BlackBoxConstant(1))
  val blackBoxFour = Module(new BlackBoxConstant(4))
  val blackBoxStringParamOne = Module(new BlackBoxStringParam("one"))
  val blackBoxStringParamTwo = Module(new BlackBoxStringParam("two"))
  val blackBoxRealParamOne = Module(new BlackBoxRealParam(1.0))
  val blackBoxRealParamNeg = Module(new BlackBoxRealParam(-1.0))
  val blackBoxTypeParamBit = Module(new BlackBoxTypeParam(1, "bit"))
  val blackBoxTypeParamWord = Module(new BlackBoxTypeParam(32, "bit [31:0]"))

  val (cycles, end) = Counter(true.B, 4)

  assert(blackBoxOne.io.out === 1.U)
  assert(blackBoxFour.io.out === 4.U)
  assert(blackBoxStringParamOne.io.out === 1.U)
  assert(blackBoxStringParamTwo.io.out === 2.U)
  assert(blackBoxRealParamOne.io.out === 0x3ff0000000000000L.U)
  assert(blackBoxRealParamNeg.io.out === BigInt("bff0000000000000", 16).U)
  assert(blackBoxTypeParamBit.io.out === 1.U)
  assert(blackBoxTypeParamWord.io.out === "hdeadbeef".U(32.W))

  when(end) { stop() }
}

class BlackBoxSpec extends AnyFlatSpec with Matchers with ChiselSim with FileCheck {
  "A BlackBoxed inverter" should "work" in {
    simulate(new BlackBoxTester)(RunUntilFinished(5))
  }
  "A BlackBoxed with flipped IO" should "work" in {
    simulate(new BlackBoxFlipTester)(RunUntilFinished(5))
  }
  "Multiple BlackBoxes" should "work" in {
    simulate(new MultiBlackBoxTester)(RunUntilFinished(5))
  }
  "A BlackBoxed register" should "work" in {
    simulate(new BlackBoxWithClockTester)(RunUntilFinished(16))
  }
  "BlackBoxes with simpler parameters" should "work" in {
    simulate(new SimplerBlackBoxWithParamsTester)(RunUntilFinished(5))
  }
  "BlackBoxes with parameters" should "work" in {
    simulate(new BlackBoxWithParamsTester)(RunUntilFinished(5))
  }
  "DataMirror.modulePorts" should "work with BlackBox" in {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {})
      val m = Module(new BlackBoxPassthrough)
      assert(DataMirror.modulePorts(m) == Seq("in" -> m.io.in, "out" -> m.io.out))
    })
  }
  "A BlackBox using suggestName(\"io\")" should "work (but don't do this)" in {
    simulate(new BlackBoxTesterSuggestName)(RunUntilFinished(5))
  }

  "A Blackbox with Flipped IO" should "work" in {
    class Top extends RawModule {
      val inst = Module(new BlackBox {
        override def desiredName: String = "MyBB"
        val io = IO(Flipped(new Bundle {
          val in = Bool()
          val out = Flipped(Bool())
        }))
      })
    }
    ChiselStage
      .emitCHIRRTL(new Top)
      .fileCheck()(
        """|CHECK:      module MyBB :
           |CHECK-NEXT:   input in : UInt<1>
           |CHECK-NEXT:   output out : UInt<1>
           |""".stripMargin
      )
  }

  "A BlackBox with no 'val io'" should "give a reasonable error message" in {
    (the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new Module {
        val inst = Module(new BlackBoxNoIO)
      })
    }).getMessage should include("must have a port named 'io' of type Record")
  }

  "A BlackBox with non-Record 'val io'" should "give a reasonable error message" in {
    (the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new Module {
        val inst = Module(new BlackBoxUIntIO)
      })
    }).getMessage should include("must have a port named 'io' of type Record")
  }

  "BlackBoxes" should "sort the verilog output of their param map by param key" in {

    class ParameterizedBlackBox(m: Map[String, Param]) extends BlackBox(m) {
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

  they should "emit FIRRTL knownlayer syntax if they have known layers" in {

    object A extends layer.Layer(layer.LayerConfig.Extract())

    sealed trait NoIo { this: BlackBox =>
      final val io = IO(new Bundle {})
    }

    // No known layers
    class Bar extends BlackBox(knownLayers = Seq.empty) with NoIo
    // Single known layer, built-in
    class Baz extends BlackBox(knownLayers = Seq(layers.Verification)) with NoIo
    // Multiple known layers
    class Qux extends BlackBox(knownLayers = Seq(layers.Verification, layers.Verification.Assert)) with NoIo
    // Single known layer, user-defined and should be added to the circuit
    class Quz extends BlackBox(knownLayers = Seq(A)) with NoIo

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

  they should "allow updates to knownLayers via adding layer-colored probe ports or via addLayer" in {

    class Bar extends BlackBox {
      final val io = IO {
        new Bundle {
          val a = Output(probe.Probe(Bool(), layers.Verification))
        }
      }
    }

    object A extends layer.Layer(layer.LayerConfig.Extract())

    class Baz extends BlackBox(knownLayers = Seq(A)) {
      final val io = IO {
        new Bundle {
          val a = Output(probe.Probe(Bool(), layers.Verification))
        }
      }
    }

    class Qux extends BlackBox {
      final val io = IO(new Bundle {})
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

}
