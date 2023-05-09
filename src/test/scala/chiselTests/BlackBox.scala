// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.experimental._
import chisel3.reflect.DataMirror
import chisel3.testers.{BasicTester, TesterDriver}
import chisel3.util._

class BlackBoxInverter extends BlackBox {
  val io = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  })
}

// Due to the removal of "val io", this technically works
// This style is discouraged, please use "val io"
class BlackBoxInverterSuggestName extends BlackBox {
  override def desiredName: String = "BlackBoxInverter"
  val foo = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  }).suggestName("io")
}

class BlackBoxPassthrough extends BlackBox {
  val io = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  })
}

// Test Flip on top-level IO
class BlackBoxPassthrough2 extends BlackBox {
  val io = IO(Flipped(new Bundle() {
    val in = Output(Bool())
    val out = Input(Bool())
  }))
}

class BlackBoxRegister extends BlackBox {
  val io = IO(new Bundle() {
    val clock = Input(Clock())
    val in = Input(Bool())
    val out = Output(Bool())
  })
}

class BlackBoxTester extends BasicTester {
  val blackBoxPos = Module(new BlackBoxInverter)
  val blackBoxNeg = Module(new BlackBoxInverter)

  blackBoxPos.io.in := 1.U
  blackBoxNeg.io.in := 0.U

  assert(blackBoxNeg.io.out === 1.U)
  assert(blackBoxPos.io.out === 0.U)
  stop()
}

class BlackBoxTesterSuggestName extends BasicTester {
  val blackBoxPos = Module(new BlackBoxInverterSuggestName)
  val blackBoxNeg = Module(new BlackBoxInverterSuggestName)

  blackBoxPos.foo.in := 1.U
  blackBoxNeg.foo.in := 0.U

  assert(blackBoxNeg.foo.out === 1.U)
  assert(blackBoxPos.foo.out === 0.U)
  stop()
}

class BlackBoxFlipTester extends BasicTester {
  val blackBox = Module(new BlackBoxPassthrough2)

  blackBox.io.in := 1.U
  assert(blackBox.io.out === 1.U)
  stop()
}

/** Instantiate multiple BlackBoxes with similar interfaces but different
  * functionality. Used to detect failures in BlackBox naming and module
  * deduplication.
  */

class MultiBlackBoxTester extends BasicTester {
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

class BlackBoxWithClockTester extends BasicTester {
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

class BlackBoxConstant(value: Int) extends BlackBox(Map("VALUE" -> value, "WIDTH" -> log2Ceil(value + 1))) {
  require(value >= 0, "value must be a UInt!")
  val io = IO(new Bundle {
    val out = Output(UInt(log2Ceil(value + 1).W))
  })
}

class BlackBoxStringParam(str: String) extends BlackBox(Map("STRING" -> str)) {
  val io = IO(new Bundle {
    val out = UInt(32.W)
  })
}

class BlackBoxRealParam(dbl: Double) extends BlackBox(Map("REAL" -> dbl)) {
  val io = IO(new Bundle {
    val out = UInt(64.W)
  })
}

class BlackBoxTypeParam(w: Int, raw: String) extends BlackBox(Map("T" -> RawParam(raw))) {
  val io = IO(new Bundle {
    val out = UInt(w.W)
  })
}

class BlackBoxNoIO extends BlackBox {
  // Whoops! typo
  val ioo = IO(new Bundle {
    val out = Output(UInt(8.W))
  })
}

class BlackBoxUIntIO extends BlackBox {
  val io = IO(Output(UInt(8.W)))
}

class SimplerBlackBoxWithParamsTester extends BasicTester {
  val blackBoxTypeParamBit = Module(new BlackBoxTypeParam(1, "bit"))
  val blackBoxTypeParamWord = Module(new BlackBoxTypeParam(32, "bit [31:0]"))

  val (cycles, end) = Counter(true.B, 4)

  assert(blackBoxTypeParamBit.io.out === 1.U)
  assert(blackBoxTypeParamWord.io.out === "hdeadbeef".U(32.W))

  when(end) { stop() }
}

class BlackBoxWithParamsTester extends BasicTester {
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

class BlackBoxSpec extends ChiselFlatSpec {
  "A BlackBoxed inverter" should "work" in {
    assertTesterPasses({ new BlackBoxTester }, Seq("/chisel3/BlackBoxTest.v"), TesterDriver.verilatorOnly)
  }
  "A BlackBoxed with flipped IO" should "work" in {
    assertTesterPasses({ new BlackBoxFlipTester }, Seq("/chisel3/BlackBoxTest.v"), TesterDriver.verilatorOnly)
  }
  "Multiple BlackBoxes" should "work" in {
    assertTesterPasses({ new MultiBlackBoxTester }, Seq("/chisel3/BlackBoxTest.v"), TesterDriver.verilatorOnly)
  }
  "A BlackBoxed register" should "work" in {
    assertTesterPasses({ new BlackBoxWithClockTester }, Seq("/chisel3/BlackBoxTest.v"), TesterDriver.verilatorOnly)
  }
  //TODO: SFC->MFC, this test is ignored because the parameters have undesired quotes around values in verilog in MFC
  "BlackBoxes with simpler parameters" should "work" ignore {
    assertTesterPasses(
      { new SimplerBlackBoxWithParamsTester },
      Seq("/chisel3/BlackBoxTest.v"),
      TesterDriver.verilatorOnly
    )
  }
  //TODO: SFC->MFC, this test is ignored because the parameters have undesired quotes around values in verilog in MFC
  "BlackBoxes with parameters" should "work" ignore {
    assertTesterPasses({ new BlackBoxWithParamsTester }, Seq("/chisel3/BlackBoxTest.v"), TesterDriver.verilatorOnly)
  }
  "DataMirror.modulePorts" should "work with BlackBox" in {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {})
      val m = Module(new BlackBoxPassthrough)
      assert(DataMirror.modulePorts(m) == Seq("in" -> m.io.in, "out" -> m.io.out))
    })
  }
  "A BlackBox using suggestName(\"io\")" should "work (but don't do this)" in {
    assertTesterPasses({ new BlackBoxTesterSuggestName }, Seq("/chisel3/BlackBoxTest.v"), TesterDriver.verilatorOnly)
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
}
