// See LICENSE for license details.

package chiselTests

import java.io.File

import org.scalatest._
import chisel3._
import chisel3.experimental._
import chisel3.testers.BasicTester
import chisel3.util._
//import chisel3.core.ExplicitCompileOptions.Strict

class BlackBoxInverter extends BlackBox {
  val io = IO(new Bundle() {
    val in = Input(Bool())
    val out = Output(Bool())
  })
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

class BlackBoxConstant(value: Int) extends BlackBox(
    Map("VALUE" -> value, "WIDTH" -> log2Ceil(value + 1))) {
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

class BlackBoxWithParamsTester extends BasicTester {
  val blackBoxOne  = Module(new BlackBoxConstant(1))
  val blackBoxFour  = Module(new BlackBoxConstant(4))
  val blackBoxStringParamOne = Module(new BlackBoxStringParam("one"))
  val blackBoxStringParamTwo = Module(new BlackBoxStringParam("two"))
  val blackBoxRealParamOne = Module(new BlackBoxRealParam(1.0))
  val blackBoxRealParamNeg = Module(new BlackBoxRealParam(-1.0))
  val blackBoxTypeParamBit = Module(new BlackBoxTypeParam(1, "bit"))
  val blackBoxTypeParamWord = Module(new BlackBoxTypeParam(32, "bit [31:0]"))

  val (cycles, end) = Counter(true.B, 4)

  assert(blackBoxOne.io.out  === 1.U)
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
    assertTesterPasses({ new BlackBoxTester },
        Seq("/chisel3/BlackBoxTest.v"))
  }
  "A BlackBoxed with flipped IO" should "work" in {
    assertTesterPasses({ new BlackBoxFlipTester },
        Seq("/chisel3/BlackBoxTest.v"))
  }
  "Multiple BlackBoxes" should "work" in {
    assertTesterPasses({ new MultiBlackBoxTester },
        Seq("/chisel3/BlackBoxTest.v"))
  }
  "A BlackBoxed register" should "work" in {
    assertTesterPasses({ new BlackBoxWithClockTester },
        Seq("/chisel3/BlackBoxTest.v"))
  }
  "BlackBoxes with parameters" should "work" in {
    assertTesterPasses({ new BlackBoxWithParamsTester },
        Seq("/chisel3/BlackBoxTest.v"))
  }
}
