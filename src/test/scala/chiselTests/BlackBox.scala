// See LICENSE for license details.

package chiselTests

import java.io.File
import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

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

  blackBoxPos.io.in := 1.asUInt
  blackBoxNeg.io.in := 0.asUInt

  assert(blackBoxNeg.io.out === 1.asUInt)
  assert(blackBoxPos.io.out === 0.asUInt)
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

  blackBoxInvPos.io.in := 1.asUInt
  blackBoxInvNeg.io.in := 0.asUInt
  blackBoxPassPos.io.in := 1.asUInt
  blackBoxPassNeg.io.in := 0.asUInt

  assert(blackBoxInvNeg.io.out === 1.asUInt)
  assert(blackBoxInvPos.io.out === 0.asUInt)
  assert(blackBoxPassNeg.io.out === 0.asUInt)
  assert(blackBoxPassPos.io.out === 1.asUInt)
  stop()
}

class BlackBoxWithClockTester extends BasicTester {
  val blackBox = Module(new BlackBoxRegister)
  val model = Reg(Bool())

  val (cycles, end) = Counter(true.asBool, 15)

  val impetus = cycles(0)
  blackBox.io.clock := clock
  blackBox.io.in := impetus
  model := impetus

  when(cycles > 0.asUInt) {
    assert(blackBox.io.out === model)
  }
  when(end) { stop() }
}

/*
// Must determine how to handle parameterized Verilog
class BlackBoxConstant(value: Int) extends BlackBox {
  val io = IO(new Bundle() {
    val out = Output(UInt(width=log2Up(value)))
  })
  override val name = s"#(WIDTH=${log2Up(value)},VALUE=$value) "
}

class BlackBoxWithParamsTester extends BasicTester {
  val blackBoxOne  = Module(new BlackBoxConstant(1))
  val blackBoxFour = Module(new BlackBoxConstant(4))

  val (cycles, end) = Counter(true.asBool, 4)

  assert(blackBoxOne.io.out  === 1.asUInt)
  assert(blackBoxFour.io.out === 4.asUInt)

  when(end) { stop() }
}
*/

class BlackBoxSpec extends ChiselFlatSpec {
  "A BlackBoxed inverter" should "work" in {
    assertTesterPasses({ new BlackBoxTester },
        Seq("/BlackBoxTest.v"))
  }
  "Multiple BlackBoxes" should "work" in {
    assertTesterPasses({ new MultiBlackBoxTester },
        Seq("/BlackBoxTest.v"))
  }
  "A BlackBoxed register" should "work" in {
    assertTesterPasses({ new BlackBoxWithClockTester },
        Seq("/BlackBoxTest.v"))
  }
}
