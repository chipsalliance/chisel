// See LICENSE for license details.

package chiselTests

import java.io.File
import org.scalatest._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._

class BlackBoxInverter extends BlackBox {
  val io = new Bundle() {
    val in = Bool(INPUT)
    val out = Bool(OUTPUT)
  }
}

class BlackBoxPassthrough extends BlackBox {
  val io = new Bundle() {
    val in = Bool(INPUT)
    val out = Bool(OUTPUT)
  }
}

class BlackBoxRegister extends BlackBox {
  val io = new Bundle() {
    val clock = Clock().asInput
    val in = Bool(INPUT)
    val out = Bool(OUTPUT)
  }
}

class BlackBoxTester extends BasicTester {
  val blackBoxPos = Module(new BlackBoxInverter)
  val blackBoxNeg = Module(new BlackBoxInverter)

  blackBoxPos.io.in := UInt(1)
  blackBoxNeg.io.in := UInt(0)

  assert(blackBoxNeg.io.out === UInt(1))
  assert(blackBoxPos.io.out === UInt(0))
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

  blackBoxInvPos.io.in := UInt(1)
  blackBoxInvNeg.io.in := UInt(0)
  blackBoxPassPos.io.in := UInt(1)
  blackBoxPassNeg.io.in := UInt(0)

  assert(blackBoxInvNeg.io.out === UInt(1))
  assert(blackBoxInvPos.io.out === UInt(0))
  assert(blackBoxPassNeg.io.out === UInt(0))
  assert(blackBoxPassPos.io.out === UInt(1))
  stop()
}

class BlackBoxWithClockTester extends BasicTester {
  val blackBox = Module(new BlackBoxRegister)
  val model = Reg(Bool())

  val (cycles, end) = Counter(Bool(true), 15)

  val impetus = cycles(0)
  blackBox.io.clock := clock
  blackBox.io.in := impetus
  model := impetus

  when(cycles > UInt(0)) {
    assert(blackBox.io.out === model)
  }
  when(end) { stop() }
}

/*
// Must determine how to handle parameterized Verilog
class BlackBoxConstant(value: Int) extends BlackBox {
  val io = new Bundle() {
    val out = UInt(width=log2Up(value)).asOutput
  }
  override val name = s"#(WIDTH=${log2Up(value)},VALUE=$value) "
}

class BlackBoxWithParamsTester extends BasicTester {
  val blackBoxOne  = Module(new BlackBoxConstant(1))
  val blackBoxFour = Module(new BlackBoxConstant(4))

  val (cycles, end) = Counter(Bool(true), 4)

  assert(blackBoxOne.io.out  === UInt(1))
  assert(blackBoxFour.io.out === UInt(4))

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
