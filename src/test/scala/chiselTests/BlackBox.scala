// See LICENSE for license details.

package chiselTests

import java.io.File
import org.scalatest._
import Chisel._
import Chisel.testers.BasicTester

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

class BlackBoxSpec extends ChiselFlatSpec {
  "A BlackBoxed inverter" should "work" in {
    assertTesterPasses({ new BlackBoxTester },
        Seq("/BlackBoxTest.v"))
  }
  "Multiple BlackBoxes" should "work" in {
    assertTesterPasses({ new MultiBlackBoxTester },
        Seq("/BlackBoxTest.v"))
  }
}
