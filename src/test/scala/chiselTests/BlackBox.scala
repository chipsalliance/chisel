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

class BlackBoxTester extends BasicTester {
  val blackBoxPos = Module(new BlackBoxInverter)
  val blackBoxNeg = Module(new BlackBoxInverter)

  blackBoxPos.io.in := UInt(1)
  blackBoxNeg.io.in := UInt(0)

  assert(blackBoxNeg.io.out === UInt(1))
  assert(blackBoxPos.io.out === UInt(0))
  stop()
}

class BlackBoxSpec extends ChiselFlatSpec {
  "A BlackBoxed inverter" should "work" in {
    assertTesterPasses({ new BlackBoxTester },
        Seq("/BlackBoxInverter.v"))
  }
}
