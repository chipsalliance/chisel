// See LICENSE for license details.

package chiselTests

import Chisel._
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class GoodBoolConversion extends Module {
  val io = new Bundle {
    val u = UInt(1, width = 1).asInput
    val b = Bool(OUTPUT)
  }
  io.b := io.u.toBool
}

class BadBoolConversion extends Module {
  val io = new Bundle {
    val u = UInt(1, width = 5).asInput
    val b = Bool(OUTPUT)
  }
  io.b := io.u.toBool
}

class UIntSpec extends ChiselPropSpec with Matchers {
  property("Bools can be created from 1 bit UInts") {
    elaborate(new GoodBoolConversion)
  }

  property("Bools cannot be created from >1 bit UInts") {
    a [Exception] should be thrownBy { elaborate(new BadBoolConversion) }
  }
}
