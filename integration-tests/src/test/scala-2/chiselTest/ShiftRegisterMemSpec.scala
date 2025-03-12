// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.{Counter, ShiftRegister}
import org.scalacheck.{Gen, Shrink}
import org.scalatest.propspec.AnyPropSpec

class ShiftMemTester(n: Int, dp_mem: Boolean) extends Module {
  val (cntVal, done) = Counter(true.B, n)
  val start = 23.U
  val sr = ShiftRegister.mem(cntVal + start, n, true.B, dp_mem, Some("simple_sr"))
  when(RegNext(done)) {
    assert(sr === start)
    stop()
  }
}

class ShiftRegisterMemSpec extends AnyPropSpec with PropertyUtils with ChiselSim {

  implicit val nonNegIntShrinker: Shrink[Int] = Shrink.shrinkIntegral[Int].suchThat(_ >= 0)

  property("ShiftRegister with dual-port SRAM should shift") {
    forAll(Gen.choose(0, 4)) { (shift: Int) =>
      simulate { new ShiftMemTester(shift, true) }(RunUntilFinished(shift + 2))
    }
  }

  property("ShiftRegister with single-port SRAM should shift") {
    forAll(Gen.choose(0, 6).suchThat(_ % 2 == 0)) { (shift: Int) =>
      simulate { new ShiftMemTester(shift, false) }(RunUntilFinished(shift + 2))
    }
  }
}
