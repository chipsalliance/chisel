// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
<<<<<<< HEAD:integration-tests/src/test/scala/chiselTest/ShiftRegisterMemSpec.scala
import chisel3.testers.BasicTester
||||||| parent of 62bdfce5 ([test] Remove unnecessary usages of BasicTester):integration-tests/src/test/scala-2/chiselTest/ShiftRegisterMemSpec.scala
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testers.BasicTester
=======
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
>>>>>>> 62bdfce5 ([test] Remove unnecessary usages of BasicTester):integration-tests/src/test/scala-2/chiselTest/ShiftRegisterMemSpec.scala
import chisel3.util.{Counter, ShiftRegister}
import org.scalacheck.{Gen, Shrink}

class ShiftMemTester(n: Int, dp_mem: Boolean) extends Module {
  val (cntVal, done) = Counter(true.B, n)
  val start = 23.U
  val sr = ShiftRegister.mem(cntVal + start, n, true.B, dp_mem, Some("simple_sr"))
  when(RegNext(done)) {
    assert(sr === start)
    stop()
  }
}

class ShiftRegisterMemSpec extends ChiselPropSpec {

  implicit val nonNegIntShrinker: Shrink[Int] = Shrink.shrinkIntegral[Int].suchThat(_ >= 0)

  property("ShiftRegister with dual-port SRAM should shift") {
    forAll(Gen.choose(0, 4)) { (shift: Int) => assertTesterPasses { new ShiftMemTester(shift, true) } }
  }

  property("ShiftRegister with single-port SRAM should shift") {
    forAll(Gen.choose(0, 6).suchThat(_ % 2 == 0)) { (shift: Int) =>
      assertTesterPasses { new ShiftMemTester(shift, false) }
    }
  }
}
