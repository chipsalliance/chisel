// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
<<<<<<< HEAD:src/test/scala/chiselTests/MulLookup.scala
import chisel3.testers.BasicTester
||||||| parent of 62bdfce5 ([test] Remove unnecessary usages of BasicTester):src/test/scala-2/chiselTests/MulLookup.scala
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testers.BasicTester
import org.scalatest.propspec.AnyPropSpec
=======
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import org.scalatest.propspec.AnyPropSpec
>>>>>>> 62bdfce5 ([test] Remove unnecessary usages of BasicTester):src/test/scala-2/chiselTests/MulLookup.scala

class MulLookup(val w: Int) extends Module {
  val io = IO(new Bundle {
    val x = Input(UInt(w.W))
    val y = Input(UInt(w.W))
    val z = Output(UInt((2 * w).W))
  })
  val tbl = VecInit(
    for {
      i <- 0 until 1 << w
      j <- 0 until 1 << w
    } yield (i * j).asUInt((2 * w).W)
  )
  io.z := tbl(((io.x << w) | io.y))
}

class MulLookupTester(w: Int, x: Int, y: Int) extends Module {
  val dut = Module(new MulLookup(w))
  dut.io.x := x.asUInt
  dut.io.y := y.asUInt
  assert(dut.io.z === (x * y).asUInt)
  stop()
}

class MulLookupSpec extends ChiselPropSpec {

  property("Mul lookup table should return the correct result") {
    forAll(smallPosInts, smallPosInts) { (x: Int, y: Int) =>
      assertTesterPasses { new MulLookupTester(3, x, y) }
    }
  }
}
