// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.PopCount
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers

class PopCountTester(n: Int) extends Module {
  val x = RegInit(0.U(n.W))
  x := x + 1.U
  when(RegNext(x === ~0.U(n.W))) { stop() }

  val result = PopCount(x.asBools)
  val expected = x.asBools.foldLeft(0.U)(_ +& _)
  assert(result === expected)

  for (c <- 0 to n + 2) {
    assert(PopCount.equalTo    (c, x) === (PopCount(x)===c.U), s"Wrong result for PopCount.equalTo    ($c,x) function, x.width=$n")
    assert(PopCount.greaterThan(c, x) === (PopCount(x)>  c.U), s"Wrong result for PopCount.greaterThan($c,x) function, x.width=$n")
    assert(PopCount.atLeast    (c, x) === (PopCount(x)>= c.U), s"Wrong result for PopCount.atLeast    ($c,x) function, x.width=$n")
  }
  require(result.getWidth == BigInt(n).bitLength)
}

class PopCountSpec extends AnyPropSpec with PropertyUtils with ChiselSim {
  property("PopCount circuitry should return the correct result") {
    forAll(smallPosInts) { (n: Int) => simulate(new PopCountTester(n))(RunUntilFinished(math.pow(2, n).toInt + 2)) }
  }
}
