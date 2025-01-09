// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.PopCount
import chisel3.testers.BasicTester

class PopCountTester(n: Int) extends BasicTester {
  val x = RegInit(0.U(n.W))
  x := x + 1.U
  when(RegNext(x === ~0.U(n.W))) { stop() }

  val result = PopCount(x.asBools)
  val expected = x.asBools.foldLeft(0.U)(_ +& _)
  assert(result === expected)

  require(result.getWidth == BigInt(n).bitLength)
}

class PopCountSpec extends ChiselPropSpec {
  property("Mul lookup table should return the correct result") {
    forAll(smallPosInts) { (n: Int) => assertTesterPasses { new PopCountTester(n) } }
  }
}
