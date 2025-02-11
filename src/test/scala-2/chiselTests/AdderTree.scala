// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.BasicTester

class AdderTree[T <: Bits with Num[T]](genType: T, vecSize: Int) extends Module {
  val io = IO(new Bundle {
    val numIn = Input(Vec(vecSize, genType))
    val numOut = Output(genType)
  })
  io.numOut := io.numIn.reduceTree((a: T, b: T) => (a + b))
}

class AdderTreeTester(bitWidth: Int, numsToAdd: List[Int]) extends BasicTester {
  val genType = UInt(bitWidth.W)
  val dut = Module(new AdderTree(genType, numsToAdd.size))
  dut.io.numIn := VecInit(numsToAdd.map(x => x.asUInt(bitWidth.W)))
  val sumCorrect = dut.io.numOut === (numsToAdd.reduce(_ + _) % (1 << bitWidth)).asUInt(bitWidth.W)
  assert(sumCorrect)
  stop()
}

class AdderTreeSpec extends ChiselPropSpec {
  property("All numbers should be added correctly by an Adder Tree") {
    forAll(safeUIntN(20)) {
      case (w: Int, v: List[Int]) => {
        whenever(v.size > 0 && w > 0) {
          assertTesterPasses { new AdderTreeTester(w, v.map(x => math.abs(x) % (1 << w)).toList) }
        }
      }
    }
  }
}
