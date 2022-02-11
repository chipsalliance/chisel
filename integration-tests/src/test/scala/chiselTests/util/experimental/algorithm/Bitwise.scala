// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.util._
import chisel3.experimental.util.algorithm._
import chiseltest._
import chiseltest.formal._
import org.scalatest.flatspec.AnyFlatSpec

class LSBOrTestModule(width: Int) extends Module {
  val input = IO(Input(UInt(width.W)))

  var lsb = false.B
  val vec = for(b <- input.asBools) yield {
    val cur = b || lsb
    lsb = cur
    cur
  }
  val ref = VecInit(vec).asUInt

  val testee = LSBOr(input)

  assert(testee === ref)
}

class MSBOrTestModule(width: Int) extends Module {
  val input = IO(Input(UInt(width.W)))

  val ref = Reverse(LSBOr(Reverse(input)))
  val testee = MSBOr(input)

  assert(testee === ref)
}

class LSBMSBOrTest extends AnyFlatSpec with ChiselScalatestTester with Formal {
  "LSBOr" should "correctly computes" in {
    for(i <- 1 to 16) {
      verify(new LSBOrTestModule(i), Seq(BoundedCheck(1)))
    }
  }

  "MSBOr" should "correctly computes" in {
    for(i <- 1 to 16) {
      verify(new MSBOrTestModule(i), Seq(BoundedCheck(1)))
    }
  }
}