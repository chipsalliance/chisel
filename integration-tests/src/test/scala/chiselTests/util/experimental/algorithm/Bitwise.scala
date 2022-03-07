// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.util._
import chisel3.experimental.util.algorithm._
import chiseltest._
import chiseltest.formal._
import org.scalatest.flatspec.AnyFlatSpec
import scala.math.min

class scanLeftOrTestModule(width: Int) extends Module {
  val input = IO(Input(UInt(width.W)))

  var lsb = false.B
  val vec = for(b <- input.asBools) yield {
    val cur = b || lsb
    lsb = cur
    cur
  }
  val ref = VecInit(vec).asUInt

  val testee = scanLeftOr(input)

  assert(testee === ref)
}

class scanRightOrTestModule(width: Int) extends Module {
  val input = IO(Input(UInt(width.W)))

  val ref = Reverse(scanLeftOr(Reverse(input)))
  val testee = scanRightOr(input)

  assert(testee === ref)
}

class scanOrTest extends AnyFlatSpec with ChiselScalatestTester with Formal {
  "scanLeftOr" should "correctly computes" in {
    for(i <- 1 to 16) {
      verify(new scanLeftOrTestModule(i), Seq(BoundedCheck(1)))
    }
  }

  "scanRightOr" should "correctly computes" in {
    for(i <- 1 to 16) {
      verify(new scanRightOrTestModule(i), Seq(BoundedCheck(1)))
    }
  }
}