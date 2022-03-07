// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.formal._
import org.scalatest.flatspec.AnyFlatSpec
import scala.math.min

class ScanLeftOrTestModule(width: Int) extends Module {
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

class ScanRightOrTestModule(width: Int) extends Module {
  val input = IO(Input(UInt(width.W)))

  val ref = Reverse(scanLeftOr(Reverse(input)))
  val testee = scanRightOr(input)

  assert(testee === ref)
}

class scanOrTest extends AnyFlatSpec with ChiselScalatestTester with Formal {
  "scanLeftOr" should "compute correctly" in {
    for(i <- 1 to 16) {
      verify(new ScanLeftOrTestModule(i), Seq(BoundedCheck(1)))
    }
  }

  "scanRightOr" should "compute correctly" in {
    for(i <- 1 to 16) {
      verify(new ScanRightOrTestModule(i), Seq(BoundedCheck(1)))
    }
  }
}
