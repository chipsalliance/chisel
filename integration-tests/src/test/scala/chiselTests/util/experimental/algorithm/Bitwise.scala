// SPDX-License-Identifier: Apache-2.0

import chisel3._
import chisel3.util._
import chisel3.experimental.util.algorithm._
import chiseltest._
import chiseltest.formal._
import org.scalatest.flatspec.AnyFlatSpec
import scala.math.min

// Copied from rocket-core
object RocketImpl {
  // Fill 1s from low bits to high bits
  def leftOR(x: UInt): UInt = leftOR(x, x.getWidth, x.getWidth)
  def leftOR(x: UInt, width: Integer, cap: Integer = 999999): UInt = {
    val stop = min(width, cap)
    def helper(s: Int, x: UInt): UInt =
      if (s >= stop) x else helper(s+s, x | (x << s)(width-1,0))
    helper(1, x)(width-1, 0)
  }

  // Fill 1s form high bits to low bits
  def rightOR(x: UInt): UInt = rightOR(x, x.getWidth, x.getWidth)
  def rightOR(x: UInt, width: Integer, cap: Integer = 999999): UInt = {
    val stop = min(width, cap)
    def helper(s: Int, x: UInt): UInt =
      if (s >= stop) x else helper(s+s, x | (x >> s))
    helper(1, x)(width-1, 0)
  }
}

class LSBOrTestModule(width: Int) extends Module {
  val input = IO(Input(UInt(width.W)))

  var lsb = false.B
  val vec = for(b <- input.asBools) yield {
    val cur = b || lsb
    lsb = cur
    cur
  }
  val ref = VecInit(vec).asUInt
  val rocketRef = RocketImpl.leftOR(input)

  val testee = LSBOr(input)

  assert(testee === ref)
  assert(testee === rocketRef)
}

class MSBOrTestModule(width: Int) extends Module {
  val input = IO(Input(UInt(width.W)))

  val ref = Reverse(LSBOr(Reverse(input)))
  val rocketRef = RocketImpl.rightOR(input)
  val testee = MSBOr(input)

  assert(testee === ref)
  assert(testee === rocketRef)
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