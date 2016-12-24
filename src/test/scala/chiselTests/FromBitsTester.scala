// See LICENSE for license details.

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._
import chisel3.internal.firrtl.KnownBinaryPoint

class FromBitsBundleTester extends BasicTester {
  class MultiTypeBundle extends Bundle {
    val u  = UInt(4.W)
    val s  = SInt(4.W)
    val fp = FixedPoint(4.W, KnownBinaryPoint(3))
  }

  val bun = new MultiTypeBundle

  val bunFromBits = bun.fromBits( ((4 << 8) + (15 << 4) + (12 << 0)).U )

  assert(bunFromBits.u === 4.U)
  assert(bunFromBits.s === -1.S)
  assert(bunFromBits.fp === FixedPoint.fromDouble(-0.5, width=4, binaryPoint=3))

  stop()
}

class FromBitsVecTester extends BasicTester {
  val vec = Vec(4, SInt(4.W)).fromBits( ((15 << 12) + (0 << 8) + (1 << 4) + (2 << 0)).U )

  assert(vec(0) === 2.S)
  assert(vec(1) === 1.S)
  assert(vec(2) === 0.S)
  assert(vec(3) === -1.S)

  stop()
}

class FromBitsSpec extends ChiselFlatSpec {
  "fromBits" should "work with Bundles containing Bits Types" in {
    assertTesterPasses{ new FromBitsBundleTester }
  }

  "fromBits" should "work with Vecs containing Bits Types" in {
    assertTesterPasses{ new FromBitsVecTester }
  }
}
