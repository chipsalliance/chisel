// See LICENSE for license details.

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util._
import chisel3.core.DataMirror

class FromBitsBundleTester extends BasicTester {
  class MultiTypeBundle extends Bundle {
    val u  = UInt(4.W)
    val s  = SInt(4.W)
    val fp = FixedPoint(4.W, 3.BP)
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

class FromBitsTruncationTester extends BasicTester {
  val truncate = UInt(3.W).fromBits( (64 + 3).U )
  val expand   = UInt(3.W).fromBits( 1.U )

  assert( DataMirror.widthOf(truncate).get == 3 )
  assert( truncate === 3.U )
  assert( DataMirror.widthOf(expand).get == 3 )
  assert( expand === 1.U )

  stop()
}

class FromBitsSpec extends ChiselFlatSpec {
  behavior of "fromBits"

  it should "work with Bundles containing Bits Types" in {
    assertTesterPasses{ new FromBitsBundleTester }
  }

  it should "work with Vecs containing Bits Types" in {
    assertTesterPasses{ new FromBitsVecTester }
  }

  it should "expand and truncate UInts of different width" in {
    assertTesterPasses{ new FromBitsTruncationTester }
  }
}
