// See LICENSE for license details.

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.experimental.{DataMirror, FixedPoint}
import chisel3.testers.BasicTester
import chisel3.util._

class AsBundleTester extends BasicTester {
  class MultiTypeBundle extends Bundle {
    val u  = UInt(4.W)
    val s  = SInt(4.W)
    val fp = FixedPoint(4.W, 3.BP)
  }

  val rawData = ((4 << 8) + (15 << 4) + (12 << 0)).U
  val bunFromBits = rawData.asTypeOf(new MultiTypeBundle)

  assert(bunFromBits.u === 4.U)
  assert(bunFromBits.s === -1.S)
  assert(bunFromBits.fp === FixedPoint.fromDouble(-0.5, 4.W, 3.BP))

  stop()
}

class AsVecTester extends BasicTester {
  val rawData = ((15 << 12) + (0 << 8) + (1 << 4) + (2 << 0)).U
  val vec = rawData.asTypeOf(Vec(4, SInt(4.W)))

  assert(vec(0) === 2.S)
  assert(vec(1) === 1.S)
  assert(vec(2) === 0.S)
  assert(vec(3) === -1.S)

  stop()
}

class AsBitsTruncationTester extends BasicTester {
  val truncate = (64 + 3).U.asTypeOf(UInt(3.W))
  val expand   = 1.U.asTypeOf(UInt(3.W))

  assert( DataMirror.widthOf(truncate).get == 3 )
  assert( truncate === 3.U )
  assert( DataMirror.widthOf(expand).get == 3 )
  assert( expand === 1.U )

  stop()
}

class AsBitsSpec extends ChiselFlatSpec {
  behavior of "fromBits"

  it should "work with Bundles containing Bits Types" in {
    assertTesterPasses{ new AsBundleTester }
  }

  it should "work with Vecs containing Bits Types" in {
    assertTesterPasses{ new AsVecTester }
  }

  it should "expand and truncate UInts of different width" in {
    assertTesterPasses{ new AsBitsTruncationTester }
  }
}
