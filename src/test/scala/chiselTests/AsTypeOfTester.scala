// See LICENSE for license details.

package chiselTests

import org.scalatest._

import chisel3._
import chisel3.experimental.{DataMirror, FixedPoint}
import chisel3.testers.BasicTester
import chisel3.util._

class AsTypeOfBundleTester extends BasicTester {
  class MultiTypeBundle extends Bundle {
    val u  = UInt(4.W)
    val s  = SInt(4.W)
    val fp = FixedPoint(4.W, 3.BP)
  }

  val bun = new MultiTypeBundle

  val bunAsTypeOf = ((4 << 8) + (15 << 4) + (12 << 0)).U.asTypeOf(bun)

  assert(bunAsTypeOf.u === 4.U)
  assert(bunAsTypeOf.s === -1.S)
  assert(bunAsTypeOf.fp === FixedPoint.fromDouble(-0.5, 4.W, 3.BP))

  stop()
}

class AsTypeOfVecTester extends BasicTester {
  val vec = ((15 << 12) + (0 << 8) + (1 << 4) + (2 << 0)).U.asTypeOf(Vec(4, SInt(4.W)))

  assert(vec(0) === 2.S)
  assert(vec(1) === 1.S)
  assert(vec(2) === 0.S)
  assert(vec(3) === -1.S)

  stop()
}

class AsTypeOfTruncationTester extends BasicTester {
  val truncate = (64 + 3).U.asTypeOf(UInt(3.W))
  val expand   = 1.U.asTypeOf(UInt(3.W))

  assert( DataMirror.widthOf(truncate).get == 3 )
  assert( truncate === 3.U )
  assert( DataMirror.widthOf(expand).get == 3 )
  assert( expand === 1.U )

  stop()
}

class ResetAsTypeOfBoolTester extends BasicTester {
  assert(reset.asTypeOf(Bool()) === reset.toBool)
  stop()
}

class AsTypeOfSpec extends ChiselFlatSpec {
  behavior of "asTypeOf"

  it should "work with Bundles containing Bits Types" in {
    assertTesterPasses{ new AsTypeOfBundleTester }
  }

  it should "work with Vecs containing Bits Types" in {
    assertTesterPasses{ new AsTypeOfVecTester }
  }

  it should "expand and truncate UInts of different width" in {
    assertTesterPasses{ new AsTypeOfTruncationTester }
  }

  it should "work for casting implicit Reset to Bool" in {
    assertTesterPasses{ new ResetAsTypeOfBoolTester  }
  }
}
