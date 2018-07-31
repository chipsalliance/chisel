// See LICENSE for license details.

package chiselTests

import Chisel.testers.BasicTester
import chisel3._
import org.scalatest._

//scalastyle:off magic.number

class InfiniteExtractMuxSpec extends FreeSpec with Matchers with ChiselRunners {
  "All four common infinite-width extract cases should work" in {
    assertTesterPasses(new SimpleInfiniteExtractTester)
  }
}

class SimpleInfiniteExtractTester extends BasicTester {
  val a = Wire(Bool())
  val b = Wire(Bool())
  val c = Wire(Bool())
  val d = Wire(Bool())

  a := 5.U.i(0)
  b := 5.U.i(25)
  c := -2.S.i(0)
  d := -2.S.i(25)
  assert(a === 1.U)
  assert(b === 0.U)
  assert(c === 0.U)
  assert(d === 1.U)
  stop()
}
