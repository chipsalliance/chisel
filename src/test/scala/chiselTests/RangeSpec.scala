// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest.{Matchers, FreeSpec}

class RangeSpec extends FreeSpec with Matchers {
  "Ranges can be specified for UInt, SInt, and FixedPoint" - {
    "to specify a UInt" in {
      val x = UInt(range"[0, 7)")
      x.getWidth should be (3)

      println(range"[4,32)")
    }
 }
}
