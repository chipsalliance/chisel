// See LICENSE for license details.

package chiselTests

import chisel3._
import org.scalatest.{Matchers, FreeSpec}

class RangeSpec extends FreeSpec with Matchers {
  "Ranges can be specified for UInt, SInt, and FixedPoint" - {
    "to specify a UInt" in {
      val x = UInt(range"[0, 8)")
      x.getWidth should be (3)

      println(range"[4,32)")

      UInt(range"[0, 8]").getWidth should be (4)
    }

    "to specify an SInt" in {
      SInt(range"[0, 8)").getWidth should be (4)

      SInt(range"[0, 8]").getWidth should be (5)

      SInt(range"[-4, 4)").getWidth should be (3)
    }

    "it should check that the range is valid for UInt" in {
      an [IllegalArgumentException] should be thrownBy {
        UInt(range"[1, 0]")
      }

      an [IllegalArgumentException] should be thrownBy {
        UInt(range"[-1, 1]")
      }

      an [IllegalArgumentException] should be thrownBy {
        UInt(range"(0,0]")
      }

      an [IllegalArgumentException] should be thrownBy {
        UInt(range"[0,0)")
      }

      an [IllegalArgumentException] should be thrownBy {
        UInt(range"(0,0)")
      }

      UInt(range"[0, 0]").getWidth should be (1)
    }

    "it should check that the range is valid for SInt" in {
      an [IllegalArgumentException] should be thrownBy {
        SInt(range"[1, 0]")
      }

      an [IllegalArgumentException] should be thrownBy {
        SInt(range"(0,0]")
      }

      an [IllegalArgumentException] should be thrownBy {
        SInt(range"[0,0)")
      }

      an [IllegalArgumentException] should be thrownBy {
        SInt(range"(0,0)")
      }

      SInt(range"[0, 0]").getWidth should be (1)
    }
  }
}
