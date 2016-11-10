// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.internal.firrtl.{Open, Closed}
import org.scalatest.{Matchers, FreeSpec}

class RangeSpec extends FreeSpec with Matchers {
  "Ranges can be specified for UInt, SInt, and FixedPoint" - {
    "range macros should allow open and closed bounds" in {
      {
        val (lo, hi) = range"[-1, 1)"
        lo should be (Closed(-1))
        hi should be (Open(1))
      }
      {
        val (lo, hi) = range"[-1, 1]"
        lo should be (Closed(-1))
        hi should be (Closed(1))
      }
      {
        val (lo, hi) = range"(-1, 1]"
        lo should be (Open(-1))
        hi should be (Closed(1))
      }
      {
        val (lo, hi) = range"(-1, 1)"
        lo should be (Open(-1))
        hi should be (Open(1))
      }
    }
    "UInt should get the correct width from a range" in {
      UInt(range"[0, 8)").getWidth should be (3)

      UInt(range"[0, 8]").getWidth should be (4)

      UInt(range"[0, 0]").getWidth should be (1)
    }

    "SInt should get the correct width from a range" in {
      SInt(range"[0, 8)").getWidth should be (4)

      SInt(range"[0, 8]").getWidth should be (5)

      SInt(range"[-4, 4)").getWidth should be (3)

      SInt(range"[0, 0]").getWidth should be (1)
    }

    "UInt should check that the range is valid" in {
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
    }

    "SInt should check that the range is valid" in {
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
    }
  }
}
