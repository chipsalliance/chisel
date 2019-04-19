// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.ChiselRange

import chisel3.internal.firrtl.{Open, Closed}
import org.scalatest.{Matchers, FreeSpec}

class RangeSpec extends FreeSpec with Matchers {
  "Ranges can be specified for UInt, SInt, and FixedPoint" - {
    "invalid range specifiers should fail at compile time" in {
      assertDoesNotCompile(""" range"" """)
      assertDoesNotCompile(""" range"[]" """)
      assertDoesNotCompile(""" range"0" """)
      assertDoesNotCompile(""" range"[0]" """)
      assertDoesNotCompile(""" range"[0, 1" """)
      assertDoesNotCompile(""" range"0, 1]" """)
      assertDoesNotCompile(""" range"[0, 1, 2]" """)
      assertDoesNotCompile(""" range"[a]" """)
      assertDoesNotCompile(""" range"[a, b]" """)
      assertCompiles(""" range"[0, 1]" """)  // syntax sanity check
    }

    "range macros should allow open and closed bounds" in {
      range"[-1, 1)" should be( (Closed(-1), Open(1)) )
      range"[-1, 1]" should be( (Closed(-1), Closed(1)) )
      range"(-1, 1]" should be( (Open(-1), Closed(1)) )
      range"(-1, 1)" should be( (Open(-1), Open(1)) )
    }

    "range specifiers should be whitespace tolerant" in {
      range"[-1,1)" should be( (Closed(-1), Open(1)) )
      range" [-1,1) " should be( (Closed(-1), Open(1)) )
      range" [ -1 , 1 ) " should be( (Closed(-1), Open(1)) )
      range"   [   -1   ,   1   )   " should be( (Closed(-1), Open(1)) )
    }

    "range macros should work with interpolated variables" in {
      val a = 10
      val b = -3

      range"[$b, $a)" should be( (Closed(b), Open(a)) )
      range"[${a + b}, $a)" should be( (Closed(a + b), Open(a)) )
      range"[${-3 - 7}, ${-3 + a})" should be( (Closed(-10), Open(-3 + a)) )

      def number(n: Int): Int = n
      range"[${number(1)}, ${number(3)})" should be( (Closed(1), Open(3)) )
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
      an [IllegalArgumentException] should be thrownBy {
        UInt(range"(0,1)")
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
      an [IllegalArgumentException] should be thrownBy {
        SInt(range"(0,1)")
      }
    }
  }
}
