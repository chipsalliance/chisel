// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.ChiselRange
import chisel3.internal.firrtl.{IntervalRange, KnownBinaryPoint}
import firrtl.ir.{Closed, Open}
import org.scalatest.{FreeSpec, Matchers}

//noinspection ScalaStyle
//scalastyle:off magic.number
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
      range"[-1, 1]" should be( IntervalRange(Closed(-1), Closed(1), 0) )
      range"[-1, 1)" should be( IntervalRange(Closed(-1), Open(1), 0))
      range"(-1, 1]" should be( IntervalRange(Open(-1), Closed(1), 0) )
      range"(-1, 1)" should be( IntervalRange(Open(-1), Open(1), 0) )
    }

    "range specifiers should be whitespace tolerant" in {
      range"[-1,1)" should be( IntervalRange(Closed(-1), Open(1), 0) )
      range" [-1,1) " should be( IntervalRange(Closed(-1), Open(1), 0) )
      range" [ -1 , 1 ) " should be( IntervalRange(Closed(-1), Open(1), 0) )
      range"   [   -1   ,   1   )   " should be( IntervalRange(Closed(-1), Open(1), 0) )
    }

    "range macros should work with interpolated variables" in {
      val a = 10
      val b = -3

      range"[$b, $a)" should be( IntervalRange(Closed(b), Open(a), 0) )
      range"[${a + b}, $a)" should be( IntervalRange(Closed(a + b), Open(a), 0) )
      range"[${-3 - 7}, ${-3 + a})" should be( IntervalRange(Closed(-10), Open(-3 + a), 0) )

      def number(n: Int): Int = n
      range"[${number(1)}, ${number(3)})" should be( IntervalRange(Closed(1), Open(3), 0) )
    }

    "range macros support precision" in {
      val a = range"[2, 6).5"
      a.serialize should be ("Interval[2, 6).5")
      val b = range"[0, 16).4"
      b.serialize should be ("Interval[0, 16).4")
      val c = range"[1,2]"
      c.serialize should be ("Interval[1, 2].0")
      val d = range"(?, ?)"
      d.serialize should be ("Interval.0")

    }

    "UInt should get the correct width from a range" in {
      val r = IntervalRange(Closed(0), Closed(8), 0)
      r.getWidth.get should be (5)
      UInt(range"[0, 8]").getWidth should be (4)
      UInt(range"[0, 8)").getWidth should be (3)
      UInt(range"[0, 0]").getWidth should be (0)
    }

    "SInt should get the correct width from a range" in {
      SInt(range"[0, 8)").getWidth should be (4)
      SInt(range"[0, 8]").getWidth should be (5)
      SInt(range"[-4, 4)").getWidth should be (3)
      SInt(range"[0, 0]").getWidth should be (0)
    }

    "Range should be left shiftable" in {
      var r1 = range"[0, 4].1"
      var r2 = r1 << 2
      r2.lower should be (Closed(0))
      r2.upper should be (Closed(16))
      r2.binaryPoint should be (KnownBinaryPoint(1))

      r1 = range"[-3, 3].1"
      r2 = r1 << 2
      r2.lower should be (Closed(-12))
      r2.upper should be (Closed(12))
      r2.binaryPoint should be (KnownBinaryPoint(1))
    }

    "Range should be right shiftable" in {
      var r1 = range"[0, 4].9"
      var r2 = r1 >> 1
      r2.lower should be (Closed(0))
      r2.upper should be (Closed(2))
      r2.binaryPoint should be (KnownBinaryPoint(9))

      r1 = range"[-12, 12].9"
      r2 = r1 >> 2
      r2.lower should be (Closed(-3))
      r2.upper should be (Closed(3))
      r2.binaryPoint should be (KnownBinaryPoint(9))
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
        val r = range"(0,1)"
        println(s"r = $r.serialize")
        SInt(range"(0,1)")
      }
    }
  }
}
