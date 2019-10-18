// See README.md for license details.

package chiselTests

import chisel3._
import chisel3.experimental._
import _root_.firrtl.{ir => firrtlir}
import chisel3.internal.firrtl.{BinaryPoint, IntervalRange, KnownBinaryPoint, UnknownBinaryPoint}
import org.scalatest.{FreeSpec, Matchers}

//scalastyle:off method.name magic.number
class IntervalRangeSpec extends FreeSpec with Matchers {

  "IntervalRanges" - {
    def C(b: BigDecimal): firrtlir.Bound = firrtlir.Closed(b)

    def O(b: BigDecimal): firrtlir.Bound = firrtlir.Open(b)

    def U(): firrtlir.Bound = firrtlir.UnknownBound

    def UBP(): BinaryPoint = UnknownBinaryPoint

    def checkRange(r: IntervalRange, l: firrtlir.Bound, u: firrtlir.Bound, b: BinaryPoint): Unit = {
      r.lowerBound should be(l)
      r.upperBound should be(u)
      r.binaryPoint should be(b)
    }

    def checkBinaryPoint(r: IntervalRange, b: BinaryPoint): Unit = {
      r.binaryPoint should be(b)
    }

    "IntervalRange describes the range of values of the Interval Type" - {
      "Factory methods can create IntervalRanges" - {
        "ranges can start or end open or closed, default binary point is none" in {
          checkRange(range"[0,10]", C(0), C(10), 0.BP)
          checkRange(range"[-1,10)", C(-1), O(10), 0.BP)
          checkRange(range"(11,12]", O(11), C(12), 0.BP)
          checkRange(range"(-21,-10)", O(-21), O(-10), 0.BP)
        }

        "ranges can have unknown bounds" in {
          checkRange(range"[?,10]", U(), C(10), 0.BP)
          checkRange(range"(?,10]", U(), C(10), 0.BP)
          checkRange(range"[-1,?]", C(-1), U(), 0.BP)
          checkRange(range"[-1,?)", C(-1), U(), 0.BP)
          checkRange(range"[?,?]", U(), U(), 0.BP)
          checkRange(range"[?,?].?", U(), U(), UBP())
        }

        "binary points can be specified" in {
          checkBinaryPoint(range"[?,10].0", 0.BP)
          checkBinaryPoint(range"[?,10].2", 2.BP)
          checkBinaryPoint(range"[?,10].?", UBP())
        }
        "malformed ranges will throw ChiselException or are compile time errors" in {
          // must be a cleverer way to show this
          intercept[ChiselException] {
            range"[19,5]"
          }
          assertDoesNotCompile(""" range"?,10] """)
          assertDoesNotCompile(""" range"?,? """)
        }
      }
    }

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
        assertCompiles(""" range"[0, 1]" """) // syntax sanity check
      }

      "range macros should allow open and closed bounds" in {
        range"[-1, 1)" should be(range"[-1,1).0")
        range"[-1, 1)" should be(IntervalRange(C(-1), O(1), 0.BP))
        range"[-1, 1]" should be(IntervalRange(C(-1), C(1), 0.BP))
        range"(-1, 1]" should be(IntervalRange(O(-1), C(1), 0.BP))
        range"(-1, 1)" should be(IntervalRange(O(-1), O(1), 0.BP))
      }

      "range specifiers should be whitespace tolerant" in {
        range"[-1,1)" should be(IntervalRange(C(-1), O(1), 0.BP))
        range" [-1,1) " should be(IntervalRange(C(-1), O(1), 0.BP))
        range" [ -1 , 1 ) " should be(IntervalRange(C(-1), O(1), 0.BP))
        range"   [   -1   ,   1   )   " should be(IntervalRange(C(-1), O(1), 0.BP))
      }

      "range macros should work with interpolated variables" in {
        val a = 10
        val b = -3

        range"[$b, $a)" should be(IntervalRange(C(b), O(a), 0.BP))
        range"[${a + b}, $a)" should be(IntervalRange(C(a + b), O(a), 0.BP))
        range"[${-3 - 7}, ${-3 + a})" should be(IntervalRange(C(-10), O(-3 + a), 0.BP))

        def number(n: Int): Int = n

        range"[${number(1)}, ${number(3)})" should be(IntervalRange(C(1), O(3), 0.BP))
      }

      "UInt should get the correct width from a range" in {
        UInt(range"[0, 8)").getWidth should be(3)
        UInt(range"[0, 8]").getWidth should be(4)
        UInt(range"[0, 0]").getWidth should be(1)
      }

      "SInt should get the correct width from a range" in {
        SInt(range"[0, 8)").getWidth should be(4)
        SInt(range"[0, 8]").getWidth should be(5)
        SInt(range"[-4, 4)").getWidth should be(3)
        SInt(range"[0, 0]").getWidth should be(1)
      }

      "UInt should check that the range is valid" in {
        an[ChiselException] should be thrownBy {
          UInt(range"[1, 0]")
        }
        an[ChiselException] should be thrownBy {
          UInt(range"[-1, 1]")
        }
        an[ChiselException] should be thrownBy {
          UInt(range"(0,0]")
        }
        an[ChiselException] should be thrownBy {
          UInt(range"[0,0)")
        }
        an[ChiselException] should be thrownBy {
          UInt(range"(0,0)")
        }
        an[ChiselException] should be thrownBy {
          UInt(range"(0,1)")
        }
      }

      "SInt should check that the range is valid" in {
        an[ChiselException] should be thrownBy {
          SInt(range"[1, 0]")
        }
        an[ChiselException] should be thrownBy {
          SInt(range"(0,0]")
        }
        an[ChiselException] should be thrownBy {
          SInt(range"[0,0)")
        }
        an[ChiselException] should be thrownBy {
          SInt(range"(0,0)")
        }
        an[ChiselException] should be thrownBy {
          SInt(range"(0,1)")
        }
      }
    }
  }

}
