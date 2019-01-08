// See LICENSE for license details.

package chiselTests

import _root_.firrtl.ir.Closed
import chisel3._
import chisel3.core.{FixedPoint, stop}
import chisel3.experimental.{ChiselRange, Interval, RawModule}
import chisel3.internal.firrtl.{IntervalRange, KnownBinaryPoint}
import chisel3.internal.sourceinfo.{SourceInfo, UnlocatableSourceInfo}
import chisel3.testers.BasicTester
import cookbook.CookbookTester
import firrtl.FIRRTLException
import org.scalatest.{FreeSpec, Matchers}

import scala.language.reflectiveCalls

//scalastyle:off magic.number

class IntervalTest1 extends Module {
  //noinspection TypeAnnotation
  val io = IO(new Bundle {
    val in1 = Input(Interval(6.W, range"[0,4].3"))
    val in2 = Input(Interval(6.W, range"[0,4].3"))
    val out = Output(Interval(8.W, range"[0,8].3"))
  })

  io.out := io.in1 + io.in2
}

class IntervalTester extends CookbookTester(10) {
  val dut = Module(new IntervalTest1)

  dut.io.in1 := 4.I
  dut.io.in2 := 4.I
  assert(dut.io.out === 8.I)

  val i = Interval(range"[0,10)")
  stop()
}

class IntervalTest2 extends Module {
  //noinspection TypeAnnotation
  val io = IO(new Bundle {
    val p = Input(Bool())
    val in1 = Input(Interval(range"[0,4]"))
    val in2 = Input(Interval(range"[0,6]"))
    val out = Output(Interval())
  })

  io.out := Mux(io.p, io.in1, io.in2)
}

class IntervalTester2 extends CookbookTester(10) {
  val dut = Module(new IntervalTest2)

  dut.io.p := 1.U
  dut.io.in1 := 4.I
  dut.io.in2 := 5.I
  assert(dut.io.out === 4.I)

  stop()
}

class IntervalAddTester extends BasicTester {
  val in1 = Wire(Interval(range"[0,4]"))
  val in2 = Wire(Interval(range"[0,4]"))

  in1 := 2.I
  in2 := 2.I

  val result = in1 +& in2

  assert(result === 4.I)

  stop()

}

class IntervalSetBinaryPointTester extends BasicTester {
  implicit val sourceinfo: SourceInfo = UnlocatableSourceInfo
  val in1 = Wire(Interval(range"[0,4].4"))
  val in2 = in1.setBinaryPoint(2)

  assert(in2.binaryPoint == KnownBinaryPoint(2))

  in1 := 2.I

  val shiftedLeft = in1.shiftLeftBinaryPoint(2)

  assert(shiftedLeft.binaryPoint == KnownBinaryPoint(6), s"Error: bpshl result ${shiftedLeft.range} expected bt = 2")

  val shiftedRight = in1.shiftRightBinaryPoint(2)

  assert(shiftedRight.binaryPoint == KnownBinaryPoint(2), s"Error: bpshl result ${shiftedRight.range} expected bt = 2")

  stop()
}

class MoreIntervalShiftTester extends BasicTester {
  implicit val sourceinfo: SourceInfo = UnlocatableSourceInfo

  val in1 = Wire(Interval(range"[0,4].4"))
  val in2 = in1.setBinaryPoint(2)

  assert(in2.binaryPoint == KnownBinaryPoint(2))

  val toShiftLeft = Wire(Interval(range"[0,4].4"))
  val shiftedLeft = in1.shiftLeftBinaryPoint(2)

  assert(shiftedLeft.binaryPoint == KnownBinaryPoint(2), s"Error: bpshl result ${shiftedLeft.range} expected bt = 2")

  val toShiftRight = Wire(Interval(range"[0,4].4"))
  val shiftedRight = in1.shiftRightBinaryPoint(2)

  assert(shiftedRight.binaryPoint == KnownBinaryPoint(6), s"Error: bpshl result ${shiftedRight.range} expected bt = 2")

  stop()
}

/**
  * This is a reality check not a test. Makes it easier to figure out
  * what is going on in other places
  * @param range        a range for inputs
  * @param targetRange  a range for outputs
  * @param startNum     start here
  * @param endNum       end here
  * @param incNum       increment by this
  */
class ClipSqueezeWrapDemo(
  range: IntervalRange,
  targetRange: IntervalRange,
  startNum: Double,
  endNum: Double,
  incNum: Double
) extends BasicTester {

  val binaryPointAsInt = range.binaryPoint.asInstanceOf[KnownBinaryPoint].value
  val startValue = Interval.fromDouble(startNum, binaryPoint = binaryPointAsInt)
  val increment = Interval.fromDouble(incNum, binaryPoint = binaryPointAsInt)
  val endValue  = Interval.fromDouble(endNum, binaryPoint = binaryPointAsInt)

  val counter = RegInit(Interval(range), startValue)

  counter := (counter + increment).squeeze(counter)
  when(counter > endValue) {
    stop()
  }

  val clipped  = counter.clip(0.U.asInterval(targetRange))
  val squeezed = counter.squeeze(0.U.asInterval(targetRange))
  val wrapped  = counter.wrap(0.U.asInterval(targetRange))

  when(counter === startValue) {
    printf(s"Target range is $range\n")
    printf("value     clip      squeeze      wrap\n")
  }

  printf("       %d       %d          %d         %d\n",
    counter.asSInt(), clipped.asSInt(), squeezed.asSInt(), wrapped.asSInt())
}

class SqueezeFunctionalityTester(
  range: IntervalRange,
  startNum: BigDecimal,
  endNum: BigDecimal,
  increment: BigDecimal
) extends BasicTester {

  val counter = RegInit(0.U(10.W))
  counter := counter + 1.U
  when(counter > 10.U) {
    stop()
  }

  val squeezeInterval = Wire(Interval(range))
  squeezeInterval := 0.I

  val squoozen = Wire(Interval(range))

  val ss = WireInit(Interval(range), (-10).S.asInterval(range))

  val toSqueeze = counter.asInterval(range) - ss

  squoozen := toSqueeze.squeeze(squeezeInterval)

  printf(s"SqueezeTest %d    %d.squeeze($range) => %d\n", counter, toSqueeze.asSInt(), squoozen.asSInt())
}

/**
  * Demonstrate a simple counter register with an Interval type
  */
class IntervalRegisterTester extends BasicTester {
  val range = range"[-2,5]"
  val counter = RegInit(Interval(range), (-1).I)
  counter := (counter + 1.I).squeeze(counter)  // this works with other types, why not Interval
  when(counter > 4.I) {
    stop()
  }
}

//noinspection ScalaStyle
class IntervalWrapTester extends BasicTester {
  val t1 = Wire(Interval(range"[-2, 12]"))
  t1 := (-2).I
  val u1 = 0.U(3.W)
  val r1 = RegInit(u1)
  r1 := u1
  val t2 = t1.wrap(u1)
  val t3 = t1.wrap(r1)

  assert(t2.range.upper == Closed(7), s"t1 upper ${t2.range.upper} expected ${Closed(7)}")
  assert(t3.range.upper == Closed(7), s"t1 upper ${t3.range.upper} expected ${Closed(7)}")

  val in1 = WireInit(Interval(range"[0,9].6"), 0.I)
  val in2 = WireInit(Interval(range"[1,6).4"), 2.I)
  val in3 = in1.wrap(in2)

  assert(in3.range.lower == Closed(1), s"in3 lower ${in3.range.lower} expected ${Closed(1)}")
  assert(in3.range.upper == Closed(5.9375), s"in3 upper ${in3.range.upper} expected ${Closed(5.9375)}")
  assert(in3.binaryPoint == KnownBinaryPoint(6), s"in3 binaryPoint ${in3.binaryPoint} expected ${KnownBinaryPoint(2)}")

  val enclosedRange = range"[-2, 5]"
  val base = Wire(Interval(range"[-4, 6]"))
  val enclosed = WireInit(Interval(enclosedRange), 0.I)
  val enclosing = WireInit(Interval(range"[-6, 8]"), 0.I)
  val overlapLeft = WireInit(Interval(range"[-10,-2]"), (-3).I)
  val overlapRight = WireInit(Interval(range"[-1,10]"), 0.I)

  val w1 = base.wrap(enclosed)
  val w2 = base.wrap(enclosing)
  val w3 = base.wrap(overlapLeft)
  val w4 = base.wrap(overlapRight)
  val w7 = base.wrap(enclosedRange)

  base := 6.I

  assert(w1 === (-2).I)
  assert(w2 === 6.I)
  assert(w3 === (-3).I)
  assert(w4 === 6.I)
  assert(w7 === (-2).I)

  stop()
}

class IntervalClipTester extends BasicTester {
  val enclosedRange = range"[-2, 5]"
  val base = Wire(Interval(range"[-4, 6]"))
  val enclosed = Wire(Interval(enclosedRange))
  val enclosing = Wire(Interval(range"[-6, 8]"))
  val overlapLeft = Wire(Interval(range"[-10,-2]"))
  val overlapRight = Wire(Interval(range"[-1,10]"))
  val disjointLeft = Wire(Interval(range"[-14,-7]"))
  val disjointRight = Wire(Interval(range"[7,11]"))

  enclosed := DontCare
  enclosing := DontCare
  overlapLeft := DontCare
  overlapRight := DontCare
  disjointLeft := DontCare
  disjointRight := DontCare

  val enclosedResult = base.clip(enclosed)
  val enclosingResult = base.clip(enclosing)
  val overlapLeftResult = base.clip(overlapLeft)
  val overlapRightResult = base.clip(overlapRight)
  val disjointLeftResult = base.clip(disjointLeft)
  val disjointRightResult = base.clip(disjointRight)
  val enclosedViaRangeString = base.clip(enclosedRange)

  base := 6.I

  assert(enclosedResult === 5.I)
  assert(enclosingResult === 6.I)
  assert(overlapLeftResult === (-2).I)
  assert(overlapRightResult === 6.I)
  assert(disjointLeftResult === (-7).I)
  assert(disjointRightResult === 7.I)

  assert(enclosedViaRangeString === 5.I)

  stop()
}

class IntervalChainedAddTester extends BasicTester {
  val intervalResult = Wire(Interval())
  val uintResult = Wire(UInt())

  intervalResult := 1.I + 1.I + 1.I + 1.I + 1.I + 1.I + 1.I
  uintResult := 1.U +& 1.U +& 1.U +& 1.U +& 1.U +& 1.U +& 1.U

  assert(intervalResult === 7.I)
  assert(uintResult === 7.U)
  stop()
}

class IntervalChainedMulTester extends BasicTester {
  val intervalResult = Wire(Interval())
  val uintResult = Wire(UInt())

  intervalResult := 2.I * 2.I * 2.I * 2.I * 2.I * 2.I * 2.I
  uintResult := 2.U * 2.U * 2.U * 2.U * 2.U * 2.U * 2.U

  assert(intervalResult === 128.I)
  assert(uintResult === 128.U)
  stop()
}

class IntervalChainedSubTester extends BasicTester {
  val intervalResult1 = Wire(Interval())
  val intervalResult2 = Wire(Interval())
  val uIntResult = Wire(UInt())
  val sIntResult = Wire(SInt())
  val fixedResult = Wire(FixedPoint())

  intervalResult1 := 17.I - 2.I - 2.I - 2.I - 2.I - 2.I - 2.I // gives same result as -& operand version below
  intervalResult2 := 17.I -& 2.I -& 2.I -& 2.I -& 2.I -& 2.I -& 2.I
  uIntResult := 17.U -& 2.U -& 2.U -& 2.U -& 2.U -& 2.U -& 2.U
  fixedResult := 17.0.F(0.BP) -& 2.0.F(0.BP) -& 2.0.F(0.BP) -& 2.0.F(0.BP) -& 2.0.F(0.BP) -& 2.0.F(0.BP) -& 2.0.F(0.BP)
  sIntResult := 17.S -& 2.S -& 2.S -& 2.S -& 2.S -& 2.S -& 2.S

  assert(uIntResult === 5.U)
  assert(sIntResult === 5.S)
  assert(fixedResult.asUInt === 5.U)
  assert(intervalResult1 === 5.I)
  assert(intervalResult2 === 5.I)

  stop()
}

class IntervalSpec extends FreeSpec with Matchers with ChiselRunners {
  "Test a simple interval add" in {
    assertTesterPasses{ new IntervalAddTester }
  }
  "Intervals can be created" in {
    assertTesterPasses{ new IntervalTester }
  }
  "Test a simple interval mux" in {
    assertTesterPasses{ new IntervalTester2 }
  }
  "Intervals can have binary points set" in {
    assertTesterPasses{ new IntervalSetBinaryPointTester }
  }
  "Let's take a look at the results of squeeze over small range" in {
    assertTesterPasses{ new ClipSqueezeWrapDemo(
      range = range"[-10,33].0",
      targetRange = range"[-4,17].0",
      startNum = -4.0, endNum = 30.0, incNum = 1.0
    )}
    assertTesterPasses{ new ClipSqueezeWrapDemo(
      range = range"[-2,5].1",
      targetRange = range"[-1,3].1",
      startNum = -2.0, endNum = 5.0, incNum = 0.5
    )}
  }
  "Intervals can be squeezed into another intervals range" in {
    assertTesterPasses{ new SqueezeFunctionalityTester(range"[-2,5]",
      BigDecimal(-10), BigDecimal(10), BigDecimal(1.0)) }
  }
  "Intervals can be wrapped with wrap operator" in {
    assertTesterPasses{ new IntervalWrapTester }
  }

  "Interval compile pathologies: clip, wrap, and squeeze have different behavior" - {
    "wrap target range is completely left of source" in {
      intercept[FIRRTLException] {
        assertTesterPasses(new BasicTester {
          val base = Wire(Interval(range"[-4, 6]"))
          base := 6.I
          val disjointLeft = WireInit(Interval(range"[-7,-5]"), (-6).I)
          val w5 = base.wrap(disjointLeft)
          stop()
        })
      }
    }
    "wrap target range is completely right of source" in {
      intercept[FIRRTLException] {
        assertTesterPasses(new BasicTester {
          val base = Wire(Interval(range"[-4, 6]"))
          base := 6.I
          val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
          val w5 = base.wrap(disjointLeft)
          stop()
        })
      }
    }
    "clip target range is completely left of source" in {
      intercept[FIRRTLException] {
        assertTesterPasses(new BasicTester {
          val base = Wire(Interval(range"[-4, 6]"))
          base := 6.I
          val disjointLeft = WireInit(Interval(range"[-7,-5]"), (-6).I)
          val w5 = base.clip(disjointLeft)
          stop()
        })
      }
    }
    "clip target range is completely right of source" in {
      intercept[FIRRTLException] {
        assertTesterPasses(new BasicTester {
          val base = Wire(Interval(range"[-4, 6]"))
          base := 6.I
          val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
          val w5 = base.clip(disjointLeft)
          stop()
        })
      }
    }
    "squeeze target range is completely right of source" in {
      intercept[FIRRTLException] {
        assertTesterPasses(new BasicTester {
          val base = Wire(Interval(range"[-4, 6]"))
          base := 6.I
          val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
          val w5 = base.squeeze(disjointLeft)
          stop()
        })
      }
    }
    "squeeze target range is completely left of source" in {
      intercept[FIRRTLException] {
        assertTesterPasses(new BasicTester {
          val base = Wire(Interval(range"[-4, 6]"))
          base := 6.I
          val disjointLeft = WireInit(Interval(range"[-7, -5]"), 8.I)
          val w5 = base.squeeze(disjointLeft)
          stop()
        })
      }
    }

    def makeCircuit(operation: String, sourceRange: IntervalRange, targetRange: IntervalRange): RawModule = {
      new Module {
        val io = IO(new Bundle { val out = Output(Interval())})
        val base = Wire(Interval(sourceRange))
        base := 6.I

        val disjointLeft = WireInit(Interval(targetRange), 8.I)
        val w5 = operation match {
          case "clip" => base.clip(disjointLeft)
          case "wrap" => base.wrap(disjointLeft)
          case "squeeze" => base.squeeze(disjointLeft)
        }
        io.out := w5
      }
    }

    "disjoint ranges should error when used with clip, wrap and squeeze" - {

      def doTest(disjointLeft: Boolean, operation: String): Unit = {
        val kindString = s"disjoint ${if (disjointLeft) "left" else "right"}"
        val (rangeA, rangeB) = if(disjointLeft) {
          (range"[-4, 6]", range"[7,10]")
        }
        else {
          (range"[7,10]", range"[-4, 6]")
        }
        try {
          makeLoFirrtl("low")(makeCircuit(operation, rangeA, rangeB))
          println(s"$kindString $operation got No exception")
        }
        catch {
          case e: FIRRTLException =>
            println(s"We don't want firrtl exceptions to hit user")
            throw e
          case t: Throwable =>
            println(s"$kindString $operation got exception ${t.getClass} ${t.getMessage}")
        }
      }

      "Range A disjoint left, operation clip should generate useful error" in {
        doTest(disjointLeft= true, "clip")
      }
      "Range A disjoint left, operation wrap should generate useful error" in {
        doTest(disjointLeft= true, "wrap")
      }
      "Range A disjoint left, operation squeeze should generate useful error" in {
        doTest(disjointLeft= true, "squeeze")
      }

      "Range A disjoint right, operation clip should generate useful error" in {
        doTest(disjointLeft= false, "clip")
      }
      "Range A disjoint right, operation wrap should generate useful error" in {
        doTest(disjointLeft= false, "wrap")
      }
      "Range A disjoint right, operation squeeze should generate useful error" in {
        doTest(disjointLeft= false, "squeeze")
      }
    }

    "Errors are sometimes inconsistent or incorrectly labelled as Firrtl Internal Error" - {
      "squeeze disjoint is not internal error when defined in BasicTester" in {
        try {
          val loFirrtl = makeLoFirrtl("low")(new BasicTester {
            val base = Wire(Interval(range"[-4, 6]"))
            val base2 = Wire(Interval(range"[-4, 6]"))
            base := 6.I
            base2 := 5.I
            val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
            val w5 = base.squeeze(disjointLeft)
            stop()
          })
          println(s"Why is this not an error")
        }
        catch {
          case e: firrtl.FIRRTLException =>
            println(s"Wrong error FirrtlException")
        }
      }
      "wrap disjoint is not internal error when defined in BasicTester" in {
        val loFirrtl = makeLoFirrtl("low")(new BasicTester {
          val base = Wire(Interval(range"[-4, 6]"))
          val base2 = Wire(Interval(range"[-4, 6]"))
          base := 6.I
          base2 := 5.I
          val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
          val w5 = base.squeeze(disjointLeft)
          stop()
        })
        println(s"Why is this not an error")
      }
      "squeeze disjoint from Module gives internal error" in {
        try {
          val loFirrtl = makeLoFirrtl("lo")(new Module {
            val io = IO(new Bundle {
              val out = Output(Interval())
            })
            val base = Wire(Interval(range"[-4, 6]"))
            base := 6.I

            val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
            val w5 = base.squeeze(disjointLeft)
            io.out := w5
          })
          println(s"LoFirrtl\n$loFirrtl")
        }
        catch {
          case e: firrtl.FIRRTLException =>
            println(s"Wrong error FirrtlException")
            throw e
        }
      }
      "clip disjoint from Module gives internal error" in {
        try {
          val loFirrtl = makeLoFirrtl("lo")(new Module {
            val io = IO(new Bundle {
              val out = Output(Interval())
            })
            val base = Wire(Interval(range"[-4, 6]"))
            base := 6.I

            val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
            val w5 = base.clip(disjointLeft)
            io.out := w5
          })
          println(s"LoFirrtl\n$loFirrtl")
        }
        catch {
          case e: firrtl.FIRRTLException =>
            println(s"Wrong error FirrtlException")
            throw e
        }
      }
      "wrap disjoint from Module gives internal error" in {
        try {
          val loFirrtl = makeLoFirrtl("lo")(new Module {
            val io = IO(new Bundle {
              val out = Output(Interval())
            })
            val base = Wire(Interval(range"[-4, 6]"))
            base := 6.I

            val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
            val w5 = base.wrap(disjointLeft)
            io.out := w5
          })
          println(s"LoFirrtl\n$loFirrtl")
        }
        catch {
          case e: firrtl.FIRRTLException =>
            println(s"Wrong error FirrtlException")
        }
      }
    }

    "assign literal out of range of interval" in {
      intercept[firrtl.passes.CheckTypes.InvalidConnect] {
        assertTesterPasses(new BasicTester {
          val base = Wire(Interval(range"[-4, 6]"))
          base := (-8).I
        })
      }
    }
  }

  "Intervals should catch assignment of literals outside of range" - {
    "when literal is too small" in {
      try {
        val loFirrtl = makeLoFirrtl("lo")(new Module {
          val io = IO(new Bundle {val out = Output(Interval())})
          val base = Wire(Interval(range"[-4, 6]"))
          base := (-8).I
          io.out := base
        })
      }
      catch {
        case e: firrtl.FIRRTLException =>
          println(s"Wrong error FirrtlException")
      }
    }
    "when literal is too big" in {
      try {
        val loFirrtl = makeLoFirrtl("lo")(new Module {
          val io = IO(new Bundle {val out = Output(Interval())})
          val base = Wire(Interval(range"[-4, 6]"))
          base := (66).I
          io.out := base
        })
      }
      catch {
        case e: firrtl.FIRRTLException =>
          println(s"Wrong error FirrtlException")
      }
    }
  }

  "Intervals can be used to construct registers" in {
    assertTesterPasses{ new IntervalRegisterTester }
  }
  "Intervals can be clipped with clip (saturate) operator" in {
    assertTesterPasses{ new IntervalClipTester }
  }
  "Intervals adds same answer as UInt" in {
    assertTesterPasses{ new IntervalChainedAddTester }
  }
  "Intervals should produce canonically smaller ranges via inference" in {
    val loFirrtl = makeLoFirrtl("low")(new Module {
      val io = IO(new Bundle {
        val in  = Input(Interval(range"[0,1]"))
        val out = Output(Interval(6.W))
      })

      val intervalResult = Wire(Interval())

      // intervalResult := 1.I * 1.I * 1.I * 1.I * 1.I * 1.I * 1.I
      intervalResult := 1.I + 1.I + 1.I + 1.I + 1.I + 1.I + 1.I
      io.out := intervalResult
    })
    println(s"LoFirrtl\n$loFirrtl")
    // why does this modify the port io width
    loFirrtl.contains("output io_out : SInt<6>") should be (true)

  }
  "Intervals muls same answer as UInt" in {
    assertTesterPasses{ new IntervalChainedMulTester }
  }
  "Intervals subs same answer as UInt" in {
    assertTesterPasses{ new IntervalChainedSubTester }
  }
  "Test clip, wrap and a variety of ranges" - {
    """range"[0.0,10.0].2" => range"[2,6].2""" in {
      assertTesterPasses( new BasicTester {

        val sourceRange = range"[0.0,10.0].2"
        val targetRange = range"[2,6].2"

        val sourceSimulator = ScalaIntervalSimulator(sourceRange)
        val targetSimulator = ScalaIntervalSimulator(targetRange)

        for (sourceValue <- sourceSimulator.allValues) {
          val clippedValue = Wire(Interval(targetRange))
          clippedValue := sourceSimulator.makeLit(sourceValue).clip(clippedValue)

          val goldClippedValue = targetSimulator.makeLit(targetSimulator.clip(sourceValue))

          // Useful for debugging
          // printf(s"source value $sourceValue clipped gold value %d compare to clipped value %d\n",
          //  goldClippedValue.asSInt(), clippedValue.asSInt())

          chisel3.assert(goldClippedValue === clippedValue)

          val wrappedValue = Wire(Interval(targetRange))
          wrappedValue := sourceSimulator.makeLit(sourceValue).wrap(wrappedValue)

          val goldWrappedValue = targetSimulator.makeLit(targetSimulator.wrap(sourceValue))

          // Useful for debugging
          // printf(s"source value $sourceValue wrapped gold value %d compare to wrapped value %d\n",
          //  goldWrappedValue.asSInt(), wrappedValue.asSInt())

          chisel3.assert(goldWrappedValue === wrappedValue)
        }

        stop()
      })
    }
  }

  "Test squeeze over a variety of ranges" - {
    """range"[2,6].2""" in {
      assertTesterPasses( new BasicTester {

        val sourceRange = range"[0.0,10.0].2"
        val targetRange = range"[2,6].3"

        val sourceSimulator = ScalaIntervalSimulator(sourceRange)
        val targetSimulator = ScalaIntervalSimulator(targetRange)

        for (sourceValue <- sourceSimulator.allValues) {
          val squeezedValue = Wire(Interval(targetRange))
          squeezedValue := sourceSimulator.makeLit(sourceValue).clip(squeezedValue)

          val goldSqueezedValue = targetSimulator.makeLit(targetSimulator.clip(sourceValue))

          // Useful for debugging
          // printf(s"source value $sourceValue squeezed gold value %d compare to squeezed value %d\n",
          //   goldSqueezedValue.asSInt(), squeezedValue.asSInt())

          chisel3.assert(goldSqueezedValue === squeezedValue)
        }

        stop()
      })
    }
  }
}

