// See LICENSE for license details.

package chiselTests

import scala.language.reflectiveCalls
import _root_.firrtl.ir.{Closed, Open}
import chisel3._
import chisel3.internal.firrtl.{IntervalRange, KnownBinaryPoint}
import chisel3.internal.sourceinfo.{SourceInfo, UnlocatableSourceInfo}
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.testers.BasicTester
import cookbook.CookbookTester
import firrtl.options.TargetDirAnnotation
import firrtl.passes.CheckTypes.InvalidConnect
import firrtl.passes.CheckWidths.{DisjointSqueeze, InvalidRange}
import firrtl.passes.{PassExceptions, WrapWithRemainder}
import firrtl.stage.{CompilerAnnotation, FirrtlCircuitAnnotation}
import firrtl.{FIRRTLException, HighFirrtlCompiler, LowFirrtlCompiler, MiddleFirrtlCompiler, MinimumVerilogCompiler, NoneCompiler, SystemVerilogCompiler, VerilogCompiler}
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

//scalastyle:off magic.number
//noinspection TypeAnnotation

object IntervalTestHelper {

  /** Compiles a Chisel Module to Verilog
    * NOTE: This uses the "test_run_dir" as the default directory for generated code.
    * @param compilerName the generator for the module
    * @param gen the generator for the module
    * @return the Verilog code as a string.
    */
  //scalastyle:off cyclomatic.complexity
  def makeFirrtl[T <: RawModule](compilerName: String)(gen: () => T): String = {
    (new ChiselStage)
      .execute(Array("--compiler", compilerName,
                     "--target-dir", "test_run_dir/IntervalSpec"),
               Seq(ChiselGeneratorAnnotation(gen)))
      .collectFirst { case FirrtlCircuitAnnotation(source) => source } match {
        case Some(circuit) => circuit.serialize
        case _ =>
          throw new Exception(
            s"makeFirrtl($compilerName) failed to generate firrtl circuit"
          )
      }

  }
}

import chiselTests.IntervalTestHelper.makeFirrtl
import chisel3.experimental._
import chisel3.experimental.Interval

class IntervalTest1 extends Module {
  val io = IO(new Bundle {
    val in1 = Input(Interval(range"[0,4]"))
    val in2 = Input(Interval(range"[0,4].3"))
    val out = Output(Interval(range"[0,8].3"))
  })

  io.out := io.in1 + io.in2
}

class IntervalTester extends CookbookTester(10) {

  val dut = Module(new IntervalTest1)

  dut.io.in1 := BigInt(4).I
  dut.io.in2 := 4.I
  assert(dut.io.out === 8.I)

  val i = Interval(range"[0,10)")
  stop()
}

class IntervalTest2 extends Module {
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

  5.U

  val result = in1 +& in2

  assert(result === 4.I)

  stop()

}

class IntervalSetBinaryPointTester extends BasicTester {
  implicit val sourceinfo: SourceInfo = UnlocatableSourceInfo
  val in1 = Wire(Interval(range"[0,4].4"))
  val in2 = in1.setPrecision(2)

  assert(in2.binaryPoint == KnownBinaryPoint(2))

  in1 := 2.I

  val shiftedLeft = in1.increasePrecision(2)

  assert(
    shiftedLeft.binaryPoint == KnownBinaryPoint(6),
    s"Error: increasePrecision result ${shiftedLeft.range} expected bt = 2"
  )

  val shiftedRight = in1.decreasePrecision(2)

  assert(
    shiftedRight.binaryPoint == KnownBinaryPoint(2),
    s"Error: increasePrecision result ${shiftedRight.range} expected bt = 2"
  )

  stop()
}

class MoreIntervalShiftTester extends BasicTester {
  implicit val sourceinfo: SourceInfo = UnlocatableSourceInfo

  val in1 = Wire(Interval(range"[0,4].4"))
  val in2 = in1.setPrecision(2)

  assert(in2.binaryPoint == KnownBinaryPoint(2))

  val toShiftLeft = Wire(Interval(range"[0,4].4"))
  val shiftedLeft = in1.increasePrecision(2)

  assert(
    shiftedLeft.binaryPoint == KnownBinaryPoint(2),
    s"Error: decreasePrecision result ${shiftedLeft.range} expected bt = 2"
  )

  val toShiftRight = Wire(Interval(range"[0,4].4"))
  val shiftedRight = in1.decreasePrecision(2)

  assert(
    shiftedRight.binaryPoint == KnownBinaryPoint(6),
    s"Error: decreasePrecision result ${shiftedRight.range} expected bt = 2"
  )

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
class ClipSqueezeWrapDemo(range: IntervalRange,
                          targetRange: IntervalRange,
                          startNum: Double,
                          endNum: Double,
                          incNum: Double)
    extends BasicTester {

  val binaryPointAsInt = range.binaryPoint.asInstanceOf[KnownBinaryPoint].value
//  val startValue = Interval.fromDouble(startNum, binaryPoint = binaryPointAsInt)
//  val increment = Interval.fromDouble(incNum, binaryPoint = binaryPointAsInt)
//  val endValue = Interval.fromDouble(endNum, binaryPoint = binaryPointAsInt)
  val startValue = startNum.I(range.binaryPoint)
  val increment = incNum.I(range.binaryPoint)
  val endValue = endNum.I(range.binaryPoint)

  val counter = RegInit(Interval(range), startValue)

  counter := (counter + increment).squeeze(counter)
  when(counter > endValue) {
    stop()
  }

  val clipped = counter.clip(0.U.asInterval(targetRange))
  val squeezed = counter.squeeze(0.U.asInterval(targetRange))
  val wrapped = counter.wrap(0.U.asInterval(targetRange))

  when(counter === startValue) {
    printf(s"Target range is $range\n")
    printf("value     clip      squeeze      wrap\n")
  }

  printf(
    "       %d       %d          %d         %d\n",
    counter.asSInt(),
    clipped.asSInt(),
    squeezed.asSInt(),
    wrapped.asSInt()
  )
}

class SqueezeFunctionalityTester(range: IntervalRange,
                                 startNum: BigDecimal,
                                 endNum: BigDecimal,
                                 increment: BigDecimal)
    extends BasicTester {

  val counter = RegInit(0.U(10.W))
  counter := counter + 1.U
  when(counter > 10.U) {
    stop()
  }

  val squeezeInterval = Wire(Interval(range))
  squeezeInterval := 0.I

  val squeezeTemplate = Wire(Interval(range))

  val ss = WireInit(Interval(range), (-10).S.asInterval(range))

  val toSqueeze = counter.asInterval(range) - ss

  squeezeTemplate := toSqueeze.squeeze(squeezeInterval)

  printf(
    s"SqueezeTest %d    %d.squeeze($range) => %d\n",
    counter,
    toSqueeze.asSInt(),
    squeezeTemplate.asSInt()
  )
}

/**
  * Demonstrate a simple counter register with an Interval type
  */
class IntervalRegisterTester extends BasicTester {

  val range = range"[-2,5]"
  val counter = RegInit(Interval(range), (-1).I)
  counter := (counter + 1.I)
    .squeeze(counter) // this works with other types, why not Interval
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

  assert(
    t2.range.upper == Closed(7),
    s"t1 upper ${t2.range.upper} expected ${Closed(7)}"
  )
  assert(
    t3.range.upper == Closed(7),
    s"t1 upper ${t3.range.upper} expected ${Closed(7)}"
  )

  val in1 = WireInit(Interval(range"[0,9].6"), 0.I)
  val in2 = WireInit(Interval(range"[1,6).4"), 2.I)
  val in3 = in1.wrap(in2)

  assert(
    in3.range.lower == Closed(1),
    s"in3 lower ${in3.range.lower} expected ${Closed(1)}"
  )
  assert(
    in3.range.upper == Open(6),
    s"in3 upper ${in3.range.upper} expected ${Open(6)}"
  )
  assert(
    in3.binaryPoint == KnownBinaryPoint(6),
    s"in3 binaryPoint ${in3.binaryPoint} expected ${KnownBinaryPoint(2)}"
  )

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
  fixedResult := 17.0.F(0.BP) -& 2.0.F(0.BP) -& 2.0.F(0.BP) -& 2.0.F(0.BP) -& 2.0
    .F(0.BP) -& 2.0.F(0.BP) -& 2.0.F(0.BP)
  sIntResult := 17.S -& 2.S -& 2.S -& 2.S -& 2.S -& 2.S -& 2.S

  assert(uIntResult === 5.U)
  assert(sIntResult === 5.S)
  assert(fixedResult.asUInt === 5.U)
  assert(intervalResult1 === 5.I)
  assert(intervalResult2 === 5.I)

  stop()
}

//TODO: need tests for dynamic shifts on intervals
class IntervalSpec extends AnyFreeSpec with Matchers with ChiselRunners {

  type TempFirrtlException = Exception

  "Test a simple interval add" in {
    assertTesterPasses { new IntervalAddTester }
  }
  "Intervals can be created" in {
    assertTesterPasses { new IntervalTester }
  }
  "Test a simple interval mux" in {
    assertTesterPasses { new IntervalTester2 }
  }
  "Intervals can have binary points set" in {
    assertTesterPasses { new IntervalSetBinaryPointTester }
  }
  "Interval literals that don't fit in explicit ranges are caught by chisel" - {
    "case 1: does not fit in specified width" in {
      intercept[ChiselException] {
        ChiselGeneratorAnnotation(
          () =>
            new BasicTester {
              val x = 5.I(3.W, 0.BP)
          }
        ).elaborate
      }
    }
    "case 2: doesn't fit in specified range" in {
      intercept[ChiselException] {
        ChiselGeneratorAnnotation(
          () =>
            new BasicTester {
              val x = 5.I(range"[0,4]")
          }
        ).elaborate
      }
    }
  }

  "Interval literals support to double and to BigDecimal" in {
    val d = -7.125
    val lit1 = d.I(3.BP)
    lit1.litToDouble should be (d)

    val d2 = BigDecimal("1232123213131123.125")
    val lit2 = d2.I(3.BP)
    lit2.litToBigDecimal should be (d2)

    // Numbers that are too big will throw exception
    intercept[ChiselException] {
      lit2.litToDouble
    }
  }

  "Interval literals creation handles edge cases" - {
    "value at closed boundaries works" in {
      val inputRange = range"[-6, 6].2"
      val in1 = (-6.0).I(inputRange)
      val in2 = 6.0.I(inputRange)
      BigDecimal(in1.litValue()) / (1 << inputRange.binaryPoint.get) should be (-6)
      BigDecimal(in2.litValue()) / (1 << inputRange.binaryPoint.get) should be (6)
      intercept[ChiselException] {
        (-6.25).I(inputRange)
      }
      intercept[ChiselException] {
        (6.25).I(inputRange)
      }
    }
    "value at open boundaries works" in {
      val inputRange = range"(-6, 6).2"
      val in1 = (-5.75).I(inputRange)
      val in2 = 5.75.I(inputRange)
      BigDecimal(in1.litValue()) / (1 << inputRange.binaryPoint.get) should be (-5.75)
      BigDecimal(in2.litValue()) / (1 << inputRange.binaryPoint.get) should be (5.75)
      intercept[ChiselException] {
        (-6.0).I(inputRange)
      }
      intercept[ChiselException] {
        (6.0).I(inputRange)
      }
    }
    "values not precisely at open boundaries works but are converted to nearest match" in {
      val inputRange = range"(-6, 6).2"
      val in1 = (-5.95).I(inputRange)
      val in2 = 5.95.I(inputRange)
      BigDecimal(in1.litValue()) / (1 << inputRange.binaryPoint.get) should be (-5.75)
      BigDecimal(in2.litValue()) / (1 << inputRange.binaryPoint.get) should be (5.75)
      intercept[ChiselException] {
        (-6.1).I(inputRange)
      }
      intercept[ChiselException] {
        (6.1).I(inputRange)
      }
    }
  }

  "Let's take a look at the results of squeeze over small range" in {
    assertTesterPasses {
      new ClipSqueezeWrapDemo(
        range = range"[-10,33].0",
        targetRange = range"[-4,17].0",
        startNum = -4.0,
        endNum = 30.0,
        incNum = 1.0
      )
    }
    assertTesterPasses {
      new ClipSqueezeWrapDemo(
        range = range"[-2,5].1",
        targetRange = range"[-1,3].1",
        startNum = -2.0,
        endNum = 5.0,
        incNum = 0.5
      )
    }
  }
  "Intervals can be squeezed into another intervals range" in {
    assertTesterPasses {
      new SqueezeFunctionalityTester(
        range"[-2,5]",
        BigDecimal(-10),
        BigDecimal(10),
        BigDecimal(1.0)
      )
    }
  }
  "Intervals can be wrapped with wrap operator" in {
    assertTesterPasses { new IntervalWrapTester }
  }

  "Interval compile pathologies: clip, wrap, and squeeze have different behavior" - {
    "wrap target range is completely left of source" in {
      intercept[TempFirrtlException] {
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
      intercept[TempFirrtlException] {
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
      assertTesterPasses(new BasicTester {
        val base = Wire(Interval(range"[-4, 6]"))
        base := 6.I
        val disjointLeft = WireInit(Interval(range"[-7,-5]"), (-6).I)
        val w5 = base.clip(disjointLeft)
        chisel3.assert(w5 === (-5).I)
        stop()
      })
    }
    "clip target range is completely right of source" in {
      assertTesterPasses(new BasicTester {
        val base = Wire(Interval(range"[-4, 6]"))
        base := 6.I
        val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
        val w5 = base.clip(disjointLeft)
        chisel3.assert(w5.asSInt === 7.S)
        stop()
      })
    }
    "squeeze target range is completely right of source" in {
      intercept[TempFirrtlException] {
        assertTesterPasses(new BasicTester {
          val base = Wire(Interval(range"[-4, 6]"))
          base := 6.I
          val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
          val w5 = base.squeeze(disjointLeft)
          chisel3.assert(w5.asSInt === 6.S)
          stop()
        })
      }
    }
    "squeeze target range is completely left of source" in {
      intercept[TempFirrtlException] {
        assertTesterPasses(new BasicTester {
          val base = Wire(Interval(range"[-4, 6]"))
          base := 6.I
          val disjointLeft = WireInit(Interval(range"[-7, -5]"), 8.I)
          val w5 = base.squeeze(disjointLeft)
          stop()
        })
      }
    }

    def makeCircuit(operation: String,
                    sourceRange: IntervalRange,
                    targetRange: IntervalRange): () => RawModule = { () =>
      new Module {
        val io = IO(new Bundle { val out = Output(Interval()) })
        val base = Wire(Interval(sourceRange))
        base := 6.I

        val disjointLeft = WireInit(Interval(targetRange), 8.I)
        val w5 = operation match {
          case "clip"    => base.clip(disjointLeft)
          case "wrap"    => base.wrap(disjointLeft)
          case "squeeze" => base.squeeze(disjointLeft)
        }
        io.out := w5
      }
    }

    "disjoint ranges should error when used with clip, wrap and squeeze" - {

      def mustGetException(disjointLeft: Boolean,
                           operation: String): Boolean = {
        val (rangeA, rangeB) = if (disjointLeft) {
          (range"[-4, 6]", range"[7,10]")
        } else {
          (range"[7,10]", range"[-4, 6]")
        }
        try {
          makeFirrtl("low")(makeCircuit(operation, rangeA, rangeB))
          false
        } catch {
          case _: InvalidConnect | _: PassExceptions | _: InvalidRange | _: WrapWithRemainder | _: DisjointSqueeze =>
            true
          case _: Throwable =>
            false
        }
      }

      "Range A disjoint left, operation clip should generate useful error" in {
        mustGetException(disjointLeft = true, "clip") should be(false)
      }
      "Range A largely out of bounds left, operation wrap should generate useful error" in {
        mustGetException(disjointLeft = true, "wrap") should be(true)
      }
      "Range A disjoint left, operation squeeze should generate useful error" in {
        mustGetException(disjointLeft = true, "squeeze") should be(true)
      }
      "Range A disjoint right, operation clip should generate useful error" in {
        mustGetException(disjointLeft = false, "clip") should be(true)
      }
      "Range A disjoint right, operation wrap should generate useful error" in {
        mustGetException(disjointLeft = false, "wrap") should be(true)
      }
      "Range A disjoint right, operation squeeze should generate useful error" in {
        mustGetException(disjointLeft = false, "squeeze") should be(true)
      }
    }

    "Errors are sometimes inconsistent or incorrectly labelled as Firrtl Internal Error" - {
      "squeeze disjoint is not internal error when defined in BasicTester" in {
        intercept[DisjointSqueeze] {
          makeFirrtl("low")(
            () =>
              new BasicTester {
                val base = Wire(Interval(range"[-4, 6]"))
                val base2 = Wire(Interval(range"[-4, 6]"))
                base := 6.I
                base2 := 5.I
                val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
                val w5 = base.squeeze(disjointLeft)
                stop()
            }
          )
        }
      }
      "wrap disjoint is not internal error when defined in BasicTester" in {
        intercept[DisjointSqueeze] {
          makeFirrtl("low")(
            () =>
              new BasicTester {
                val base = Wire(Interval(range"[-4, 6]"))
                val base2 = Wire(Interval(range"[-4, 6]"))
                base := 6.I
                base2 := 5.I
                val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
                val w5 = base.squeeze(disjointLeft)
                stop()
            }
          )
        }
      }
      "squeeze disjoint from Module gives exception" in {
        intercept[DisjointSqueeze] {
          makeFirrtl("low")(
            () =>
              new Module {
                val io = IO(new Bundle {
                  val out = Output(Interval())
                })
                val base = Wire(Interval(range"[-4, 6]"))
                base := 6.I

                val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
                val w5 = base.squeeze(disjointLeft)
                io.out := w5
            }
          )
        }
      }
      "clip disjoint from Module gives no error" in {
        makeFirrtl("low")(
          () =>
            new Module {
              val io = IO(new Bundle {
                val out = Output(Interval())
              })
              val base = Wire(Interval(range"[-4, 6]"))
              base := 6.I

              val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
              val w5 = base.clip(disjointLeft)
              io.out := w5
          }
        )
      }
      "wrap disjoint from Module wrap with remainder" in {
        intercept[WrapWithRemainder] {
          makeFirrtl("low")(
            () =>
              new Module {
                val io = IO(new Bundle {
                  val out = Output(Interval())
                })
                val base = Wire(Interval(range"[-4, 6]"))
                base := 6.I

                val disjointLeft = WireInit(Interval(range"[7,10]"), 8.I)
                val w5 = base.wrap(disjointLeft)
                io.out := w5
            }
          )
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
      intercept[InvalidConnect] {
        makeFirrtl("low")(
          () =>
            new Module {
              val io = IO(new Bundle { val out = Output(Interval()) })
              val base = Wire(Interval(range"[-4, 6]"))
              base := (-7).I
              io.out := base
          }
        )
      }
    }
    "when literal is too big" in {
      intercept[InvalidConnect] {
        makeFirrtl("low")(
          () =>
            new Module {
              val io = IO(new Bundle { val out = Output(Interval()) })
              val base = Wire(Interval(range"[-4, 6]"))
              base := 9.I
              io.out := base
          }
        )
      }
    }
  }

  "Intervals can be shifted left" in {
    assertTesterPasses(new BasicTester {
      val i1 = 3.0.I(range"[0,4]")
      val shifted1 = i1 << 2
      val shiftUInt = WireInit(1.U(8.W))
      val shifted2 = i1 << shiftUInt

      chisel3.assert(shifted1 === 12.I, "shifted 1 should be 12, it wasn't")
      chisel3.assert(shifted2 === 6.I, "shifted 2 should be 6 it wasn't")
      stop()
    })
  }

  "Intervals can be shifted right" in {
    assertTesterPasses(new BasicTester {
      val i1 = 12.0.I(range"[0,15]")
      val shifted1 = i1 >> 2
      val shiftUInt = 1.U
      val shifted2 = i1 >> shiftUInt

      chisel3.assert(shifted1 === 3.I)
      chisel3.assert(shifted2 === 6.I)
      stop()
    })
  }

  "Intervals can be used to construct registers" in {
    assertTesterPasses { new IntervalRegisterTester }
  }
  "Intervals can be clipped with clip (saturate) operator" in {
    assertTesterPasses { new IntervalClipTester }
  }
  "Intervals adds same answer as UInt" in {
    assertTesterPasses { new IntervalChainedAddTester }
  }
  "Intervals should produce canonically smaller ranges via inference" in {
    val loFirrtl = makeFirrtl("low")(
      () =>
        new Module {
          val io = IO(new Bundle {
            val in = Input(Interval(range"[0,1]"))
            val out = Output(Interval())
          })

          val intervalResult = Wire(Interval())

          intervalResult := 1.I + 1.I + 1.I + 1.I + 1.I + 1.I + 1.I
          io.out := intervalResult
      }
    )
    loFirrtl.contains("output io_out : SInt<4>") should be(true)

  }
  "Intervals multiplication same answer as UInt" in {
    assertTesterPasses { new IntervalChainedMulTester }
  }
  "Intervals subs same answer as UInt" in {
    assertTesterPasses { new IntervalChainedSubTester }
  }
  "Test clip, wrap and a variety of ranges" - {
    """range"[0.0,10.0].2" => range"[2,6].2"""" in {
      assertTesterPasses(new BasicTester {

        val sourceRange = range"[0.0,10.0].2"
        val targetRange = range"[2,6].2"

        val sourceSimulator = ScalaIntervalSimulator(sourceRange)
        val targetSimulator = ScalaIntervalSimulator(targetRange)

        for (sourceValue <- sourceSimulator.allValues) {
          val clippedValue = Wire(Interval(targetRange))
          clippedValue := sourceSimulator
            .makeLit(sourceValue)
            .clip(clippedValue)

          val goldClippedValue =
            targetSimulator.makeLit(targetSimulator.clip(sourceValue))

          // Useful for debugging
          // printf(s"source value $sourceValue clipped gold value %d compare to clipped value %d\n",
          //  goldClippedValue.asSInt(), clippedValue.asSInt())

          chisel3.assert(goldClippedValue === clippedValue)

          val wrappedValue = Wire(Interval(targetRange))
          wrappedValue := sourceSimulator
            .makeLit(sourceValue)
            .wrap(wrappedValue)

          val goldWrappedValue =
            targetSimulator.makeLit(targetSimulator.wrap(sourceValue))

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
      assertTesterPasses(new BasicTester {

        val sourceRange = range"[0.0,10.0].2"
        val targetRange = range"[2,6].3"

        val sourceSimulator = ScalaIntervalSimulator(sourceRange)
        val targetSimulator = ScalaIntervalSimulator(targetRange)

        for (sourceValue <- sourceSimulator.allValues) {
          val squeezedValue = Wire(Interval(targetRange))
          squeezedValue := sourceSimulator
            .makeLit(sourceValue)
            .clip(squeezedValue)

          val goldSqueezedValue =
            targetSimulator.makeLit(targetSimulator.clip(sourceValue))

          // Useful for debugging
          // printf(s"source value $sourceValue squeezed gold value %d compare to squeezed value %d\n",
          //   goldSqueezedValue.asSInt(), squeezedValue.asSInt())

          chisel3.assert(goldSqueezedValue === squeezedValue)
        }

        stop()
      })
    }
  }

  "test asInterval" - {
    "use with UInt" in {
      assertTesterPasses(new BasicTester {
        val u1 = Wire(UInt(5.W))
        u1 := 7.U
        val i1 = u1.asInterval(range"[0,15]")
        val i2 = u1.asInterval(range"[0,15].2")
        printf("i1 %d\n", i1.asUInt)
        chisel3.assert(i1 === 7.I, "i1")
        stop()
      })
    }
    "use with SInt" in {
      assertTesterPasses(new BasicTester {
        val s1 = Wire(SInt(5.W))
        s1 := 7.S
        val s2 = Wire(SInt(5.W))
        s2 := 7.S
        val i1 = s1.asInterval(range"[-16,15]")
        val i2 = s1.asInterval(range"[-16,15].1")
        printf("i1 %d\n", i1.asSInt)
        printf("i2 %d\n", i2.asSInt)
        chisel3.assert(i1 === 7.I, "i1 is wrong")
        chisel3.assert(i2 === (3.5).I(binaryPoint = 1.BP), "i2 is wrong")
        stop()
      })
    }
    "more SInt tests" in {
      assertTesterPasses(new BasicTester {
        chisel3.assert(7.S.asInterval(range"[-16,15].1") === 3.5.I(binaryPoint = 1.BP), "adding binary point")
        stop()
      })
    }
  }
}
