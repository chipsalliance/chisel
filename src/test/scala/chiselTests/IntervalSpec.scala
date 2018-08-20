// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{ChiselRange, Interval}
import chisel3.internal.firrtl.{IntervalRange, KnownBinaryPoint}
import chisel3.internal.sourceinfo.{SourceInfo, UnlocatableSourceInfo}
import chisel3.testers.BasicTester
import cookbook.CookbookTester
import logger.LogLevel
import _root_.firrtl.ir.{Closed, Open, UnknownBound}

import org.scalatest.{FreeSpec, Matchers}

//scalastyle:off magic.number

class IntervalTest1 extends Module {
  val io = IO(new Bundle {
    val in1 = Input(Interval(6.W, range"[0,4].3"))
    val in2 = Input(Interval(6.W, range"[0,4].3"))
    val out = Output(Interval(8.W, range"[0,8].3"))
  })

  io.out := io.in1 + io.in2
}

class IntervalTester extends CookbookTester(10) {
  //logger.Logger.setLevel(LogLevel.Info)
  val dut = Module(new IntervalTest1)

  dut.io.in1 := 4.I()
  dut.io.in2 := 4.I()
  printf("dut.io.out: %b\n", dut.io.out.asUInt)
  assert(dut.io.out === 8.I())

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
  //logger.Logger.setLevel(LogLevel.Info)
  val dut = Module(new IntervalTest2)

  dut.io.p := 1.U
  dut.io.in1 := 4.I()
  dut.io.in2 := 5.I()
  printf("dut.io.out: %b\n", dut.io.out.asUInt)
  assert(dut.io.out === 4.I())

  stop()
}


class SIntTest1 extends Module {
  val io = IO(new Bundle {
    val in1 = Input(SInt(6.W))
    val in2 = Input(SInt(6.W))
    val out = Output(SInt(8.W))
  })

  io.out := io.in1 + io.in2
}
class SIntTest1Tester extends CookbookTester(10) {
  val dut = Module(new SIntTest1)

  dut.io.in1 := 4.S
  dut.io.in2 := 4.S
  assert(dut.io.out === 8.S)

  val i = SInt(range"[0,10)")
  stop()
}

class IntervalAddTester extends BasicTester {
  //logger.Logger.setLevel(LogLevel.Info)

  val in1 = Wire(Interval(range"[0,4]"))
  val in2 = Wire(Interval(range"[0,4]"))
  val in3 = Wire(Interval(range"[?,?]"))

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

  val toShiftLeft = Wire(Interval(range"[0,4].4"))
  val shiftedLeft = in1.shiftLeftBinaryPoint(2)

  assert(shiftedLeft.binaryPoint == KnownBinaryPoint(6), s"Error: bpshl result ${shiftedLeft.range} expected bt = 2")

  val toShiftRight = Wire(Interval(range"[0,4].4"))
  val shiftedRight = in1.shiftRightBinaryPoint(2)

  assert(shiftedRight.binaryPoint == KnownBinaryPoint(2), s"Error: bpshl result ${shiftedRight.range} expected bt = 2")

  stop()
}

class MoreIntervalShiftTester extends BasicTester {
  implicit val sourceinfo: SourceInfo = UnlocatableSourceInfo

  for {
    rangeMin <- 0 to 0
  } {

  }
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


class IntervalWrapTester extends BasicTester {
  implicit val sourceinfo: SourceInfo = UnlocatableSourceInfo

  val t1 = Wire(Interval(range"[-20, 19]"))
  val u1 = Wire(UInt(3.W))
  val r1 = Reg(UInt())
  r1 := u1
  val t2 = t1.wrap(u1)
  val t3 = t1.wrap(r1)

  assert(t2.range.upper == Closed(7), s"t1 upper ${t2.range.upper} expected ${Closed(7)}")
  assert(t3.range.upper == UnknownBound, s"t1 upper ${t3.range.upper} expected $UnknownBound")

  val in1 = Wire(Interval(range"[0,15].6"))
  val in2 = Wire(Interval(range"[1,6).4"))
  val in3 = in1.wrap(in2)

  //  assert(in3.range.lower == Closed(1), s"in3 lower ${in3.range.lower} expected ${Closed(1)}")
  assert(in3.range.lower == UnknownBound, s"in3 lower ${in3.range.lower} expected ${Closed(1)}")
  assert(in3.range.upper == UnknownBound, s"in3 upper ${in3.range.upper} expected ${Open(6)}")
  assert(in3.binaryPoint == KnownBinaryPoint(6), s"in3 binaryPoint ${in3.binaryPoint} expected ${KnownBinaryPoint(2)}")

  val enclosedRange = range"[-2, 5]"
  val base = Wire(Interval(range"[-4, 6]"))
  val enclosed = Wire(Interval(enclosedRange))
  val enclosing = Wire(Interval(range"[-6, 8]"))
  val overlapLeft = Wire(Interval(range"[-10,-2]"))
  val overlapRight = Wire(Interval(range"[-1,10]"))
  val disjointLeft = Wire(Interval(range"[-14,-7]"))
  val disjointRight = Wire(Interval(range"[7,11]"))

  val w1 = base.wrap(enclosed)
  val w2 = base.wrap(enclosing)
  val w3 = base.wrap(overlapLeft)
  val w4 = base.wrap(overlapRight)
  val w5 = base.wrap(disjointLeft)
  val w6 = base.wrap(disjointRight)
  val w7 = base.wrap(enclosedRange)

  base := 6.I()

  assert(w1 === (-2).I())
  assert(w2 === 6.I())
  assert(w3 === (-3).I())
  assert(w4 === 6.I())
  assert(w5 === (-8).I())
  //TODO (chick, adam) Why is this not 10.I
  // assert(w6 === 10.I())
  printf("w6 is %d\n", w6.asSInt())

  assert(w7 === (-2).I())

  //TODO (chick, adam) Why do these print out as positive numbers
  printf("w1 is %d\n", w1.asSInt())
  printf("enclosedViaRangeString is %d\n", w7.asSInt())

  println(s"enclosed ${w1.range.lower} ${w1.range.upper}")
  println(s"enclosing ${w2.range.lower} ${w2.range.upper}")
  println(s"overlapLeft ${w3.range.lower} ${w3.range.upper}")
  println(s"overlapRight ${w4.range.lower} ${w4.range.upper}")
  println(s"disjointLeft ${w5.range.lower} ${w5.range.upper}")
  println(s"disjointRight ${w6.range.lower} ${w6.range.upper}")
  println(s"enclosed from string ${w7.range.lower} ${w7.range.upper}")

  stop()
}

class IntervalClipTester extends BasicTester {
  implicit val sourceinfo: SourceInfo = UnlocatableSourceInfo

  val enclosedRange = range"[-2, 5]"
  val base = Wire(Interval(range"[-4, 6]"))
  val enclosed = Wire(Interval(enclosedRange))
  val enclosing = Wire(Interval(range"[-6, 8]"))
  val overlapLeft = Wire(Interval(range"[-10,-2]"))
  val overlapRight = Wire(Interval(range"[-1,10]"))
  val disjointLeft = Wire(Interval(range"[-14,-7]"))
  val disjointRight = Wire(Interval(range"[7,11]"))

  val enclosedResult = base.clip(enclosed)
  val enclosingResult = base.clip(enclosing)
  val overlapLeftResult = base.clip(overlapLeft)
  val overlapRightResult = base.clip(overlapRight)
  val disjointLeftResult = base.clip(disjointLeft)
  val disjointRightResult = base.clip(disjointRight)
  val enclosedViaRangeString = base.clip(enclosedRange)

  base := 6.I()

  assert(enclosedResult === 5.I())
  assert(enclosingResult === 6.I())
  assert(overlapLeftResult === (-2).I())
  assert(overlapRightResult === 6.I())
  assert(disjointLeftResult === (-7).I())
  assert(disjointRightResult === 7.I())

  assert(enclosedViaRangeString === 5.I())

  println(s"enclosed ${enclosedResult.range.lower} ${enclosedResult.range.upper}")
  println(s"enclosing ${enclosingResult.range.lower} ${enclosingResult.range.upper}")
  println(s"overlapLeft ${overlapLeftResult.range.lower} ${overlapLeftResult.range.upper}")
  println(s"overlapRight ${overlapRightResult.range.lower} ${overlapRightResult.range.upper}")
  println(s"disjointLeft ${disjointLeftResult.range.lower} ${disjointLeftResult.range.upper}")
  println(s"disjointRight ${disjointRightResult.range.lower} ${disjointRightResult.range.upper}")
  println(s"enclosedViaRangeString ${enclosedViaRangeString.range.lower} ${enclosedViaRangeString.range.upper}")

  stop()
}

class IntervalChainedAddTester extends BasicTester {
  //logger.Logger.setLevel(LogLevel.Info)

  val intervalResult = Wire(Interval())
  val uintResult = Wire(UInt())

  intervalResult := 2.I + 2.I + 2.I + 2.I + 2.I + 2.I + 2.I
  uintResult := 2.U +& 2.U +& 2.U +& 2.U +& 2.U +& 2.U +& 2.U

  printf("Interval result: %d\n", intervalResult.asUInt)
  assert(intervalResult === 14.I)
  printf("UInt result: %d\n", uintResult)
  assert(uintResult === 14.U)
  stop()
}

class IntervalChainedMulTester extends BasicTester {
  //logger.Logger.setLevel(LogLevel.Info)

  val intervalResult = Wire(Interval())
  val uintResult = Wire(UInt())

  intervalResult := 2.I * 2.I * 2.I * 2.I * 2.I * 2.I * 2.I
  uintResult := 2.U * 2.U * 2.U * 2.U * 2.U * 2.U * 2.U

  printf("Interval result: %d\n", intervalResult.asUInt)
  printf("UInt result: %d\n", uintResult)
  assert(intervalResult === 128.I)
  assert(uintResult === 128.U)
  stop()
}

class IntervalChainedSubTester extends BasicTester {
  //logger.Logger.setLevel(LogLevel.Info)

  val intervalResult = Wire(Interval())
  val uintResult = Wire(UInt())

  intervalResult := 17.I - 2.I - 2.I - 2.I - 2.I - 2.I - 2.I
  uintResult := 17.U -& 2.U -& 2.U -& 2.U -& 2.U -& 2.U -& 2.U

  printf("Interval result: %d\n", intervalResult.asUInt)
  printf("UInt result: %d\n", uintResult)
  assert(intervalResult === 5.I)
  assert(uintResult === 5.U)
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
  "SInt for use comparing to Interval" in {
    assertTesterPasses{ new SIntTest1Tester }
  }
  "Intervals can have binary points set" in {
    assertTesterPasses{ new IntervalSetBinaryPointTester }
  }
  "Intervals can be wrapped with wrap operator" in {
    assertTesterPasses{ new IntervalWrapTester }
  }
  "Intervals can be clipped with clip (saturate) operator" in {
    assertTesterPasses{ new IntervalClipTester }
  }
  "Intervals adds same answer as UInt" in {
    assertTesterPasses{ new IntervalChainedAddTester }
  }
  "Intervals muls same answer as UInt" in {
    assertTesterPasses{ new IntervalChainedMulTester }
  }
  "Intervals subs same answer as UInt" in {
    assertTesterPasses{ new IntervalChainedSubTester }
  }
  "show the firrtl" in {
    Driver.execute(Array("--no-run-firrtl"), () => new IntervalChainedAddTester) match {
      case result: ChiselExecutionSuccess =>
        println(result.emitted)
      case _ =>
        assert(false, "Failed to generate chirrtl")
    }
  }
}
