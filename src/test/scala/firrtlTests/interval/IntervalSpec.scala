package firrtlTests
package interval

import java.io._

import firrtl._
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.Parser.IgnoreInfo
import firrtl.passes.CheckTypes.InvalidConnect
import firrtl.passes.CheckWidths.DisjointSqueeze

class IntervalSpec extends FirrtlFlatSpec {
  private def executeTest(input: String, expected: Seq[String], passes: Seq[Transform]) = {
    val c = passes.foldLeft(Parser.parse(input.split("\n").toIterator)) {
      (c: Circuit, p: Transform) => 
        p.runTransform(CircuitState(c, UnknownForm, AnnotationSeq(Nil), None)).circuit
    }
    val lines = c.serialize.split("\n") map normalized

    expected foreach { e =>
      lines should contain(e)
    }
  }

  "Interval types" should "parse correctly" in {
    val passes = Seq(ToWorkingIR)
    val input =
      """circuit Unit :
        |  module Unit :
        |    input in0 : Interval(-0.32, 10.1).4
        |    input in1 : Interval[0, 10.1].4
        |    input in2 : Interval(-0.32, 10].4
        |    input in3 : Interval[-3, 10.1).4
        |    input in4 : Interval(-0.32, 10.1)
        |    input in5 : Interval.4
        |    input in6 : Interval
        |    output out0 : Interval.2
        |    output out1 : Interval
        |    out0 <= add(in0, add(in1, add(in2, add(in3, add(in4, add(in5, in6))))))
        |    out1 <= add(in0, add(in1, add(in2, add(in3, add(in4, add(in5, in6))))))""".stripMargin
    executeTest(input, input.split("\n") map normalized, passes)
  }

  "Interval types" should "infer bp correctly" in {
    val passes = Seq(ToWorkingIR, InferTypes, ResolveGenders, new InferBinaryPoints())
    val input =
      """circuit Unit :
        |  module Unit :
        |    input in0 : Interval(-0.32, 10.1).4
        |    input in1 : Interval[0, 10.1].3
        |    input in2 : Interval(-0.32, 10].2
        |    output out0 : Interval
        |    out0 <= add(in0, add(in1, in2))""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input in0 : Interval(-0.32, 10.1).4
        |    input in1 : Interval[0, 10.1].3
        |    input in2 : Interval(-0.32, 10].2
        |    output out0 : Interval.4
        |    out0 <= add(in0, add(in1, in2))""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Interval types" should "trim known intervals correctly" in {
    val passes = Seq(ToWorkingIR, InferTypes, ResolveGenders, new InferBinaryPoints(), new TrimIntervals())
    val input =
      """circuit Unit :
        |  module Unit :
        |    input in0 : Interval(-0.32, 10.1).4
        |    input in1 : Interval[0, 10.1].3
        |    input in2 : Interval(-0.32, 10].2
        |    output out0 : Interval
        |    out0 <= add(in0, add(in1, in2))""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input in0 : Interval[-0.3125, 10.0625].4
        |    input in1 : Interval[0, 10].3
        |    input in2 : Interval[-0.25, 10].2
        |    output out0 : Interval.4
        |    out0 <= add(in0, incp(add(in1, incp(in2, 1)), 1))""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Interval types" should "infer intervals correctly" in {
    val passes = Seq(ToWorkingIR, InferTypes, ResolveGenders, new InferBinaryPoints(), new TrimIntervals(), new InferWidths())
    val input =
      """circuit Unit :
        |  module Unit :
        |    input in0 : Interval(0, 10).4
        |    input in1 : Interval(0, 10].3
        |    input in2 : Interval(-1, 3].2
        |    output out0 : Interval
        |    output out1 : Interval
        |    output out2 : Interval
        |    out0 <= add(in0, add(in1, in2))
        |    out1 <= mul(in0, mul(in1, in2))
        |    out2 <= sub(in0, sub(in1, in2))""".stripMargin
    val check =
      """output out0 : Interval[-0.5625, 22.9375].4
        |output out1 : Interval[-74.53125, 298.125].9
        |output out2 : Interval[-10.6875, 12.8125].4""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

  "Interval types" should "be removed correctly" in {
    val passes = Seq(ToWorkingIR, InferTypes, ResolveGenders, new InferBinaryPoints(), new TrimIntervals(), new InferWidths(), new RemoveIntervals())
    val input =
      """circuit Unit :
        |  module Unit :
        |    input in0 : Interval(0, 10).4
        |    input in1 : Interval(0, 10].3
        |    input in2 : Interval(-1, 3].2
        |    output out0 : Interval
        |    output out1 : Interval
        |    output out2 : Interval
        |    out0 <= add(in0, add(in1, in2))
        |    out1 <= mul(in0, mul(in1, in2))
        |    out2 <= sub(in0, sub(in1, in2))""".stripMargin
    val check =
      """circuit Unit :
        |  module Unit :
        |    input in0 : SInt<9>
        |    input in1 : SInt<8>
        |    input in2 : SInt<5>
        |    output out0 : SInt<10>
        |    output out1 : SInt<19>
        |    output out2 : SInt<9>
        |    out0 <= add(in0, shl(add(in1, shl(in2, 1)), 1))
        |    out1 <= mul(in0, mul(in1, in2))
        |    out2 <= sub(in0, shl(sub(in1, shl(in2, 1)), 1))""".stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }

"Interval types" should "infer multiplication by zero correctly" in {
  val passes = Seq(ToWorkingIR, InferTypes, ResolveGenders, new InferBinaryPoints(), new TrimIntervals(), new InferWidths())
    val input =
      s"""circuit Unit :
      |  module Unit :
      |    input  in1 : Interval[0, 0.5].1
      |    input  in2 : Interval[0, 0].1
      |    output mul : Interval
      |    mul <= mul(in2, in1)
      |    """.stripMargin
  val check = s"""output mul : Interval[0, 0].2 """.stripMargin
  executeTest(input, check.split("\n") map normalized, passes)
}

  "Interval types" should "infer muxes correctly" in {
    val passes = Seq(ToWorkingIR, InferTypes, ResolveGenders, new InferBinaryPoints(), new TrimIntervals(), new InferWidths())
      val input =
        s"""circuit Unit :
        |  module Unit :
        |    input  p   : UInt<1>
        |    input  in1 : Interval[0, 0.5].1
        |    input  in2 : Interval[0, 0].1
        |    output out : Interval
        |    out <= mux(p, in2, in1)
        |    """.stripMargin
    val check = s"""output out : Interval[0, 0.5].1 """.stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Interval types" should "infer dshl correctly" in {
    val passes = Seq(ToWorkingIR, InferTypes, ResolveKinds, ResolveGenders, new InferBinaryPoints(), new TrimIntervals, new InferWidths())
      val input =
        s"""circuit Unit :
        |  module Unit :
        |    input  p   : UInt<3>
        |    input  in1 : Interval[-1, 1].0
        |    output out : Interval
        |    out <= dshl(in1, p)
        |    """.stripMargin
    val check = s"""output out : Interval[-128, 128].0 """.stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Interval types" should "infer asInterval correctly" in {
    val passes = Seq(ToWorkingIR, InferTypes, ResolveGenders, new InferWidths())
      val input =
        s"""circuit Unit :
        |  module Unit :
        |    input  p   : UInt<3>
        |    output out : Interval
        |    out <= asInterval(p, 0, 4, 1)
        |    """.stripMargin
    val check = s"""output out : Interval[0, 2].1 """.stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Interval types" should "do wrap/clip correctly" in {
    val passes = Seq(ToWorkingIR, new ResolveAndCheck())
      val input =
        s"""circuit Unit :
        |  module Unit :
        |    input  s:     SInt<2>
        |    input  u:     UInt<3>
        |    input  in1:   Interval[-3, 5].0
        |    output wrap3: Interval
        |    output wrap4: Interval
        |    output wrap5: Interval
        |    output wrap6: Interval
        |    output wrap7: Interval
        |    output clip3: Interval
        |    output clip4: Interval
        |    output clip5: Interval
        |    output clip6: Interval
        |    output clip7: Interval
        |    wrap3 <= wrap(in1, asInterval(s, -2, 4, 0))
        |    wrap4 <= wrap(in1, asInterval(s, -1, 1, 0))
        |    wrap5 <= wrap(in1, asInterval(s, -4, 4, 0))
        |    wrap6 <= wrap(in1, asInterval(s, -1, 7, 0))
        |    wrap7 <= wrap(in1, asInterval(s, -4, 7, 0))
        |    clip3 <= clip(in1, asInterval(s, -2, 4, 0))
        |    clip4 <= clip(in1, asInterval(s, -1, 1, 0))
        |    clip5 <= clip(in1, asInterval(s, -4, 4, 0))
        |    clip6 <= clip(in1, asInterval(s, -1, 7, 0))
        |    clip7 <= clip(in1, asInterval(s, -4, 7, 0))
        """.stripMargin
        //|    output wrap1: Interval
        //|    output wrap2: Interval
        //|    output clip1: Interval
        //|    output clip2: Interval
        //|    wrap1 <= wrap(in1, u, 0)
        //|    wrap2 <= wrap(in1, s, 0)
        //|    clip1 <= clip(in1, u)
        //|    clip2 <= clip(in1, s)
    val check = s"""
        |    output wrap3 : Interval[-2, 4].0
        |    output wrap4 : Interval[-1, 1].0
        |    output wrap5 : Interval[-4, 4].0
        |    output wrap6 : Interval[-1, 7].0
        |    output wrap7 : Interval[-4, 7].0
        |    output clip3 : Interval[-2, 4].0
        |    output clip4 : Interval[-1, 1].0
        |    output clip5 : Interval[-3, 4].0
        |    output clip6 : Interval[-1, 5].0
        |    output clip7 : Interval[-3, 5].0 """.stripMargin
        // TODO: this optimization
        //|    output wrap1 : Interval[0, 7].0
        //|    output wrap2 : Interval[-2, 1].0
        //|    output clip1 : Interval[0, 5].0
        //|    output clip2 : Interval[-2, 1].0
        //|    output wrap7 : Interval[-3, 5].0
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Interval types" should "remove wrap/clip correctly" in {
    val passes = Seq(ToWorkingIR, new ResolveAndCheck(), new RemoveIntervals())
      val input =
        s"""circuit Unit :
        |  module Unit :
        |    input  s:     SInt<2>
        |    input  u:     UInt<3>
        |    input  in1:   Interval[-3, 5].0
        |    output wrap3: Interval
        |    output wrap5: Interval
        |    output wrap6: Interval
        |    output wrap7: Interval
        |    output clip3: Interval
        |    output clip4: Interval
        |    output clip5: Interval
        |    output clip6: Interval
        |    output clip7: Interval
        |    wrap3 <= wrap(in1, asInterval(s, -2, 4, 0))
        |    wrap5 <= wrap(in1, asInterval(s, -4, 4, 0))
        |    wrap6 <= wrap(in1, asInterval(s, -1, 7, 0))
        |    wrap7 <= wrap(in1, asInterval(s, -4, 7, 0))
        |    clip3 <= clip(in1, asInterval(s, -2, 4, 0))
        |    clip4 <= clip(in1, asInterval(s, -1, 1, 0))
        |    clip5 <= clip(in1, asInterval(s, -4, 4, 0))
        |    clip6 <= clip(in1, asInterval(s, -1, 7, 0))
        |    clip7 <= clip(in1, asInterval(s, -4, 7, 0))
        |    """.stripMargin
    val check = s"""
        |    wrap3 <= mux(gt(in1, SInt<4>("h4")), sub(in1, SInt<4>("h7")), mux(lt(in1, SInt<2>("h-2")), add(in1, SInt<4>("h7")), in1))
        |    wrap5 <= mux(gt(in1, SInt<4>("h4")), sub(in1, SInt<5>("h9")), in1)
        |    wrap6 <= mux(lt(in1, SInt<1>("h-1")), add(in1, SInt<5>("h9")), in1)
        |    wrap7 <= in1
        |    clip3 <= mux(gt(in1, SInt<4>("h4")), SInt<4>("h4"), mux(lt(in1, SInt<2>("h-2")), SInt<2>("h-2"), in1))
        |    clip4 <= mux(gt(in1, SInt<2>("h1")), SInt<2>("h1"), mux(lt(in1, SInt<1>("h-1")), SInt<1>("h-1"), in1))
        |    clip5 <= mux(gt(in1, SInt<4>("h4")), SInt<4>("h4"), in1)
        |    clip6 <= mux(lt(in1, SInt<1>("h-1")), SInt<1>("h-1"), in1)
        |    clip7 <= in1
        """.stripMargin
        //|    output wrap4: Interval
        //|    wrap4 <= wrap(in1, asInterval(s, -1, 1, 0), 0)
        //|    wrap4 <= add(rem(sub(in1, SInt<1>("h-1")), sub(SInt<2>("h1"), SInt<1>("h-1"))), SInt<1>("h-1"))
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Interval types" should "shift wrap/clip correctly" in {
    val passes = Seq(ToWorkingIR, new ResolveAndCheck, new RemoveIntervals())
      val input =
        s"""circuit Unit :
        |  module Unit :
        |    input  s:     SInt<2>
        |    input  in1:   Interval[-3, 5].1
        |    output wrap1: Interval
        |    output clip1: Interval
        |    wrap1 <= wrap(in1, asInterval(s, -2, 2, 0))
        |    clip1 <= clip(in1, asInterval(s, -2, 2, 0))
        |    """.stripMargin
    val check = s"""
        |    wrap1 <= mux(gt(in1, SInt<4>("h4")), sub(in1, SInt<5>("h9")), mux(lt(in1, SInt<3>("h-4")), add(in1, SInt<5>("h9")), in1))
        |    clip1 <= mux(gt(in1, SInt<4>("h4")), SInt<4>("h4"), mux(lt(in1, SInt<3>("h-4")), SInt<3>("h-4"), in1))
        """.stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Interval types" should "infer negative binary points" in {
    val passes = Seq(ToWorkingIR, new ResolveAndCheck())
      val input =
        s"""circuit Unit :
        |  module Unit :
        |    input  in1:   Interval[-2, 4].-1
        |    input  in2:   Interval[-4, 8].-2
        |    output out: Interval
        |    out <= add(in1, in2)
        |    """.stripMargin
    val check = s"""
        |    output out : Interval[-6, 12].-1
        """.stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Interval types" should "remove negative binary points" in {
    val passes = Seq(ToWorkingIR, InferTypes, ResolveGenders, new InferBinaryPoints(), new TrimIntervals(), new InferWidths(), new RemoveIntervals())
      val input =
        s"""circuit Unit :
        |  module Unit :
        |    input  in1:   Interval[-2, 4].-1
        |    input  in2:   Interval[-4, 8].-2
        |    output out: Interval.0
        |    out <= add(in1, in2)
        |    """.stripMargin
    val check = s"""
        |    output out : SInt<5>
        |    out <= shl(add(in1, shl(in2, 1)), 1)
        """.stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Interval types" should "implement squz properly" in {
    val passes = Seq(ToWorkingIR, new ResolveAndCheck)
    val input =
      s"""circuit Unit :
         |  module Unit :
         |    input min: Interval[-1, 4].1
         |    input max: Interval[-3, 5].1
         |    input left: Interval[-3, 3].1
         |    input right: Interval[0, 5].1
         |    input off: Interval[-1, 4].2
         |    output minMax: Interval
         |    output maxMin: Interval
         |    output minLeft: Interval
         |    output leftMin: Interval
         |    output minRight: Interval
         |    output rightMin: Interval
         |    output minOff: Interval
         |    output offMin: Interval
         |
         |    minMax <= squz(min, max)
         |    maxMin <= squz(max, min)
         |    minLeft <= squz(min, left)
         |    leftMin <= squz(left, min)
         |    minRight <= squz(min, right)
         |    rightMin <= squz(right, min)
         |    minOff <= squz(min, off)
         |    offMin <= squz(off, min)
         |    """.stripMargin
    val check =
      s"""
         |    output minMax : Interval[-1, 4].1
         |    output maxMin : Interval[-1, 4].1
         |    output minLeft : Interval[-1, 3].1
         |    output leftMin : Interval[-1, 3].1
         |    output minRight : Interval[0, 4].1
         |    output rightMin : Interval[0, 4].1
         |    output minOff : Interval[-1, 4].1
         |    output offMin : Interval[-1, 4].2
        """.stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Interval types" should "lower squz properly" in {
    val passes = Seq(ToWorkingIR, new ResolveAndCheck, new RemoveIntervals)
    val input =
      s"""circuit Unit :
         |  module Unit :
         |    input min: Interval[-1, 4].1
         |    input max: Interval[-3, 5].1
         |    input left: Interval[-3, 3].1
         |    input right: Interval[0, 5].1
         |    input off: Interval[-1, 4].2
         |    output minMax: Interval
         |    output maxMin: Interval
         |    output minLeft: Interval
         |    output leftMin: Interval
         |    output minRight: Interval
         |    output rightMin: Interval
         |    output minOff: Interval
         |    output offMin: Interval
         |
         |    minMax <= squz(min, max)
         |    maxMin <= squz(max, min)
         |    minLeft <= squz(min, left)
         |    leftMin <= squz(left, min)
         |    minRight <= squz(min, right)
         |    rightMin <= squz(right, min)
         |    minOff <= squz(min, off)
         |    offMin <= squz(off, min)
         |    """.stripMargin
    val check =
      s"""
         |    minMax <= asSInt(bits(min, 4, 0))
         |    maxMin <= asSInt(bits(max, 4, 0))
         |    minLeft <= asSInt(bits(min, 3, 0))
         |    leftMin <= left
         |    minRight <= asSInt(bits(min, 4, 0))
         |    rightMin <= asSInt(bits(right, 4, 0))
         |    minOff <= asSInt(bits(min, 4, 0))
         |    offMin <= asSInt(bits(off, 5, 0))
        """.stripMargin
    executeTest(input, check.split("\n") map normalized, passes)
  }
  "Assigning a larger interval to a smaller interval" should "error!" in {
    val passes = Seq(ToWorkingIR, new ResolveAndCheck, new RemoveIntervals)
    val input =
      s"""circuit Unit :
         |  module Unit :
         |    input in: Interval[1, 4].1
         |    output out: Interval[2, 3].1
         |    out <= in
         |    """.stripMargin
    intercept[InvalidConnect]{
      executeTest(input, Nil, passes)
    }
  }
  "Assigning a more precise interval to a less precise interval" should "error!" in {
    val passes = Seq(ToWorkingIR, new ResolveAndCheck, new RemoveIntervals)
    val input =
      s"""circuit Unit :
         |  module Unit :
         |    input in: Interval[2, 3].3
         |    output out: Interval[2, 3].1
         |    out <= in
         |    """.stripMargin
    intercept[InvalidConnect]{
      executeTest(input, Nil, passes)
    }
  }
  "Chick's example" should "work" in {
    val input =
      s"""circuit IntervalChainedSubTester :
         |  module IntervalChainedSubTester :
         |    input clock : Clock
         |    input reset : UInt<1>
         |    node _GEN_0 = sub(SInt<6>("h11"), SInt<6>("h2")) @[IntervalSpec.scala 337:26 IntervalSpec.scala 337:26]
         |    node _GEN_1 = bits(_GEN_0, 4, 0) @[IntervalSpec.scala 337:26 IntervalSpec.scala 337:26]
         |    node intervalResult = asSInt(_GEN_1) @[IntervalSpec.scala 337:26 IntervalSpec.scala 337:26]
         |    skip
         |    node _T_1 = asUInt(intervalResult) @[IntervalSpec.scala 338:50]
         |    skip
         |    node _T_3 = eq(reset, UInt<1>("h0")) @[IntervalSpec.scala 338:9]
         |    node _T_4 = eq(intervalResult, SInt<5>("hf")) @[IntervalSpec.scala 339:25]
         |    skip
         |    node _T_6 = or(_T_4, reset) @[IntervalSpec.scala 339:9]
         |    node _T_7 = eq(_T_6, UInt<1>("h0")) @[IntervalSpec.scala 339:9]
         |    skip
         |    skip
         |    printf(clock, _T_3, "Interval result: %d", _T_1) @[IntervalSpec.scala 338:9]
         |    printf(clock, _T_7, "Assertion failed    at IntervalSpec.scala:339 assert(intervalResult === 15.I)") @[IntervalSpec.scala 339:9]
         |    stop(clock, _T_7, 1) @[IntervalSpec.scala 339:9]
         |    stop(clock, _T_3, 0) @[IntervalSpec.scala 340:7]
         |
       """.stripMargin
    compileToVerilog(input)
  }

  "Squeeze with disjoint intervals" should "error" in {
    intercept[DisjointSqueeze] {
      val input =
        s"""circuit Unit :
           |  module Unit :
           |    input in1: Interval[2, 3).3
           |    input in2: Interval[3, 6].3
           |    node out = squz(in1, in2)
        """.stripMargin
      compileToVerilog(input)
    }
    intercept[DisjointSqueeze] {
      val input =
        s"""circuit Unit :
           |  module Unit :
           |    input in1: Interval[2, 3).3
           |    input in2: Interval[3, 6].3
           |    node out = squz(in2, in1)
        """.stripMargin
      compileToVerilog(input)
    }
  }

  "Clip with disjoint intervals" should "work" in {
    compileToVerilog(
      s"""circuit Unit :
         |  module Unit :
         |    input in1: Interval[2, 3).3
         |    input in2: Interval[3, 6].3
         |    output out: Interval
         |    out <= clip(in1, in2)
      """.stripMargin
    )
    compileToVerilog(
      s"""circuit Unit :
         |  module Unit :
         |    input in1: Interval[2, 3).3
         |    input in2: Interval[4, 6].3
         |    node out = clip(in1, in2)
      """.stripMargin
    )
  }


  "Wrap with remainder" should "error" in {
    intercept[WrapWithRemainder] {
      val input =
        s"""circuit Unit :
           |  module Unit :
           |    input in1: Interval[0, 300).3
           |    input in2: Interval[3, 6].3
           |    node out = wrap(in1, in2)
      """.stripMargin
      compileToVerilog(input)
    }
  }
}
