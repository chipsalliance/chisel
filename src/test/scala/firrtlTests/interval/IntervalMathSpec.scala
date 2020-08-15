// See LICENSE for license details.

package firrtlTests.interval

import firrtl.Implicits.constraint2bound
import firrtl.{ChirrtlForm, CircuitState, LowFirrtlCompiler}
import firrtl.ir._

import firrtl.constraint._
import firrtl.testutils.FirrtlFlatSpec

class IntervalMathSpec extends FirrtlFlatSpec {
  val SumPattern = """.*output sum.*<(\d+)>.*""".r
  val ProductPattern = """.*output product.*<(\d+)>.*""".r
  val DifferencePattern = """.*output difference.*<(\d+)>.*""".r
  val ComparisonPattern = """.*output (\w+).*UInt<(\d+)>.*""".r
  val ShiftLeftPattern = """.*output shl.*<(\d+)>.*""".r
  val ShiftRightPattern = """.*output shr.*<(\d+)>.*""".r
  val DShiftLeftPattern = """.*output dshl.*<(\d+)>.*""".r
  val DShiftRightPattern = """.*output dshr.*<(\d+)>.*""".r
  val ArithAssignPattern = """\s*(\w+) <= asSInt\(bits\((\w+)\((.*)\).*\)\)\s*""".r
  def getBound(bound: String, value: BigDecimal): IsKnown = bound match {
    case "[" => Closed(value)
    case "]" => Closed(value)
    case "(" => Open(value)
    case ")" => Open(value)
  }

  val prec = 0.5

  for {
    lb1 <- Seq("[", "(")
    lv1 <- Range.BigDecimal(-1.0, 1.0, prec)
    uv1 <- if (lb1 == "[") Range.BigDecimal(lv1, 1.0, prec) else Range.BigDecimal(lv1 + prec, 1.0, prec)
    ub1 <- if (lv1 == uv1) Seq("]") else Seq("]", ")")
    bp1 <- 0 to 1
    lb2 <- Seq("[", "(")
    lv2 <- Range.BigDecimal(-1.0, 1.0, prec)
    uv2 <- if (lb2 == "[") Range.BigDecimal(lv2, 1.0, prec) else Range.BigDecimal(lv2 + prec, 1.0, prec)
    ub2 <- if (lv2 == uv2) Seq("]") else Seq("]", ")")
    bp2 <- 0 to 1
  } {
    val it1 = IntervalType(getBound(lb1, lv1), getBound(ub1, uv1), IntWidth(bp1.toInt))
    val it2 = IntervalType(getBound(lb2, lv2), getBound(ub2, uv2), IntWidth(bp2.toInt))
    (it1.range, it2.range) match {
      case (Some(Nil), _) =>
      case (_, Some(Nil)) =>
      case _ =>
        def config = s"$lb1$lv1,$uv1$ub1.$bp1 and $lb2$lv2,$uv2$ub2.$bp2"

        s"Configuration $config" should "pass" in {

          val input =
            s"""circuit Unit :
               |  module Unit :
               |    input  in1 : Interval$lb1$lv1, $uv1$ub1.$bp1
               |    input  in2 : Interval$lb2$lv2, $uv2$ub2.$bp2
               |    input  amt : UInt<3>
               |    output sum        : Interval
               |    output difference : Interval
               |    output product    : Interval
               |    output shl        : Interval
               |    output shr        : Interval
               |    output dshl       : Interval
               |    output dshr       : Interval
               |    output lt         : UInt
               |    output leq        : UInt
               |    output gt         : UInt
               |    output geq        : UInt
               |    output eq         : UInt
               |    output neq        : UInt
               |    output cat        : UInt
               |    sum        <= add(in1, in2)
               |    difference <= sub(in1, in2)
               |    product    <= mul(in1, in2)
               |    shl        <= shl(in1, 3)
               |    shr        <= shr(in1, 3)
               |    dshl       <= dshl(in1, amt)
               |    dshr       <= dshr(in1, amt)
               |    lt         <= lt(in1, in2)
               |    leq        <= leq(in1, in2)
               |    gt         <= gt(in1, in2)
               |    geq        <= geq(in1, in2)
               |    eq         <= eq(in1, in2)
               |    neq        <= lt(in1, in2)
               |    cat        <= cat(in1, in2)
               |    """.stripMargin

          val lowerer = new LowFirrtlCompiler
          val res = lowerer.compileAndEmit(CircuitState(parse(input), ChirrtlForm))
          val output = res.getEmittedCircuit.value.split("\n")
          val min1 = Closed(it1.min.get)
          val max1 = Closed(it1.max.get)
          val min2 = Closed(it2.min.get)
          val max2 = Closed(it2.max.get)
          for (line <- output) {
            line match {
              case SumPattern(varWidth) =>
                val bp = IntWidth(Math.max(bp1.toInt, bp2.toInt))
                val it = IntervalType(IsAdd(min1, min2), IsAdd(max1, max2), bp)
                assert(varWidth.toInt == it.width.asInstanceOf[IntWidth].width, s"$line,${it.range}")
              case ProductPattern(varWidth) =>
                val bp = IntWidth(bp1.toInt + bp2.toInt)
                val lv = IsMin(Seq(IsMul(min1, min2), IsMul(min1, max2), IsMul(max1, min2), IsMul(max1, max2)))
                val uv = IsMax(Seq(IsMul(min1, min2), IsMul(min1, max2), IsMul(max1, min2), IsMul(max1, max2)))
                assert(varWidth.toInt == IntervalType(lv, uv, bp).width.asInstanceOf[IntWidth].width, "product")
              case DifferencePattern(varWidth) =>
                val bp = IntWidth(Math.max(bp1.toInt, bp2.toInt))
                val lv = min1 + max2.neg
                val uv = max1 + min2.neg
                assert(varWidth.toInt == IntervalType(lv, uv, bp).width.asInstanceOf[IntWidth].width, "diff")
              case ShiftLeftPattern(varWidth) =>
                val bp = IntWidth(bp1.toInt)
                val lv = min1 * Closed(8)
                val uv = max1 * Closed(8)
                val it = IntervalType(lv, uv, bp)
                assert(varWidth.toInt == it.width.asInstanceOf[IntWidth].width, "shl")
              case ShiftRightPattern(varWidth) =>
                val bp = IntWidth(bp1.toInt)
                val lv = min1 * Closed(1 / 3)
                val uv = max1 * Closed(1 / 3)
                assert(varWidth.toInt == IntervalType(lv, uv, bp).width.asInstanceOf[IntWidth].width, "shr")
              case DShiftLeftPattern(varWidth) =>
                val bp = IntWidth(bp1.toInt)
                val lv = min1 * Closed(128)
                val uv = max1 * Closed(128)
                assert(varWidth.toInt == IntervalType(lv, uv, bp).width.asInstanceOf[IntWidth].width, "dshl")
              case DShiftRightPattern(varWidth) =>
                val bp = IntWidth(bp1.toInt)
                val lv = min1
                val uv = max1
                assert(varWidth.toInt == IntervalType(lv, uv, bp).width.asInstanceOf[IntWidth].width, "dshr")
              case ComparisonPattern(varWidth) => assert(varWidth.toInt == 1, "==")
              case ArithAssignPattern(varName, operation, args) =>
                val arg1 =
                  if (IntervalType(getBound(lb1, lv1), getBound(ub1, uv1), IntWidth(bp1)).width == IntWidth(0))
                    """SInt<1>("h0")"""
                  else "in1"
                val arg2 =
                  if (IntervalType(getBound(lb2, lv2), getBound(ub2, uv2), IntWidth(bp2)).width == IntWidth(0))
                    """SInt<1>("h0")"""
                  else "in2"
                varName match {
                  case "sum" =>
                    assert(operation === "add", s"""var sum should be result of an add in ${output.mkString("\n")}""")
                    if (bp1 > bp2) {
                      if (arg1 != arg2)
                        assert(!args.contains(s"shl($arg1"), s"$config first arg should be just $arg1 in $line")
                      assert(args.contains(s"shl($arg2, ${bp1 - bp2})"), s"$config second arg incorrect in $line")
                    } else if (bp1 < bp2) {
                      assert(args.contains(s"shl($arg1, ${(bp1 - bp2).abs})"), s"$config second arg incorrect in $line")
                      assert(!args.contains("shl($arg2"), s"$config second arg should be just $arg2 in $line")
                    } else {
                      assert(!args.contains(s"shl($arg1"), s"$config first arg should be just $arg1 in $line")
                      assert(!args.contains(s"shl($arg2"), s"$config second arg should be just $arg2 in $line")
                    }
                  case "product" =>
                    assert(operation === "mul", s"var sum should be result of an add in $line")
                    assert(!args.contains(s"shl($arg1"), s"$config first arg should be just $arg1 in $line")
                    assert(!args.contains(s"shl($arg2"), s"$config second arg should be just $arg2 in $line")
                  case "difference" =>
                    assert(operation === "sub", s"var difference should be result of an sub in $line")
                    if (bp1 > bp2) {
                      if (arg1 != arg2)
                        assert(!args.contains(s"shl($arg1"), s"$config first arg should be just $arg1 in $line")
                      assert(args.contains(s"shl($arg2, ${bp1 - bp2})"), s"$config second arg incorrect in $line")
                    } else if (bp1 < bp2) {
                      assert(args.contains(s"shl($arg1, ${(bp1 - bp2).abs})"), s"$config second arg incorrect in $line")
                      if (arg1 != arg2)
                        assert(!args.contains(s"shl($arg2"), s"$config second arg should be just $arg2 in $line")
                    } else {
                      assert(!args.contains(s"shl($arg1"), s"$config first arg should be just $arg1 in $line")
                      assert(!args.contains(s"shl($arg2"), s"$config second arg should be just $arg2 in $line")
                    }
                  case _ =>
                }
              case _ =>
            }
          }
        }
    }
  }
}

// vim: set ts=4 sw=4 et:
