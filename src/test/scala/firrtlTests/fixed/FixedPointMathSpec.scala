// See LICENSE for license details.

package firrtlTests.fixed

import firrtl.{CircuitState, ChirrtlForm, LowFirrtlCompiler, Parser}
import firrtl.Parser.IgnoreInfo
import firrtlTests.FirrtlFlatSpec

class FixedPointMathSpec extends FirrtlFlatSpec {

  val SumPattern        = """.*output sum.*<(\d+)>.*.*""".r
  val ProductPattern    = """.*output product.*<(\d+)>.*""".r
  val DifferencePattern = """.*output difference.*<(\d+)>.*""".r

  val AssignPattern     = """\s*(\w+) <= (\w+)\((.*)\)\s*""".r

  for {
    bits1        <- 1 to 4
    binaryPoint1 <- 1 to 4
    bits2        <- 1 to 4
    binaryPoint2 <- 1 to 4
  } {
    def config = s"($bits1,$binaryPoint1)($bits2,$binaryPoint2)"

    s"Configuration $config" should "pass" in {

      val input =
        s"""circuit Unit :
        |  module Unit :
        |    input  a : Fixed<$bits1><<$binaryPoint1>>
        |    input  b : Fixed<$bits2><<$binaryPoint2>>
        |    output sum        : Fixed
        |    output product    : Fixed
        |    output difference : Fixed
        |    sum        <= add(a, b)
        |    product    <= mul(a, b)
        |    difference <= sub(a, b)
        |    """.stripMargin

      val lowerer = new LowFirrtlCompiler

      val res = lowerer.compileAndEmit(CircuitState(parse(input), ChirrtlForm))

      val output = res.getEmittedCircuit.value split "\n"

      def inferredAddWidth: Int = {
        val binaryDifference = binaryPoint1 - binaryPoint2
        val (newW1, newW2) = if(binaryDifference > 0) {
          (bits1, bits2 + binaryDifference)
        } else {
          (bits1 + binaryDifference.abs, bits2)
        }
        newW1.max(newW2) + 1
      }

      for (line <- output) {
        line match {
          case SumPattern(varWidth)     =>
            assert(varWidth.toInt === inferredAddWidth, s"$config sum sint bits wrong for $line")
          case ProductPattern(varWidth) =>
            assert(varWidth.toInt === bits1 + bits2, s"$config product bits wrong for $line")
          case DifferencePattern(varWidth)     =>
            assert(varWidth.toInt === inferredAddWidth, s"$config difference bits wrong for $line")
          case AssignPattern(varName, operation, args) =>
            varName match {
              case "sum" =>
                assert(operation === "add", s"var sum should be result of an add in $line")
                if (binaryPoint1 > binaryPoint2) {
                  assert(!args.contains("shl(a"), s"$config first arg should be just a in $line")
                  assert(args.contains(s"shl(b, ${binaryPoint1 - binaryPoint2})"),
                    s"$config second arg incorrect in $line")
                } else if (binaryPoint1 < binaryPoint2) {
                  assert(args.contains(s"shl(a, ${(binaryPoint1 - binaryPoint2).abs})"),
                    s"$config second arg incorrect in $line")
                  assert(!args.contains("shl(b"), s"$config second arg should be just b in $line")
                } else {
                  assert(!args.contains("shl(a"), s"$config first arg should be just a in $line")
                  assert(!args.contains("shl(b"), s"$config second arg should be just b in $line")
                }
              case "product" =>
                assert(operation === "mul", s"var sum should be result of an add in $line")
                assert(!args.contains("shl(a"), s"$config first arg should be just a in $line")
                assert(!args.contains("shl(b"), s"$config second arg should be just b in $line")
              case "difference" =>
                assert(operation === "sub", s"var difference should be result of an sub in $line")
                if (binaryPoint1 > binaryPoint2) {
                  assert(!args.contains("shl(a"), s"$config first arg should be just a in $line")
                  assert(args.contains(s"shl(b, ${binaryPoint1 - binaryPoint2})"),
                    s"$config second arg incorrect in $line")
                } else if (binaryPoint1 < binaryPoint2) {
                  assert(args.contains(s"shl(a, ${(binaryPoint1 - binaryPoint2).abs})"),
                    s"$config second arg incorrect in $line")
                  assert(!args.contains("shl(b"), s"$config second arg should be just b in $line")
                } else {
                  assert(!args.contains("shl(a"), s"$config first arg should be just a in $line")
                  assert(!args.contains("shl(b"), s"$config second arg should be just b in $line")
                }
              case _ =>
            }
          case _ =>
        }
      }
    }
  }
}

