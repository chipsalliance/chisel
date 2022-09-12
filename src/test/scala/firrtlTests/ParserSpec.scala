// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlSourceAnnotation, FirrtlStage}
import firrtl.testutils._
import firrtl.testutils.FirrtlCheckers._
import org.scalacheck.Gen

class ParserSpec extends FirrtlFlatSpec {

  private object MemTests {
    val prelude = Seq("circuit top :", "  module top :", "    mem m : ")
    val fields = Map(
      "data-type" -> "UInt<32>",
      "depth" -> "4",
      "read-latency" -> "1",
      "write-latency" -> "1",
      "reader" -> "a",
      "writer" -> "b",
      "readwriter" -> "c"
    )
    def fieldsToSeq(m: Map[String, String]): Seq[String] =
      m.map { case (k, v) => s"      ${k} => ${v}" }.toSeq
  }

  private object RegTests {
    val prelude = Seq("circuit top :", "  module top :")
    val regName = "r"
    val reg = s"    reg $regName : UInt<32>, clock"
    val reset = "reset => (radReset, UInt(\"hdeadbeef\"))"
    val sourceLocator = "Reg.scala 33:10"
    val finfo = s"@[$sourceLocator]"
    val fileInfo = FileInfo(StringLit(sourceLocator))
  }

  private object KeywordTests {
    val prelude = Seq("circuit top :", "  module top :")
    val keywords = Seq(
      "circuit",
      "module",
      "extmodule",
      "parameter",
      "input",
      "output",
      "UInt",
      "SInt",
      "Analog",
      "Fixed",
      "Interval",
      "flip",
      "Clock",
      "Reset",
      "AsyncReset",
      "wire",
      "reg",
      "reset",
      "with",
      "mem",
      "depth",
      "reader",
      "writer",
      "readwriter",
      "inst",
      "of",
      "node",
      "is",
      "invalid",
      "when",
      "else",
      "stop",
      "printf",
      "skip",
      "old",
      "new",
      "undefined",
      "mux",
      "validif",
      "cmem",
      "smem",
      "mport",
      "infer",
      "read",
      "write",
      "rdwr",
      "attach",
      "assert",
      "assume",
      "cover",
      "version"
    ) ++ PrimOps.listing
  }

  // ********** FIRRTL version number **********
  "Version 1.1.0" should "be accepted" in {
    val input = """
                  |FIRRTL version 1.1.0
                  |circuit Test :
                  |  module Test :
                  |    input in : UInt<1>
                  |    in <= UInt(0)
      """.stripMargin
    val c = firrtl.Parser.parse(input)
    firrtl.Parser.parse(c.serialize)
  }

  "Version 1.1.1" should "be accepted" in {
    val input = """
                  |FIRRTL version 1.1.1
                  |circuit Test :
                  |  module Test :
                  |    input in : UInt<1>
                  |    in <= UInt(0)
      """.stripMargin
    val c = firrtl.Parser.parse(input)
    firrtl.Parser.parse(c.serialize)
  }

  "No version" should "be accepted" in {
    val input = """
                  |circuit Test :
                  |  module Test :
                  |    input in : { 0 : { 0 : { 0 : UInt<32>, flip 1 : UInt<32> } } }
                  |    in.0.0.1 <= in.0.0.0
      """.stripMargin
    val c = firrtl.Parser.parse(input)
    firrtl.Parser.parse(c.serialize)
  }

  an[UnsupportedVersionException] should be thrownBy {
    val input = """
                  |FIRRTL version 1.2.0
                  |circuit Test :
                  |  module Test :
                  |    input in : UInt<1>
                  |    in <= UInt(0)
      """.stripMargin
    firrtl.Parser.parse(input)
  }

  an[UnsupportedVersionException] should be thrownBy {
    val input = """
                  |FIRRTL version 2.0.0
                  |crcuit Test :
                  |  module Test @@#!# :
                  |    input in1 : UInt<2>
                  |    input in2 : UInt<3>
                  |    output out : UInt<4>
                  |    out[1:0] <= in1
                  |    out[3:2] <= in2[1:0]
      """.stripMargin
    firrtl.Parser.parse(input)
  }

  // ********** Memories **********
  "Memories" should "allow arbitrary ordering of fields" in {
    val fields = MemTests.fieldsToSeq(MemTests.fields)
    val golden = firrtl.Parser.parse((MemTests.prelude ++ fields))

    fields.permutations.foreach { permutation =>
      val circuit = firrtl.Parser.parse((MemTests.prelude ++ permutation))
      assert(golden === circuit)
    }
  }

  it should "have exactly one of each: data-type, depth, read-latency, and write-latency" in {
    import MemTests._
    def parseWithoutField(s:  String) = firrtl.Parser.parse((prelude ++ fieldsToSeq(fields - s)))
    def parseWithDuplicate(k: String, v: String) =
      firrtl.Parser.parse((prelude ++ fieldsToSeq(fields) :+ s"      ${k} => ${v}"))

    Seq("data-type", "depth", "read-latency", "write-latency").foreach { field =>
      an[ParameterNotSpecifiedException] should be thrownBy { parseWithoutField(field) }
      an[ParameterRedefinedException] should be thrownBy { parseWithDuplicate(field, fields(field)) }
    }
  }

  // ********** Registers **********
  "Registers" should "allow no implicit reset" in {
    import RegTests._
    firrtl.Parser.parse((prelude :+ reg))
  }

  it should "allow same-line reset" in {
    import RegTests._
    firrtl.Parser.parse((prelude :+ s"${reg} with : (${reset})" :+ "    wire a : UInt"))
  }

  it should "allow multi-line reset" in {
    import RegTests._
    firrtl.Parser.parse((prelude :+ s"${reg} with :\n      (${reset})"))
  }

  it should "allow source locators with same-line reset" in {
    import RegTests._
    val res = firrtl.Parser.parse((prelude :+ s"${reg} with : (${reset}) $finfo" :+ "    wire a : UInt"))
    CircuitState(res, Nil) should containTree {
      case DefRegister(`fileInfo`, `regName`, _, _, _, _) => true
    }
  }

  it should "allow source locators with multi-line reset" in {
    import RegTests._
    val res = firrtl.Parser.parse((prelude :+ s"${reg} with :\n      (${reset}) $finfo"))
    CircuitState(res, Nil) should containTree {
      case DefRegister(`fileInfo`, `regName`, _, _, _, _) => true
    }
  }

  it should "allow source locators with no reset" in {
    import RegTests._
    val res = firrtl.Parser.parse((prelude :+ s"${reg} $finfo"))
    CircuitState(res, Nil) should containTree {
      case DefRegister(`fileInfo`, `regName`, _, _, _, _) => true
    }
  }

  // ********** Statement labels **********
  it should "allow certain statement to have a label" in {
    val prelude = Seq("circuit top :", "  module top :", "    input c : Clock")
    val statements = Seq("stop(c, UInt(1), 0)", "printf(c, UInt(1), \"\")") ++
      Seq("assert", "assume", "cover").map(_ + "(c, UInt(1), UInt(1), \"\")")
    val validLabels = Seq(":test" -> "test", " :test" -> "test", " : test" -> "test", " : test01" -> "test01")
    statements.foreach { stmt =>
      validLabels.foreach {
        case (lbl, expected) =>
          val line = "    " + stmt + lbl
          val src = (prelude :+ line).mkString("\n") + "\n"
          val res = firrtl.Parser.parse(src)
          CircuitState(res, Nil) should containTree {
            case s: Stop         => s.name == expected
            case s: Print        => s.name == expected
            case s: Verification => s.name == expected
          }
      }
    }
  }

  // ********** Keywords **********
  "Keywords" should "be allowed as Ids" in {
    import KeywordTests._
    keywords.foreach { keyword =>
      firrtl.Parser.parse((prelude :+ s"      wire ${keyword} : UInt"))
    }
  }

  it should "be allowed on lhs in connects" in {
    import KeywordTests._
    keywords.foreach { keyword =>
      firrtl.Parser.parse((prelude ++ Seq(s"      wire ${keyword} : UInt", s"      ${keyword} <= ${keyword}")))
    }
  }

  they should "be allowed as names for side effecting statements" in {
    import KeywordTests._
    keywords.foreach { keyword =>
      firrtl.Parser.parse {
        prelude :+ s"""    assert($keyword, UInt(1), UInt(1), "") : $keyword"""
      }
    }
  }

  // ********** Digits as Fields **********
  "Digits" should "be legal fields in bundles and in subexpressions" in {
    val input = """
                  |circuit Test :
                  |  module Test :
                  |    input in : { 0 : { 0 : UInt<32>, flip 1 : UInt<32> } }
                  |    input in2 : { 4 : { 23 : { foo : UInt<32>, bar : { flip 123 : UInt<32> } } } }
                  |    in.0.1 <= in.0.0
                  |    in2.4.23.bar.123 <= in2.4.23.foo
      """.stripMargin
    val c = firrtl.Parser.parse(input)
    firrtl.Parser.parse(c.serialize)
  }

  // ********** Literal Formats **********
  "Literals of different bases and signs" should "produce correct values" in {
    def circuit(lit: String): firrtl.ir.Circuit = {
      val input = s"""circuit Top :
                     |  module lits:
                     |    output litout : SInt<16>
                     |    litout <= SInt(${lit})
                     |""".stripMargin
      firrtl.Parser.parse(input)
    }

    def check(inFormat: String, ref: Integer): Unit = {
      (circuit(inFormat)) should be(circuit(ref.toString))
    }

    val checks = Map(
      """       12 """ -> 12,
      """      -14 """ -> -14,
      """      +15 """ -> 15,
      """     "hA" """ -> 10,
      """    "h-C" """ -> -12,
      """   "h+1B" """ -> 27,
      """    "o66" """ -> 54,
      """   "o-33" """ -> -27,
      """  "b1101" """ -> 13,
      """ "b-1001" """ -> -9,
      """ "b+1000" """ -> 8
    )

    checks.foreach { case (k, v) => check(k, v) }
  }

  // ********** Doubles as parameters **********
  "Doubles" should "be legal parameters for extmodules" in {
    val nums = Seq("1.0", "7.6", "3.00004", "1.0E10", "1.0023E-17")
    val signs = Seq("", "+", "-")
    val tests = "0.0" +: (signs.flatMap(s => nums.map(n => s + n)))
    for (test <- tests) {
      val input = s"""
                     |circuit Test :
                     |  extmodule Ext :
                     |    input foo : UInt<32>
                     |
                     |    defname = MyExtModule
                     |    parameter REAL = $test
                     |
                     |  module Test :
                     |    input foo : UInt<32>
                     |    output bar : UInt<32>
        """.stripMargin
      val c = firrtl.Parser.parse(input)
      firrtl.Parser.parse(c.serialize)
    }
  }

  "Strings" should "be legal parameters for extmodules" in {
    val input = s"""
                   |circuit Test :
                   |  extmodule Ext :
                   |    input foo : UInt<32>
                   |
                   |    defname = MyExtModule
                   |    parameter STR = "hello=%d"
                   |
                   |  module Test :
                   |    input foo : UInt<32>
                   |    output bar : UInt<32>
      """.stripMargin
    val c = firrtl.Parser.parse(input)
    firrtl.Parser.parse(c.serialize)
  }

  "Parsing errors" should "be reported as normal exceptions" in {
    val input = s"""
                   |circuit Test
                   |  module Test :

                   |""".stripMargin
    a[SyntaxErrorsException] shouldBe thrownBy {
      (new FirrtlStage).execute(Array(), Seq(FirrtlSourceAnnotation(input)))
    }
  }

  "Trailing syntax errors" should "be caught in the parser" in {
    val input = s"""
                   |circuit Foo:
                   |  module Bar:
                   |    input a: UInt<1>
                   |output b: UInt<1>
                   |    b <- a
                   |
                   |  module Foo:
                   |    input a: UInt<1>
                   |    output b: UInt<1>
                   |    inst bar of Bar
                   |    bar.a <- a
                   |    b <- bar.b
      """.stripMargin
    a[SyntaxErrorsException] shouldBe thrownBy {
      (new FirrtlStage).execute(Array(), Seq(FirrtlSourceAnnotation(input)))
    }
  }

  it should "be able to parse a MultiInfo as a FileInfo" in {
    // currently MultiInfo gets flattened into a single string which can only be recovered as a FileInfo
    val info = ir.MultiInfo(Seq(ir.MultiInfo(Seq(ir.FileInfo("a"))), ir.FileInfo("b"), ir.FileInfo("c")))
    val input =
      s"""circuit m:${info.serialize}
         |  module m:
         |    skip
         |""".stripMargin
    val c = firrtl.Parser.parse(input)
    assert(c.info == ir.FileInfo("a b c"))
  }
}

class ParserPropSpec extends FirrtlPropSpec {
  // Disable shrinking on error.
  import org.scalacheck.Shrink
  implicit val noShrinkString = Shrink[String](_ => Stream.empty)

  def legalStartChar = Gen.frequency((1, '_'), (20, Gen.alphaChar))
  def legalChar = Gen.frequency((1, Gen.numChar), (1, '$'), (10, legalStartChar))

  def identifier = for {
    x <- legalStartChar
    xs <- Gen.listOf(legalChar)
  } yield (x :: xs).mkString

  property("Identifiers should allow [A-Za-z0-9_$] but not allow starting with a digit or $") {
    forAll(identifier) { id =>
      whenever(id.nonEmpty) {
        val input = s"""
                       |circuit Test :
                       |  module Test :
                       |    input $id : UInt<32>
                       |""".stripMargin
        firrtl.Parser.parse(input.split("\n"))
      }
    }
  }

  def bundleField = for {
    xs <- Gen.nonEmptyListOf(legalChar)
  } yield xs.mkString

  property("Bundle fields should allow [A-Za-z0-9_] including starting with a digit or $") {
    forAll(identifier, bundleField) {
      case (id, field) =>
        whenever(id.nonEmpty && field.nonEmpty) {
          val input = s"""
                         |circuit Test :
                         |  module Test :
                         |    input $id : { $field : UInt<32> }
                         |""".stripMargin
          firrtl.Parser.parse(input.split("\n"))
        }
    }
  }
}
