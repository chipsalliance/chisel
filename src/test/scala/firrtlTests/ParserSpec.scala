// See LICENSE for license details.

package firrtlTests

import org.scalatest._
import firrtl._
import org.scalacheck.Gen
import org.scalacheck.Prop.forAll

class ParserSpec extends FirrtlFlatSpec {

  private object MemTests {
    val prelude = Seq("circuit top :", "  module top :", "    mem m : ")
    val fields = Map("data-type" -> "UInt<32>",
                     "depth" -> "4",
                     "read-latency" -> "1",
                     "write-latency" -> "1",
                     "reader" -> "a",
                     "writer" -> "b",
                     "readwriter" -> "c"
                    )
    def fieldsToSeq(m: Map[String, String]): Seq[String] =
      m map { case (k,v) => s"      ${k} => ${v}" } toSeq
  }

  private object RegTests {
    val prelude = Seq("circuit top :", "  module top :")
    val reg = "    reg r : UInt<32>, clock"
    val reset = "reset => (radReset, UInt(\"hdeadbeef\"))"
    val finfo = "@[Reg.scala:33:10]"
  }

  private object KeywordTests {
    val prelude = Seq("circuit top :", "  module top :")
    val keywords = Seq("circuit", "module", "extmodule", "parameter", "input", "output", "UInt",
      "SInt", "Analog", "Fixed", "flip", "Clock", "wire", "reg", "reset", "with", "mem", "depth",
      "reader", "writer", "readwriter", "inst", "of", "node", "is", "invalid", "when", "else",
      "stop", "printf", "skip", "old", "new", "undefined", "mux", "validif", "cmem", "smem",
      "mport", "infer", "read", "write", "rdwr") ++ PrimOps.listing
  }

  // ********** Memories **********
  "Memories" should "allow arbitrary ordering of fields" in {
    val fields = MemTests.fieldsToSeq(MemTests.fields)
    val golden = firrtl.Parser.parse((MemTests.prelude ++ fields))

    fields.permutations foreach { permutation =>
      val circuit = firrtl.Parser.parse((MemTests.prelude ++ permutation))
      assert(golden === circuit)
    }
  }

  it should "have exactly one of each: data-type, depth, read-latency, and write-latency" in {
    import MemTests._
    def parseWithoutField(s: String) = firrtl.Parser.parse((prelude ++ fieldsToSeq(fields - s)))
    def parseWithDuplicate(k: String, v: String) =
      firrtl.Parser.parse((prelude ++ fieldsToSeq(fields) :+ s"      ${k} => ${v}"))

    Seq("data-type", "depth", "read-latency", "write-latency") foreach { field =>
      an [ParameterNotSpecifiedException] should be thrownBy { parseWithoutField(field) }
      an [ParameterRedefinedException] should be thrownBy { parseWithDuplicate(field, fields(field)) }
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
    firrtl.Parser.parse((prelude :+ s"${reg} with : (${reset}) $finfo" :+ "    wire a : UInt"))
  }

  it should "allow source locators with multi-line reset" in {
    import RegTests._
    firrtl.Parser.parse((prelude :+ s"${reg} with :\n      (${reset}) $finfo"))
  }

  // ********** Keywords **********
  "Keywords" should "be allowed as Ids" in {
    import KeywordTests._
    keywords foreach { keyword =>
      firrtl.Parser.parse((prelude :+ s"      wire ${keyword} : UInt"))
    }
  }

  it should "be allowed on lhs in connects" in {
    import KeywordTests._
    keywords foreach { keyword =>
      firrtl.Parser.parse((prelude ++ Seq(s"      wire ${keyword} : UInt",
                                          s"      ${keyword} <= ${keyword}")))
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

  // ********** Doubles as parameters **********
  "Doubles" should "be legal parameters for extmodules" in {
    val nums = Seq("1.0", "7.6", "3.00004", "1.0E10", "1.0023E-17")
    val signs = Seq("", "+", "-")
    val tests = "0.0" +: (signs flatMap (s => nums map (n => s + n)))
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
    val manager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
      firrtlOptions = FirrtlExecutionOptions(firrtlSource = Some(input))
    }
    a [SyntaxErrorsException] shouldBe thrownBy {
      Driver.execute(manager)
    }
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
    forAll (identifier) { id =>
      whenever(id.nonEmpty) {
        val input = s"""
           |circuit Test :
           |  module Test :
           |    input $id : UInt<32>
           |""".stripMargin
        firrtl.Parser.parse(input split "\n")
      }
    }
  }

  def bundleField = for {
    xs <- Gen.nonEmptyListOf(legalChar)
  } yield xs.mkString

  property("Bundle fields should allow [A-Za-z0-9_] including starting with a digit or $") {
    forAll (identifier, bundleField) { case (id, field) =>
      whenever(id.nonEmpty && field.nonEmpty) {
        val input = s"""
           |circuit Test :
           |  module Test :
           |    input $id : { $field : UInt<32> }
           |""".stripMargin
        firrtl.Parser.parse(input split "\n")
      }
    }
  }
}
