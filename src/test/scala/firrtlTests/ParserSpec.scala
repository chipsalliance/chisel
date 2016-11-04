// See LICENSE for license details.

package firrtlTests

import org.scalatest._
import firrtl._

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
  }

  private object KeywordTests {
    val prelude = Seq("circuit top :", "  module top :")
    val keywords = Seq( "circuit", "module", "extmodule", "input", "output",
      "UInt", "SInt", "flip", "Clock", "wire", "reg", "reset", "with", "mem",
      "data-type", "depth", "read-latency", "write-latency",
      "read-under-write", "reader", "writer", "readwriter", "inst", "of",
      "node", "is", "invalid", "when", "else", "stop", "printf", "skip", "old",
      "new", "undefined", "mux", "validif", "UBits", "SBits", "cmem", "smem",
      "mport", "infer", "read", "write", "rdwr"
    ) ++ PrimOps.listing
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
      evaluating { parseWithoutField(field) } should produce [ParameterNotSpecifiedException]
      evaluating { parseWithDuplicate(field, fields(field)) } should produce [ParameterRedefinedException]
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
}
