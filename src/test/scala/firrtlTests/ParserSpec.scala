/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

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
