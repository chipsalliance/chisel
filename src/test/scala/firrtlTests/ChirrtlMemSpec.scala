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

import firrtl._
import firrtl.ir._
import firrtl.passes._
import firrtl.Mappers._
import Annotations._

class ChirrtlMemSpec extends LowTransformSpec {
  object MemEnableCheckPass extends Pass {
    val name = "Check Enable Signal for Chirrtl Mems"
    type Netlist = collection.mutable.HashMap[String, Expression]
    def buildNetlist(netlist: Netlist)(s: Statement): Statement = {
      s match {
        case s: Connect => Utils.kind(s.loc) match {
          case MemKind => netlist(s.loc.serialize) = s.expr
          case _ =>
        }
        case _ =>
      }
      s map buildNetlist(netlist)
    }

    // walks on memories and checks whether or not read enables are high
    def checkStmt(netlist: Netlist)(s: Statement): Boolean = s match {
      case s: DefMemory if s.name == "mem" && s.readers.size == 1=>
        val en = MemPortUtils.memPortField(s, s.readers.head, "en")
        // memory read enable ?= 1
        WrappedExpression.weq(netlist(en.serialize), Utils.one)
      case s: Block =>
        s.stmts exists checkStmt(netlist)
      case _ => false
    }

    def run (c: Circuit) = {
      val errors = new Errors
      val check = c.modules exists {
        case m: Module =>
          val netlist = new Netlist
          checkStmt(netlist)(buildNetlist(netlist)(m.body))
        case m: ExtModule => false
      }
      if (!check) {
        errors append new PassException(
          "Enable signal for the read port is incorrect!")
        errors.trigger
      }
      c
    }
  }

  def transform = new PassBasedTransform {
    def inputForm = LowForm
    def outputForm = LowForm
    def passSeq = Seq(ConstProp, MemEnableCheckPass)
  }

  "Sequential Memory" should "have correct enable signals" in {
    val input = """
circuit foo :
  module foo :
    input clk : Clock
    input reset : UInt<1>
    output io : {flip wen : UInt<1>, flip in : UInt<1>, flip counter : UInt<2>, out : UInt<1>}

    io is invalid
    smem mem : UInt<1>[4]
    node T_0 = add(io.counter, UInt<1>("h01"))
    node temp = tail(T_0, 1)
    read mport bar = mem[temp], clk
    when io.wen :
      write mport T_1 = mem[io.counter], clk
      T_1 <= io.in
    io.out <= bar
""".stripMargin

    val annotationMap = AnnotationMap(Nil)
    val writer = new java.io.StringWriter
    compile(CircuitState(parse(input), ChirrtlForm, Some(annotationMap)), writer)
    // Check correctness of firrtl
    parse(writer.toString)
  }

  "Combinational Memory" should "have correct enable signals" in {
    val input = """
circuit foo :
  module foo :
    input clk : Clock
    input reset : UInt<1>
    output io : {flip ren: UInt<1>, flip wen : UInt<1>, flip in : UInt<1>, flip counter : UInt<2>, out : UInt<1>}

    io is invalid
    cmem mem : UInt<1>[4]
    reg counter : UInt<1>, clk with : (reset => (reset, UInt<1>("h0")))
    read mport bar = mem[counter], clk
    when io.ren:
      counter <= add(counter, UInt<1>("h1"))
    when io.wen :
      write mport T_1 = mem[io.counter], clk
      T_1 <= io.in
    io.out <= bar
""".stripMargin

    val annotationMap = AnnotationMap(Nil)
    val writer = new java.io.StringWriter
    compile(CircuitState(parse(input), ChirrtlForm, Some(annotationMap)), writer)
    // Check correctness of firrtl
    parse(writer.toString)
  }
}
