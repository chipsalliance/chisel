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

class CInferMDir extends LowTransformSpec {
  object CInferMDirCheckPass extends Pass {
    val name = "Check Enable Signal for Chirrtl Mems"

    // finds the memory and check its read port
    def checkStmt(s: Statement): Boolean = s match {
      case s: DefMemory if s.name == "indices" =>
        (s.readers contains "index") &&
        (s.writers contains "bar") &&
        s.readwriters.isEmpty
      case s: Block =>
        s.stmts exists checkStmt
      case _ => false
    }

    def run (c: Circuit) = {
      val errors = new Errors
      val check = c.modules exists {
        case m: Module => checkStmt(m.body)
        case m: ExtModule => false
      }
      if (!check) {
        errors append new PassException(
          "Memory has incorrect port directions!")
        errors.trigger
      }
      c
    }
  }

  object CInferMDirCheck extends Transform with SimpleRun {
    def execute(c: Circuit, map: AnnotationMap) =
      run(c, Seq(ConstProp, CInferMDirCheckPass))
  }

  def transform = CInferMDirCheck

  "Memory" should "have correct mem port directions" in {
    val input = """
circuit foo :
  module foo :
    input clk : Clock
    input reset : UInt<1>
    output io : {flip wen : UInt<1>, flip in : UInt<1>, flip counter : UInt<2>, ren: UInt<1>[4], out : UInt<1>[4]}

    io is invalid
    cmem indices : UInt<2>[4]
    node T_0 = add(io.counter, UInt<1>("h01"))
    node temp = tail(T_0, 1)
    infer mport index = indices[temp], clk
    io.out[0] <= UInt<1>("h0")
    io.out[1] <= UInt<1>("h0")
    io.out[2] <= UInt<1>("h0")
    io.out[3] <= UInt<1>("h0")
    when io.ren[index] :
      io.out[index] <= io.in
    else :
      when io.wen :
        infer mport bar = indices[temp], clk
        bar <= io.in
""".stripMargin

    val annotationMap = AnnotationMap(Nil)
    val writer = new java.io.StringWriter
    compile(parse(input), annotationMap, writer)
    // Check correctness of firrtl
    parse(writer.toString)
  }
}
