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

class InferReadWriteSpec extends SimpleTransformSpec {
  object InferReadWriteCheckPass extends Pass {
    val name = "Check Infer ReadWrite Ports"
    var foundReadWrite = false
    def findReadWrite(s: Statement): Unit = s match {
      case s: DefMemory if s.readLatency > 0 =>
        foundReadWrite = s.name == "mem" && s.readwriters.size == 1
      case s: Block =>
        s.stmts foreach findReadWrite
      case _ =>
    }

    def run (c: Circuit) = {
      val errors = new Errors
      c.modules foreach {
        case m: Module => findReadWrite(m.body)
        case m: ExtModule => m
      }
      if (!foundReadWrite) {
        errors append new PassException("Readwrite ports are not found!")
      }
      errors.trigger
      c
    }
  }

  object InferReadWriteCheck extends Transform with SimpleRun {
    def execute (c: Circuit, map: AnnotationMap) =
      run(c, Seq(InferReadWriteCheckPass))
  }

  def transforms (writer: java.io.Writer) = Seq(
     new Chisel3ToHighFirrtl(),
     new IRToWorkingIR(),
     new ResolveAndCheck(),
     new HighFirrtlToMiddleFirrtl(),
     new InferReadWrite(TransID(-1)),
     InferReadWriteCheck,
     new EmitFirrtl(writer)
  )

  "Infer ReadWrite Ports" should "infer readwrite ports" in {
    val input = """
circuit sram6t :
  module sram6t :
    input clk : Clock
    input reset : UInt<1>
    output io : {flip en : UInt<1>, flip wen : UInt<1>, flip waddr : UInt<8>, flip wdata : UInt<32>, flip raddr : UInt<8>, rdata : UInt<32>}

    io is invalid
    smem mem : UInt<32>[128]
    node T_0 = eq(io.wen, UInt<1>("h00"))
    node T_1 = and(io.en, T_0)
    wire T_2 : UInt
    T_2 is invalid
    when T_1 :
      T_2 <= io.raddr
    read mport T_3 = mem[T_2], clk
    io.rdata <= T_3
    node T_4 = and(io.en, io.wen)
    when T_4 :
      write mport T_5 = mem[io.waddr], clk
      T_5 <= io.wdata
""".stripMargin

    val annotaitonMap = AnnotationMap(Seq(InferReadWriteAnnotation("sram6t", TransID(-1))))
    compile(parse(input), annotaitonMap, new java.io.StringWriter)
  }
}
