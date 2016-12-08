// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.passes._
import firrtl.Mappers._
import annotations._

class InferReadWriteSpec extends SimpleTransformSpec {
  class InferReadWriteCheckException extends PassException(
    "Readwrite ports are not found!")

  object InferReadWriteCheckPass extends Pass {
    val name = "Check Infer ReadWrite Ports"
    def findReadWrite(s: Statement): Boolean = s match {
      case s: DefMemory if s.readLatency > 0 && s.readwriters.size == 1 =>
        s.name == "mem" && s.readwriters.head == "rw"
      case s: Block =>
        s.stmts exists findReadWrite
      case _ => false
    }

    def run (c: Circuit) = {
      val errors = new Errors
      val foundReadWrite = c.modules exists {
        case m: Module => findReadWrite(m.body)
        case m: ExtModule => false
      }
      if (!foundReadWrite) {
        errors append new InferReadWriteCheckException
        errors.trigger
      }
      c
    }
  }

  class InferReadWriteCheck extends PassBasedTransform {
    def inputForm = MidForm
    def outputForm = MidForm
    def passSeq = Seq(InferReadWriteCheckPass)
  }

  def transforms = Seq(
    new ChirrtlToHighFirrtl,
    new IRToWorkingIR,
    new ResolveAndCheck,
    new HighFirrtlToMiddleFirrtl,
    new memlib.InferReadWrite,
    new InferReadWriteCheck
  )

  "Infer ReadWrite Ports" should "infer readwrite ports for the same clock" in {
    val input = """
circuit sram6t :
  module sram6t :
    input clock : Clock
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
    read mport T_3 = mem[T_2], clock
    io.rdata <= T_3
    node T_4 = and(io.en, io.wen)
    when T_4 :
      write mport T_5 = mem[io.waddr], clock
      T_5 <= io.wdata
""".stripMargin

    val annotationMap = AnnotationMap(Seq(memlib.InferReadWriteAnnotation("sram6t")))
    val writer = new java.io.StringWriter
    compile(CircuitState(parse(input), ChirrtlForm, Some(annotationMap)), writer)
    // Check correctness of firrtl
    parse(writer.toString)
  }

  "Infer ReadWrite Ports" should "not infer readwrite ports for the difference clocks" in {
    val input = """
circuit sram6t :
  module sram6t :
    input clk1 : Clock
    input clk2 : Clock
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
    read mport T_3 = mem[T_2], clk1
    io.rdata <= T_3
    node T_4 = and(io.en, io.wen)
    when T_4 :
      write mport T_5 = mem[io.waddr], clk2
      T_5 <= io.wdata
""".stripMargin

    val annotationMap = AnnotationMap(Seq(memlib.InferReadWriteAnnotation("sram6t")))
    val writer = new java.io.StringWriter
    intercept[InferReadWriteCheckException] {
      compile(CircuitState(parse(input), ChirrtlForm, Some(annotationMap)), writer)
    }
  }
}
