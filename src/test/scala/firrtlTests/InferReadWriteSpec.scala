// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.passes._
import firrtl.Mappers._
import annotations._
import FirrtlCheckers._

class InferReadWriteSpec extends SimpleTransformSpec {
  class InferReadWriteCheckException extends PassException(
    "Readwrite ports are not found!")

  object InferReadWriteCheck extends Pass {
    override def inputForm = MidForm
    override def outputForm = MidForm
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

  def emitter = new MiddleFirrtlEmitter
  def transforms = Seq(
    new ChirrtlToHighFirrtl,
    new IRToWorkingIR,
    new ResolveAndCheck,
    new HighFirrtlToMiddleFirrtl,
    new memlib.InferReadWrite,
    InferReadWriteCheck
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

    val annos = Seq(memlib.InferReadWriteAnnotation)
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    parse(res.getEmittedCircuit.value)
  }

  "Infer ReadWrite Ports" should "infer readwrite ports from exclusive when statements" in {
    val input = """
circuit sram6t :
  module sram6t :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip addr : UInt<11>, flip ren : UInt<1>, flip wen : UInt<1>, flip dataIn : UInt<32>, dataOut : UInt<32>}

    io is invalid
    smem mem : UInt<32> [2048]
    when io.wen :
      write mport _T_14 = mem[io.addr], clock
      _T_14 <= io.dataIn
    node _T_16 = eq(io.wen, UInt<1>("h0"))
    when _T_16 :
      wire _T_18 : UInt
      _T_18 is invalid
      when io.ren :
        _T_18 <= io.addr
        node _T_20 = or(_T_18, UInt<11>("h0"))
        node _T_21 = bits(_T_20, 10, 0)
        read mport _T_22 = mem[_T_21], clock
      io.dataOut <= _T_22
""".stripMargin

    val annos = Seq(memlib.InferReadWriteAnnotation)
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    parse(res.getEmittedCircuit.value)
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

    val annos = Seq(memlib.InferReadWriteAnnotation)
    intercept[InferReadWriteCheckException] {
      compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    }
  }

  "wmode" should "be simplified" in {
    val input = """
circuit sram6t :
  module sram6t :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip addr : UInt<11>, flip valid : UInt<1>, flip write : UInt<1>, flip dataIn : UInt<32>, dataOut : UInt<32>}

    io is invalid
    smem mem : UInt<32> [2048]
    node wen = and(io.valid, io.write)
    node ren = and(io.valid, not(io.write))
    when wen :
      write mport _T_14 = mem[io.addr], clock
      _T_14 <= io.dataIn
    node _T_16 = eq(wen, UInt<1>("h0"))
    when _T_16 :
      wire _T_18 : UInt
      _T_18 is invalid
      when ren :
        _T_18 <= io.addr
        node _T_20 = or(_T_18, UInt<11>("h0"))
        node _T_21 = bits(_T_20, 10, 0)
        read mport _T_22 = mem[_T_21], clock
      io.dataOut <= _T_22
""".stripMargin

    val annos = Seq(memlib.InferReadWriteAnnotation)
    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm, annos))
    // Check correctness of firrtl
    res should containLine (s"mem.rw.wmode <= wen")
  }
}
