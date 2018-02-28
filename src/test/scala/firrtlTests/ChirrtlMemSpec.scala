// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.passes._
import firrtl.transforms._
import firrtl.Mappers._
import annotations._
import FirrtlCheckers._
import firrtl.PrimOps.AsClock

class ChirrtlMemSpec extends LowTransformSpec {
  object MemEnableCheckPass extends Pass {
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

  def transform = new SeqTransform {
    def inputForm = LowForm
    def outputForm = LowForm
    def transforms = Seq(new ConstantPropagation, MemEnableCheckPass)
  }

  "Sequential Memory" should "have correct enable signals" in {
    val input = """
circuit foo :
  module foo :
    input clock : Clock
    input reset : UInt<1>
    output io : {flip wen : UInt<1>, flip in : UInt<1>, flip counter : UInt<2>, out : UInt<1>}

    io is invalid
    smem mem : UInt<1>[4]
    node T_0 = add(io.counter, UInt<1>("h01"))
    node temp = tail(T_0, 1)
    read mport bar = mem[temp], clock
    when io.wen :
      write mport T_1 = mem[io.counter], clock
      T_1 <= io.in
    io.out <= bar
""".stripMargin

    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm))
    // Check correctness of firrtl
    parse(res.getEmittedCircuit.value)
  }

  "Combinational Memory" should "have correct enable signals" in {
    val input = """
circuit foo :
  module foo :
    input clock : Clock
    input reset : UInt<1>
    output io : {flip ren: UInt<1>, flip wen : UInt<1>, flip in : UInt<1>, flip counter : UInt<2>, out : UInt<1>}

    io is invalid
    cmem mem : UInt<1>[4]
    reg counter : UInt<1>, clock with : (reset => (reset, UInt<1>("h0")))
    read mport bar = mem[counter], clock
    when io.ren:
      counter <= add(counter, UInt<1>("h1"))
    when io.wen :
      write mport T_1 = mem[io.counter], clock
      T_1 <= io.in
    io.out <= bar
""".stripMargin

    val res = compileAndEmit(CircuitState(parse(input), ChirrtlForm))
    // Check correctness of firrtl
    parse(res.getEmittedCircuit.value)
  }

  ignore should "Memories should not have validif on port clocks when declared in a when" in {
    val input =
      """;buildInfoPackage: chisel3, version: 3.0-SNAPSHOT, scalaVersion: 2.11.11, sbtVersion: 0.13.16, builtAtString: 2017-10-06 20:55:20.367, builtAtMillis: 1507323320367
        |circuit Stack :
        |  module Stack :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    output io : {flip push : UInt<1>, flip pop : UInt<1>, flip en : UInt<1>, flip dataIn : UInt<32>, dataOut : UInt<32>}
        |
        |    clock is invalid
        |    reset is invalid
        |    io is invalid
        |    cmem stack_mem : UInt<32>[4] @[Stack.scala 15:22]
        |    reg sp : UInt<3>, clock with : (reset => (reset, UInt<3>("h00"))) @[Stack.scala 16:26]
        |    reg out : UInt<32>, clock with : (reset => (reset, UInt<32>("h00"))) @[Stack.scala 17:26]
        |    when io.en : @[Stack.scala 19:16]
        |      node _T_14 = lt(sp, UInt<3>("h04")) @[Stack.scala 20:25]
        |      node _T_15 = and(io.push, _T_14) @[Stack.scala 20:18]
        |      when _T_15 : @[Stack.scala 20:42]
        |        node _T_16 = bits(sp, 1, 0)
        |        infer mport _T_17 = stack_mem[_T_16], clock
        |        _T_17 <= io.dataIn @[Stack.scala 21:21]
        |        node _T_19 = add(sp, UInt<1>("h01")) @[Stack.scala 22:16]
        |        node _T_20 = tail(_T_19, 1) @[Stack.scala 22:16]
        |        sp <= _T_20 @[Stack.scala 22:10]
        |        skip @[Stack.scala 20:42]
        |      else : @[Stack.scala 23:39]
        |        node _T_22 = gt(sp, UInt<1>("h00")) @[Stack.scala 23:31]
        |        node _T_23 = and(io.pop, _T_22) @[Stack.scala 23:24]
        |        when _T_23 : @[Stack.scala 23:39]
        |          node _T_25 = sub(sp, UInt<1>("h01")) @[Stack.scala 24:16]
        |          node _T_26 = asUInt(_T_25) @[Stack.scala 24:16]
        |          node _T_27 = tail(_T_26, 1) @[Stack.scala 24:16]
        |          sp <= _T_27 @[Stack.scala 24:10]
        |          skip @[Stack.scala 23:39]
        |      node _T_29 = gt(sp, UInt<1>("h00")) @[Stack.scala 26:14]
        |      when _T_29 : @[Stack.scala 26:21]
        |        node _T_31 = sub(sp, UInt<1>("h01")) @[Stack.scala 27:27]
        |        node _T_32 = asUInt(_T_31) @[Stack.scala 27:27]
        |        node _T_33 = tail(_T_32, 1) @[Stack.scala 27:27]
        |        node _T_34 = bits(_T_33, 1, 0)
        |        infer mport _T_35 = stack_mem[_T_34], clock
        |        out <= _T_35 @[Stack.scala 27:11]
        |        skip @[Stack.scala 26:21]
        |      skip @[Stack.scala 19:16]
        |    io.dataOut <= out @[Stack.scala 31:14]
        """.stripMargin
    val res = (new LowFirrtlCompiler).compile(CircuitState(parse(input), ChirrtlForm), Seq()).circuit
    assert(res search {
      case Connect(_, WSubField(WSubField(WRef("stack_mem", _, _, _), "_T_35",_, _), "clk", _, _), WRef("clock", _, _, _)) => true
      case Connect(_, WSubField(WSubField(WRef("stack_mem", _, _, _), "_T_17",_, _), "clk", _, _), WRef("clock", _, _, _)) => true
    })
  }

  ignore should "Mem non-local clock port assignment should be ok assign in only one side of when" in {
    val input =
      """circuit foo :
        |  module foo :
        |    input clock : Clock
        |    input en : UInt<1>
        |    input addr: UInt<2>
        |    output out: UInt<32>
        |    out is invalid
        |    cmem mem : UInt<32>[4]
        |    when en:
        |      read mport bar = mem[addr], clock
        |      out <= bar
        |""".stripMargin
    val res = (new LowFirrtlCompiler).compile(CircuitState(parse(input), ChirrtlForm), Seq()).circuit
    assert(res search {
      case Connect(_, WSubField(WSubField(WRef("mem", _, _, _), "bar",_, _), "clk", _, _), WRef("clock", _, _, _)) => true
    })
  }

  ignore should "Mem local clock port assignment should be ok" in {
    val input =
      """circuit foo :
        |  module foo :
        |    input clock : Clock
        |    input en : UInt<1>
        |    input addr: UInt<2>
        |    output out: UInt<32>
        |    out is invalid
        |    cmem mem : UInt<32>[4]
        |    when en:
        |      node local = clock
        |      read mport bar = mem[addr], local
        |      out <= bar
        |""".stripMargin
    val res = new LowFirrtlCompiler().compile(CircuitState(parse(input), ChirrtlForm), Seq()).circuit
    assert(res search {
      case Connect(_, WSubField(WSubField(WRef("mem", _, _, _), "bar",_, _), "clk", _, _), WRef("clock", _, _, _)) => true
    })
  }

  ignore should "Mem local nested clock port assignment should be ok" in {
    val input =
      """circuit foo :
        |  module foo :
        |    input clock : Clock
        |    input en : UInt<1>
        |    input addr: UInt<2>
        |    output out: UInt<32>
        |    out is invalid
        |    cmem mem : UInt<32>[4]
        |    when en:
        |      node local = clock
        |      read mport bar = mem[addr], asClock(local)
        |      out <= bar
        |""".stripMargin
    val res = new LowFirrtlCompiler().compile(CircuitState(parse(input), ChirrtlForm), Seq()).circuit
    assert(res search {
      case Connect(_, WSubField(WSubField(WRef("mem", _, _, _), "bar",_, _), "clk", _, _), DoPrim(AsClock, Seq(WRef("clock", _, _, _)), Nil, _)) => true
    })
  }


  ignore should "Mem non-local nested clock port assignment should be ok" in {
    val input =
      """circuit foo :
        |  module foo :
        |    input clock : Clock
        |    input en : UInt<1>
        |    input addr: UInt<2>
        |    output out: UInt<32>
        |    out is invalid
        |    cmem mem : UInt<32>[4]
        |    when en:
        |      read mport bar = mem[addr], asClock(clock)
        |      out <= bar
        |""".stripMargin
    val res = (new HighFirrtlCompiler).compile(CircuitState(parse(input), ChirrtlForm), Seq()).circuit
    assert(res search {
      case Connect(_, SubField(SubField(Reference("mem", _), "bar", _), "clk", _), DoPrim(AsClock, Seq(Reference("clock", _)), _, _)) => true
    })
  }
}
