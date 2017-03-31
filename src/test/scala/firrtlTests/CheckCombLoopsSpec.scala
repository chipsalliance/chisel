// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.passes._
import firrtl.Mappers._
import annotations._

class CheckCombLoopsSpec extends SimpleTransformSpec {

  def emitter = new LowFirrtlEmitter

  def transforms = Seq(
    new ChirrtlToHighFirrtl,
    new IRToWorkingIR,
    new ResolveAndCheck,
    new HighFirrtlToMiddleFirrtl,
    new MiddleFirrtlToLowFirrtl
  )

  "Simple combinational loop" should "throw an exception" in {
    val input = """circuit hasloops :
                   |  module hasloops :
                   |    input clk : Clock
                   |    input a : UInt<1>
                   |    input b : UInt<1>
                   |    output c : UInt<1>
                   |    output d : UInt<1>
                   |    wire y : UInt<1>
                   |    wire z : UInt<1>
                   |    c <= b
                   |    z <= y
                   |    y <= z
                   |    d <= z
                   |""".stripMargin

    val writer = new java.io.StringWriter
    intercept[CheckCombLoops.CombLoopException] {
      compile(CircuitState(parse(input), ChirrtlForm, None), writer)
    }
  }

  "Node combinational loop" should "throw an exception" in {
    val input = """circuit hasloops :
                   |  module hasloops :
                   |    input clk : Clock
                   |    input a : UInt<1>
                   |    input b : UInt<1>
                   |    output c : UInt<1>
                   |    output d : UInt<1>
                   |    wire y : UInt<1>
                   |    c <= b
                   |    node z = and(c,y)
                   |    y <= z
                   |    d <= z
                   |""".stripMargin

    val writer = new java.io.StringWriter
    intercept[CheckCombLoops.CombLoopException] {
      compile(CircuitState(parse(input), ChirrtlForm, None), writer)
    }
  }

  "Combinational loop through a combinational memory read port" should "throw an exception" in {
    val input = """circuit hasloops :
                   |  module hasloops :
                   |    input clk : Clock
                   |    input a : UInt<1>
                   |    input b : UInt<1>
                   |    output c : UInt<1>
                   |    output d : UInt<1>
                   |    wire y : UInt<1>
                   |    wire z : UInt<1>
                   |    c <= b
                   |    mem m :
                   |      data-type => UInt<1>
                   |      depth => 2
                   |      read-latency => 0
                   |      write-latency => 1
                   |      reader => r
                   |      read-under-write => undefined
                   |    m.r.clk <= clk
                   |    m.r.addr <= y
                   |    m.r.en <= UInt(1)
                   |    z <= m.r.data
                   |    y <= z
                   |    d <= z
                   |""".stripMargin

    val writer = new java.io.StringWriter
    intercept[CheckCombLoops.CombLoopException] {
      compile(CircuitState(parse(input), ChirrtlForm, None), writer)
    }
  }

  "Combination loop through an instance" should "throw an exception" in {
    val input = """circuit hasloops :
                   |  module thru :
                   |    input in : UInt<1>
                   |    output out : UInt<1>
                   |    out <= in
                   |  module hasloops :
                   |    input clk : Clock
                   |    input a : UInt<1>
                   |    input b : UInt<1>
                   |    output c : UInt<1>
                   |    output d : UInt<1>
                   |    wire y : UInt<1>
                   |    wire z : UInt<1>
                   |    c <= b
                   |    inst inner of thru
                   |    inner.in <= y
                   |    z <= inner.out
                   |    y <= z
                   |    d <= z
                   |""".stripMargin

    val writer = new java.io.StringWriter
    intercept[CheckCombLoops.CombLoopException] {
      compile(CircuitState(parse(input), ChirrtlForm, None), writer)
    }
  }

  "Multiple simple loops in one SCC" should "throw an exception" in {
    val input = """circuit hasloops :
                   |  module hasloops :
                   |    input i : UInt<1>
                   |    output o : UInt<1>
                   |    wire a : UInt<1>
                   |    wire b : UInt<1>
                   |    wire c : UInt<1>
                   |    wire d : UInt<1>
                   |    wire e : UInt<1>
                   |    a <= and(c,i)
                   |    b <= and(a,d)
                   |    c <= b
                   |    d <= and(c,e)
                   |    e <= b
                   |    o <= e
                   |""".stripMargin

    val writer = new java.io.StringWriter
    intercept[CheckCombLoops.CombLoopException] {
      compile(CircuitState(parse(input), ChirrtlForm, None), writer)
    }
  }

}
