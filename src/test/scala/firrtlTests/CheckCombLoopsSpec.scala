// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.passes._
import firrtl.transforms._
import firrtl.Mappers._
import annotations._
import java.io.File
import java.nio.file.Paths

class CheckCombLoopsSpec extends SimpleTransformSpec {

  def emitter = new LowFirrtlEmitter

  def transforms = Seq(
    new ChirrtlToHighFirrtl,
    new IRToWorkingIR,
    new ResolveAndCheck,
    new HighFirrtlToMiddleFirrtl,
    new MiddleFirrtlToLowFirrtl
  )

  "Loop-free circuit" should "not throw an exception" in {
    val input = """circuit hasnoloops :
                   |  module thru :
                   |    input in1 : UInt<1>
                   |    input in2 : UInt<1>
                   |    output out1 : UInt<1>
                   |    output out2 : UInt<1>
                   |    out1 <= in1
                   |    out2 <= in2
                   |  module hasnoloops :
                   |    input clk : Clock
                   |    input a : UInt<1>
                   |    output b : UInt<1>
                   |    wire x : UInt<1>
                   |    inst inner of thru
                   |    inner.in1 <= a
                   |    x <= inner.out1
                   |    inner.in2 <= x
                   |    b <= inner.out2
                   |""".stripMargin

    val writer = new java.io.StringWriter
    compile(CircuitState(parse(input), ChirrtlForm), writer)
  }

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
      compile(CircuitState(parse(input), ChirrtlForm), writer)
    }
  }

  "Single-element combinational loop" should "throw an exception" in {
    val input = """circuit loop :
                   |  module loop :
                   |    output y : UInt<8>
                   |    wire w : UInt<8>
                   |    w <= w
                   |    y <= w
                   |""".stripMargin

    val writer = new java.io.StringWriter
    intercept[CheckCombLoops.CombLoopException] {
      compile(CircuitState(parse(input), ChirrtlForm), writer)
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
      compile(CircuitState(parse(input), ChirrtlForm), writer)
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
      compile(CircuitState(parse(input), ChirrtlForm), writer)
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
      compile(CircuitState(parse(input), ChirrtlForm), writer)
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
      compile(CircuitState(parse(input), ChirrtlForm), writer)
    }
  }

  "Circuit" should "create an annotation" in {
    val input = """circuit hasnoloops :
                  |  module thru :
                  |    input in1 : UInt<1>
                  |    input in2 : UInt<1>
                  |    output out1 : UInt<1>
                  |    output out2 : UInt<1>
                  |    out1 <= in1
                  |    out2 <= in2
                  |  module hasnoloops :
                  |    input clk : Clock
                  |    input a : UInt<1>
                  |    output b : UInt<1>
                  |    wire x : UInt<1>
                  |    inst inner of thru
                  |    inner.in1 <= a
                  |    x <= inner.out1
                  |    inner.in2 <= x
                  |    b <= inner.out2
                  |""".stripMargin

    val writer = new java.io.StringWriter
    val cs = compile(CircuitState(parse(input), ChirrtlForm), writer)
    val mn = ModuleName("hasnoloops", CircuitName("hasnoloops"))
    cs.annotations.collect {
      case c @ CombinationalPath(ComponentName("b", `mn`), Seq(ComponentName("a", `mn`))) => c
    }.nonEmpty should be (true)
  }
}

class CheckCombLoopsCommandLineSpec extends FirrtlFlatSpec {

  val testDir = createTestDirectory("CombLoopChecker")
  val inputFile = Paths.get(getClass.getResource("/features/HasLoops.fir").toURI()).toFile()
  val outFile = new File(testDir, "HasLoops.v")
  val args = Array("-i", inputFile.getAbsolutePath, "-o", outFile.getAbsolutePath, "-X", "verilog")

  "Combinational loops detection" should "run by default" in {
    a [CheckCombLoops.CombLoopException] should be thrownBy {
      firrtl.Driver.execute(args)
    }
  }

  it should "not run when given --no-check-comb-loops option" in {
    firrtl.Driver.execute(args :+ "--no-check-comb-loops")
  }
}
