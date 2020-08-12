// See LICENSE for license details.

package firrtlTests

import firrtl.Parser
import firrtl.passes._
import firrtl.transforms._
import firrtl._
import firrtl.annotations._
import firrtl.options.Dependency
import firrtl.stage.TransformManager
import firrtl.testutils._
import firrtl.util.TestOptions

/** Integration style tests for [[LowerTypes]].
  * You can find additional unit test style tests in [[passes.LowerTypesUnitTestSpec]]
  */
class LowerTypesSpec extends FirrtlFlatSpec {
  private val compiler = new TransformManager(Seq(Dependency(LowerTypes)))

  private def executeTest(input: String, expected: Seq[String]) = {
    val fir = Parser.parse(input.split("\n").toIterator)
    val c = compiler.runTransform(CircuitState(fir, Seq())).circuit
    val lines = c.serialize.split("\n") map normalized

    expected foreach { e =>
      lines should contain(e)
    }
  }

  behavior of "Lower Types"

  it should "lower ports" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input w : UInt<1>
        |    input x : {a : UInt<1>, b : UInt<1>}
        |    input y : UInt<1>[4]
        |    input z : { c : { d : UInt<1>, e : UInt<1>}, f : UInt<1>[2] }[2]
      """.stripMargin
    val expected = Seq("w", "x_a", "x_b", "y_0", "y_1", "y_2", "y_3", "z_0_c_d",
      "z_0_c_e", "z_0_f_0", "z_0_f_1", "z_1_c_d", "z_1_c_e", "z_1_f_0",
      "z_1_f_1") map (x => s"input $x : UInt<1>") map normalized

    executeTest(input, expected)
  }

  it should "lower mixed-direction ports" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input foo : {flip a : UInt<1>, b : UInt<1>}[1]
        |    foo is invalid
      """.stripMargin
    val expected = Seq(
      "output foo_0_a : UInt<1>",
      "input foo_0_b : UInt<1>"
    ) map normalized

    executeTest(input, expected)
  }

  it should "lower registers" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    input reset : UInt<1>
        |    reg w : UInt<1>, clock
        |    reg x : {a : UInt<1>, b : UInt<1>}, clock
        |    reg y : UInt<1>[4], clock
        |    reg z : { c : { d : UInt<1>, e : UInt<1>}, f : UInt<1>[2] }[2], clock
      """.stripMargin
    val expected = Seq("w", "x_a", "x_b", "y_0", "y_1", "y_2", "y_3", "z_0_c_d",
      "z_0_c_e", "z_0_f_0", "z_0_f_1", "z_1_c_d", "z_1_c_e", "z_1_f_0",
      "z_1_f_1") map (x => s"reg $x : UInt<1>, clock with :") map normalized

    executeTest(input, expected)
  }

  it should "lower registers with aggregate initialization" in {
    val input =
     """circuit Test :
       |  module Test :
       |    input clock : Clock
       |    input reset : UInt<1>
       |    input init : { a : UInt<1>, b : UInt<1>}[2]
       |    reg x : { a : UInt<1>, b : UInt<1>}[2], clock with :
       |      reset => (reset, init)
     """.stripMargin
    val expected = Seq(
      "reg x_0_a : UInt<1>, clock with :", "reset => (reset, init_0_a)",
      "reg x_0_b : UInt<1>, clock with :", "reset => (reset, init_0_b)",
      "reg x_1_a : UInt<1>, clock with :", "reset => (reset, init_1_a)",
      "reg x_1_b : UInt<1>, clock with :", "reset => (reset, init_1_b)"
    ) map normalized

    executeTest(input, expected)
  }

  it should "lower DefRegister expressions: clock, reset, and init" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock[2]
        |    input reset : { a : UInt<1>, b : UInt<1>}
        |    input init : { a : UInt<4>, b : { c : UInt<4>, d : UInt<4>}[2]}[4]
        |    reg foo : UInt<4>, clock[1], with :
        |      reset => (reset.a, init[3].b[1].d)
      """.stripMargin
    val expected = Seq(
      "reg foo : UInt<4>, clock_1 with :",
      "reset => (reset_a, init_3_b_1_d)"
    ) map normalized

    executeTest(input, expected)
  }

  it should "lower DefInstances (but not too far!)" in {
    val input =
     """circuit Test :
       |  module Other :
       |    input a : { b : UInt<1>, c : UInt<1>}
       |    output d : UInt<1>[2]
       |    d[0] <= a.b
       |    d[1] <= a.c
       |  module Test :
       |    input x : UInt<1>
       |    inst mod of Other
       |    mod.a.b <= x
       |    mod.a.c <= x
       |    node y = mod.d[0]
     """.stripMargin
    val expected = Seq(
      "mod.a_b <= x",
      "mod.a_c <= x",
      "node y = mod.d_0") map normalized

    executeTest(input, expected)
  }

  it should "lower aggregate memories" in {
    val input =
     """circuit Test :
       |  module Test :
       |    input clock : Clock
       |    mem m :
       |      data-type => { a : UInt<8>, b : UInt<8>}[2]
       |      depth => 32
       |      read-latency => 0
       |      write-latency => 1
       |      reader => read
       |      writer => write
       |    m.read.clk <= clock
       |    m.read.en <= UInt<1>(1)
       |    m.read.addr is invalid
       |    node x = m.read.data
       |    node y = m.read.data[0].b
       |
       |    m.write.clk <= clock
       |    m.write.en <= UInt<1>(0)
       |    m.write.mask is invalid
       |    m.write.addr is invalid
       |    wire w : { a : UInt<8>, b : UInt<8>}[2]
       |    w[0].a <= UInt<4>(2)
       |    w[0].b <= UInt<4>(3)
       |    w[1].a <= UInt<4>(4)
       |    w[1].b <= UInt<4>(5)
       |    m.write.data <= w

     """.stripMargin
    val expected = Seq(
      "mem m_0_a :", "mem m_0_b :", "mem m_1_a :", "mem m_1_b :",
      "m_0_a.read.clk <= clock", "m_0_b.read.clk <= clock",
      "m_1_a.read.clk <= clock", "m_1_b.read.clk <= clock",
      "m_0_a.read.addr is invalid", "m_0_b.read.addr is invalid",
      "m_1_a.read.addr is invalid", "m_1_b.read.addr is invalid",
      "node x_0_a = m_0_a.read.data", "node x_0_b = m_0_b.read.data",
      "node x_1_a = m_1_a.read.data", "node x_1_b = m_1_b.read.data",
      "m_0_a.write.mask is invalid", "m_0_b.write.mask is invalid",
      "m_1_a.write.mask is invalid", "m_1_b.write.mask is invalid",
      "m_0_a.write.data <= w_0_a", "m_0_b.write.data <= w_0_b",
      "m_1_a.write.data <= w_1_a", "m_1_b.write.data <= w_1_b"
    ) map normalized

    executeTest(input, expected)
  }
}

/** Uniquify used to be its own pass. We ported the tests to run with the combined LowerTypes pass. */
class LowerTypesUniquifySpec extends FirrtlFlatSpec {
  private val compiler = new TransformManager(Seq(Dependency(firrtl.passes.LowerTypes)))

  private def executeTest(input: String, expected: Seq[String]): Unit = executeTest(input, expected, Seq.empty, Seq.empty)
  private def executeTest(input: String, expected: Seq[String],
                          inputAnnos: Seq[Annotation], expectedAnnos: Seq[Annotation]): Unit = {
    val circuit = Parser.parse(input.split("\n").toIterator)
    val result = compiler.runTransform(CircuitState(circuit, inputAnnos))
    val lines = result.circuit.serialize.split("\n") map normalized

    expected.map(normalized).foreach { e =>
      assert(lines.contains(e), f"Failed to find $e in ${lines.mkString("\n")}")
    }

    result.annotations.toSeq should equal(expectedAnnos)
  }

  behavior of "LowerTypes"

  it should "rename colliding ports" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input a : { flip b : UInt<1>, c : { d : UInt<2>, flip e : UInt<3>}[2], c_1_e : UInt<4>}[2]
        |    output a_0_c_ : UInt<5>
        |    output a__0 : UInt<6>
      """.stripMargin
    val expected = Seq(
      "output a___0_b : UInt<1>",
      "input a___0_c__0_d : UInt<2>",
      "output a___0_c__0_e : UInt<3>",
      "output a_0_c_ : UInt<5>",
      "output a__0 : UInt<6>")

    val m = CircuitTarget("Test").module("Test")
    val inputAnnos = Seq(
      DontTouchAnnotation(m.ref("a").index(0).field("b")),
      DontTouchAnnotation(m.ref("a").index(0).field("c").index(0).field("e")))

    val expectedAnnos = Seq(
      DontTouchAnnotation(m.ref("a___0_b")),
      DontTouchAnnotation(m.ref("a___0_c__0_e")))


    executeTest(input, expected, inputAnnos, expectedAnnos)
  }

  it should "rename colliding registers" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    reg a : { b : UInt<1>, c : { d : UInt<2>, e : UInt<3>}[2], c_1_e : UInt<4>}[2], clock
        |    reg a_0_c_ : UInt<5>, clock
        |    reg a__0 : UInt<6>, clock
      """.stripMargin
    val expected = Seq(
      "reg a___0_b : UInt<1>, clock with :",
      "reg a___1_c__1_e : UInt<3>, clock with :",
      "reg a___0_c_1_e : UInt<4>, clock with :",
      "reg a_0_c_ : UInt<5>, clock with :",
      "reg a__0 : UInt<6>, clock with :")

    executeTest(input, expected)
  }

  it should "rename colliding nodes" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    reg x : { b : UInt<1>, c : { d : UInt<2>, e : UInt<3>}[2], c_1_e : UInt<4>}[2], clock
        |    node a = x
        |    node a_0_c_ = a[0].b
        |    node a__0  = a[1].c[0].d
      """.stripMargin
    val expected = Seq(
      "node a___0_b = x_0_b",
      "node a___1_c__1_e = x_1_c__1_e",
      "node a___1_c_1_e = x_1_c_1_e"
    )

    executeTest(input, expected)
  }


  it should "rename DefRegister expressions: clock, reset, and init" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock[2]
        |    input clock_0 : Clock
        |    input reset : { a : UInt<1>, b : UInt<1>}
        |    input reset_a : UInt<1>
        |    input init : { a : UInt<4>, b : { c : UInt<4>, d : UInt<4>}[2], b_1_c : UInt<4>}[4]
        |    input init_0_a : UInt<4>
        |    reg foo : UInt<4>, clock[1], with :
        |      reset => (reset.a, init[3].b[1].d)
      """.stripMargin
    val expected = Seq(
      "reg foo : UInt<4>, clock__1 with :",
      "reset => (reset__a, init__3_b__1_d)"
    )

    executeTest(input, expected)
  }

  it should "rename ports before statements" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input data : { a : UInt<4>, b : UInt<4>}[2]
        |    node data_0_a = data[0].a
      """.stripMargin
    val expected = Seq(
      "input data_0_a : UInt<4>",
      "input data_0_b : UInt<4>",
      "input data_1_a : UInt<4>",
      "input data_1_b : UInt<4>",
      "node data_0_a_ = data_0_a"
    )

    executeTest(input, expected)
  }

  it should "rename ports before statements (instance)" in {
    val input =
      """circuit Test :
        |  module Child:
        |    skip
        |  module Test :
        |    input data : { a : UInt<4>, b : UInt<4>}[2]
        |    inst data_0_a of Child
      """.stripMargin
    val expected = Seq(
      "input data_0_a : UInt<4>",
      "input data_0_b : UInt<4>",
      "input data_1_a : UInt<4>",
      "input data_1_b : UInt<4>",
      "inst data_0_a_ of Child"
    )

    executeTest(input, expected)
  }

  it should "rename ports before statements (mem)" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input data : { a : UInt<4>, b : UInt<4>}[2]
        |    mem data_0_a :
        |      data-type => UInt<1>
        |      depth => 32
        |      read-latency => 0
        |      write-latency => 1
        |      reader => read
        |      writer => write
      """.stripMargin
    val expected = Seq(
      "input data_0_a : UInt<4>",
      "input data_0_b : UInt<4>",
      "input data_1_a : UInt<4>",
      "input data_1_b : UInt<4>",
      "mem data_0_a_ :"
    )

    executeTest(input, expected)
  }

  it should "rename node expressions" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input data : { a : UInt<4>, b : UInt<4>[2]}
        |    input data_a : UInt<4>
        |    input data__b_1 : UInt<4>
        |    node foo = data.a
        |    node bar = data.b[1]
      """.stripMargin
    val expected = Seq(
      "node foo = data___a",
      "node bar = data___b_1")

    executeTest(input, expected)
  }

  it should "rename both side of connects" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input a : { b : UInt<1>, flip c : { d : UInt<2>, e : UInt<3>}[2], c_1_e : UInt<4>}[2]
        |    output a_0_b : UInt<1>
        |    input a__0_c_ : { d : UInt<2>, e : UInt<3>}[2]
        |    a_0_b <= a[0].b
        |    a[0].c <- a__0_c_
      """.stripMargin
    val expected = Seq(
      "a_0_b <= a___0_b",
      "a___0_c__0_d <= a__0_c__0_d",
      "a___0_c__0_e <= a__0_c__0_e",
      "a___0_c__1_d <= a__0_c__1_d",
      "a___0_c__1_e <= a__0_c__1_e"
    )

    executeTest(input, expected)
  }

  it should "rename deeply nested expressions" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input a : { b : UInt<1>, flip c : { d : UInt<2>, e : UInt<3>}[2], c_1_e : UInt<4>}[2]
        |    output a_0_b : UInt<1>
        |    input a__0_c_ : { d : UInt<2>, e : UInt<3>}[2]
        |    a_0_b <= mux(a[UInt(0)].c_1_e, or(a[or(a[0].b, a[1].b)].b, xorr(a[0].c_1_e)), orr(cat(a__0_c_[0].e, a[1].c_1_e)))
      """.stripMargin
    val expected = Seq(
      "a_0_b <= mux(a___0_c_1_e, or(_a_or_b, xorr(a___0_c_1_e)), orr(cat(a__0_c__0_e, a___1_c_1_e)))"
    )

    executeTest(input, expected)
  }

  it should "rename memories" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    mem mem :
        |      data-type => { a : UInt<8>, b : UInt<8>[2]}[2]
        |      depth => 32
        |      read-latency => 0
        |      write-latency => 1
        |      reader => read
        |      writer => write
        |    node mem_0_b = mem.read.data[0].b
        |
        |    mem.read.addr is invalid
        |    mem.read.en <= UInt(1)
        |    mem.read.clk <= clock
        |    mem.write.data is invalid
        |    mem.write.mask is invalid
        |    mem.write.addr is invalid
        |    mem.write.en <= UInt(0)
        |    mem.write.clk <= clock
      """.stripMargin
    val expected = Seq(
      "mem mem__0_b_0 :",
      "node mem_0_b_0 = mem__0_b_0.read.data",
      "node mem_0_b_1 = mem__0_b_1.read.data",
      "mem__0_b_0.read.addr is invalid")

    executeTest(input, expected)
  }

  it should "rename aggregate typed memories" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input clock : Clock
        |    mem mem :
        |      data-type => { a : UInt<8>, b : UInt<8>[2], b_0 : UInt<8> }
        |      depth => 32
        |      read-latency => 0
        |      write-latency => 1
        |      reader => read
        |      writer => write
        |    node x = mem.read.data.b[0]
        |
        |    mem.read.addr is invalid
        |    mem.read.en <= UInt(1)
        |    mem.read.clk <= clock
        |    mem.write.data is invalid
        |    mem.write.mask is invalid
        |    mem.write.addr is invalid
        |    mem.write.en <= UInt(0)
        |    mem.write.clk <= clock
      """.stripMargin
    val expected = Seq(
      "mem mem_a :",
      "mem mem_b__0 :",
      "mem mem_b__1 :",
      "mem mem_b_0 :",
      "node x = mem_b__0.read.data")

    executeTest(input, expected)
  }

  it should "rename instances and their ports" in {
    val input =
      """circuit Test :
        |  module Other :
        |    input a : { b : UInt<4>, c : UInt<4> }
        |    output a_b : UInt<4>
        |    a_b <= a.b
        |
        |  module Test :
        |    node x = UInt(6)
        |    inst mod of Other
        |    mod.a.b <= x
        |    mod.a.c <= x
        |    node mod_a_b = mod.a_b
     """.stripMargin
    val expected = Seq(
      "inst mod_ of Other",
      "mod_.a__b <= x",
      "mod_.a__c <= x",
      "node mod_a_b = mod_.a_b")

    executeTest(input, expected)
  }

  it should "quickly rename deep bundles" in {
    val depth = 500
    // We previously used a fixed time to determine if this test passed or failed.
    // This test would pass under normal conditions, but would fail during coverage tests.
    // Instead of using a fixed time, we run the test once (with a rename depth of 1), and record the time,
    //  then run it again with a depth of 500 and verify that the difference is below a fixed threshold.
    // Additionally, since executions times vary significantly under coverage testing, we check a global
    //  to see if timing measurements are accurate enough to enforce the timing checks.
    val threshold = depth * 2.0
    // As of 20-Feb-2019, this still fails occasionally:
    //  [info]   9038.99351 was not less than 6113.865 (UniquifySpec.scala:317)
    // Run the "quick" test three times and choose the longest time as the basis.
    val nCalibrationRuns = 3
    def mkType(i: Int): String = {
      if(i == 0) "UInt<8>" else s"{x: ${mkType(i - 1)}}"
    }
    val timesMs = (
      for (depth <- (List.fill(nCalibrationRuns)(1) :+ depth)) yield {
        val input = s"""circuit Test:
                       |  module Test :
                       |    input in: ${mkType(depth)}
                       |    output out: ${mkType(depth)}
                       |    out <= in
                       |""".stripMargin
        val (ms, _) = Utils.time(compileToVerilog(input))
        ms
      }
      ).toArray
    // The baseMs will be the maximum of the first calibration runs
    val baseMs = timesMs.slice(0, nCalibrationRuns - 1).max
    val renameMs = timesMs(nCalibrationRuns)
    if (TestOptions.accurateTiming)
      renameMs shouldBe < (baseMs * threshold)
  }
}

