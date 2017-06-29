// See LICENSE for license details.

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl.Parser
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl.transforms._
import firrtl._

class LowerTypesSpec extends FirrtlFlatSpec {
  private val transforms = Seq(
    ToWorkingIR,
    CheckHighForm,
    ResolveKinds,
    InferTypes,
    CheckTypes,
    ResolveGenders,
    CheckGenders,
    InferWidths,
    CheckWidths,
    PullMuxes,
    ExpandConnects,
    RemoveAccesses,
    ExpandWhens,
    CheckInitialization,
    Legalize,
    new ConstantPropagation,
    ResolveKinds,
    InferTypes,
    ResolveGenders,
    InferWidths,
    LowerTypes)

  private def executeTest(input: String, expected: Seq[String]) = {
    val circuit = Parser.parse(input.split("\n").toIterator)
    val result = transforms.foldLeft(CircuitState(circuit, UnknownForm)) {
      (c: CircuitState, p: Transform) => p.runTransform(c)
    }
    val c = result.circuit
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
