// See LICENSE for license details.

package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl.Parser
import firrtl.ir.Circuit
import firrtl.passes._
import firrtl._
import firrtl.annotations._
import firrtl.annotations.TargetToken._
import firrtl.transforms.DontTouchAnnotation

class UniquifySpec extends FirrtlFlatSpec {

  private val transforms = Seq(
    ToWorkingIR,
    CheckHighForm,
    ResolveKinds,
    InferTypes,
    Uniquify
  )

  private def executeTest(input: String, expected: Seq[String]): Unit = executeTest(input, expected, Seq.empty, Seq.empty)
  private def executeTest(input: String, expected: Seq[String],
    inputAnnos: Seq[Annotation], expectedAnnos: Seq[Annotation]): Unit = {
    val circuit = Parser.parse(input.split("\n").toIterator)
    val result = transforms.foldLeft(CircuitState(circuit, UnknownForm, inputAnnos)) {
      (c: CircuitState, p: Transform) => p.runTransform(c)
    }
    val c = result.circuit
    val lines = c.serialize.split("\n") map normalized

    expected foreach { e =>
      lines should contain(e)
    }

    result.annotations.toSeq should equal(expectedAnnos)
  }

  behavior of "Uniquify"

  it should "rename colliding ports" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input a : { flip b : UInt<1>, c : { d : UInt<2>, flip e : UInt<3>}[2], c_1_e : UInt<4>}[2]
        |    output a_0_c_ : UInt<5>
        |    output a__0 : UInt<6>
      """.stripMargin
    val expected = Seq(
      "input a__ : { flip b : UInt<1>, c_ : { d : UInt<2>, flip e : UInt<3>}[2], c_1_e : UInt<4>}[2]",
      "output a_0_c_ : UInt<5>",
      "output a__0 : UInt<6>") map normalized

    val inputAnnos = Seq(DontTouchAnnotation(ReferenceTarget("Test", "Test", Seq.empty, "a", Seq(Index(0), Field("b")))),
      DontTouchAnnotation(ReferenceTarget("Test", "Test", Seq.empty, "a", Seq(Index(0), Field("c"), Index(0), Field("e")))))

    val expectedAnnos = Seq(DontTouchAnnotation(ReferenceTarget("Test", "Test", Seq.empty, "a__", Seq(Index(0), Field("b")))),
      DontTouchAnnotation(ReferenceTarget("Test", "Test", Seq.empty, "a__", Seq(Index(0), Field("c_"), Index(0), Field("e")))))

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
      "reg a__ : { b : UInt<1>, c_ : { d : UInt<2>, e : UInt<3>}[2], c_1_e : UInt<4>}[2], clock with :",
      "reg a_0_c_ : UInt<5>, clock with :",
      "reg a__0 : UInt<6>, clock with :") map normalized

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
    val expected = Seq("node a__ = x") map normalized

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
      "reg foo : UInt<4>, clock_[1] with :",
      "reset => (reset_.a, init_[3].b_[1].d)"
    ) map normalized

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
      "input data : { a : UInt<4>, b : UInt<4>}[2]",
      "node data_0_a_ = data[0].a"
    ) map normalized

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
      "node foo = data__.a",
      "node bar = data__.b[1]") map normalized

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
      "a_0_b <= a__[0].b",
      "a__[0].c_ <- a__0_c_") map normalized

    executeTest(input, expected)
  }

  it should "rename SubAccesses" in {
    val input =
     """circuit Test :
       |  module Test :
       |    input a : { b : UInt<1>, c : { d : UInt<2>, e : UInt<3>}[2], c_1_e : UInt<4>}[2]
       |    output a_0_b : UInt<2>
       |    input i : UInt<1>[2]
       |    output i_0 : UInt<1>
       |    a_0_b <= a.c[i[1]].d
     """.stripMargin
    val expected = Seq(
      "a_0_b <= a_.c_[i_[1]].d") map normalized

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
      "a_0_b <= mux(a__[UInt<1>(\"h0\")].c_1_e, or(a__[or(a__[0].b, a__[1].b)].b, xorr(a__[0].c_1_e)), orr(cat(a__0_c_[0].e, a__[1].c_1_e)))"
    ) map normalized

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
      "mem mem_ :",
      "node mem_0_b = mem_.read.data[0].b",
      "mem_.read.addr is invalid") map normalized

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
      "data-type => { a : UInt<8>, b_ : UInt<8>[2], b_0 : UInt<8>}",
      "node x = mem.read.data.b_[0]") map normalized

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
      "mod_.a_.b <= x",
      "mod_.a_.c <= x",
      "node mod_a_b = mod_.a_b") map normalized

    executeTest(input, expected)
  }
}
