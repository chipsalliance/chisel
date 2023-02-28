// SPDX-License-Identifier: Apache-2.0

package firrtlTests.analyses

import firrtl.{ChirrtlForm, CircuitState, FileUtils, IRToWorkingIR, UnknownForm}
import firrtl.analyses.{CircuitGraph, ConnectionGraph}
import firrtl.annotations.ModuleTarget
import firrtl.options.Dependency
import firrtl.passes.ExpandWhensAndCheck
import firrtl.stage.{Forms, TransformManager}
import firrtl.testutils.FirrtlFlatSpec

class ConnectionGraphSpec extends FirrtlFlatSpec {

  "ConnectionGraph" should "build connection graph for rocket-chip" in {
    ConnectionGraph(
      new firrtl.stage.transforms.Compiler(Seq(Dependency[ExpandWhensAndCheck]))
        .runTransform(
          CircuitState(parse(FileUtils.getTextResource("/regress/RocketCore.fir")), UnknownForm)
        )
        .circuit
    )
  }

  val input =
    """circuit Test:
      |  module Test :
      |    input in: UInt<8>
      |    input clk: Clock
      |    input reset: UInt<1>
      |    output out: {a: UInt<8>, b: UInt<8>[2]}
      |    out is invalid
      |    reg r: UInt<8>, clk with:
      |      (reset => (reset, UInt(0)))
      |    r <= in
      |    node x = r
      |    wire y: UInt<8>
      |    y <= x
      |    out.b[0] <= and(y, asUInt(SInt(-1)))
      |    inst child of Child
      |    child.in <= in
      |    out.a <= child.out
      |  module Child:
      |    input in: UInt<8>
      |    output out: UInt<8>
      |    out <= in
      |""".stripMargin

  val circuit = new firrtl.stage.transforms.Compiler(Seq(Dependency[ExpandWhensAndCheck]))
    .runTransform(
      CircuitState(parse(input), UnknownForm)
    )
    .circuit

  "ConnectionGraph" should "work with pathsInDAG" in {
    val Test = ModuleTarget("Test", "Test")
    val irGraph = ConnectionGraph(circuit)

    val paths = irGraph.pathsInDAG(Test.ref("in"))
    paths(Test.ref("out").field("b").index(0)) shouldBe Seq(
      Seq(
        Test.ref("in"),
        Test.ref("r"),
        Test.ref("x"),
        Test.ref("y"),
        Test.ref("@and#0"),
        Test.ref("out").field("b").index(0)
      )
    )
    paths(Test.ref("out").field("a")) shouldBe Seq(
      Seq(
        Test.ref("in"),
        Test.ref("child").field("in"),
        Test.instOf("child", "Child").ref("in"),
        Test.instOf("child", "Child").ref("out"),
        Test.ref("child").field("out"),
        Test.ref("out").field("a")
      )
    )

  }

  "ConnectionGraph" should "work with path" in {
    val Test = ModuleTarget("Test", "Test")
    val irGraph = ConnectionGraph(circuit)

    irGraph.path(Test.ref("in"), Test.ref("out").field("b").index(0)) shouldBe Seq(
      Test.ref("in"),
      Test.ref("r"),
      Test.ref("x"),
      Test.ref("y"),
      Test.ref("@and#0"),
      Test.ref("out").field("b").index(0)
    )

    irGraph.path(Test.ref("in"), Test.ref("out").field("a")) shouldBe Seq(
      Test.ref("in"),
      Test.ref("child").field("in"),
      Test.instOf("child", "Child").ref("in"),
      Test.instOf("child", "Child").ref("out"),
      Test.ref("child").field("out"),
      Test.ref("out").field("a")
    )

    irGraph.path(Test.ref("@invalid#0"), Test.ref("out").field("b").index(1)) shouldBe Seq(
      Test.ref("@invalid#0"),
      Test.ref("out").field("b").index(1)
    )
  }

}
