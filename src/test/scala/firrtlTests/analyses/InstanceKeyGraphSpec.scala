// See LICENSE for license details.

package firrtlTests.analyses

import firrtl.analyses.InstanceKeyGraph
import firrtl.analyses.InstanceKeyGraph.InstanceKey
import firrtl.testutils.FirrtlFlatSpec

class InstanceKeyGraphSpec extends FirrtlFlatSpec {
  behavior of "InstanceKeyGraph"

  // Note that due to optimized implementations of Map1-4, at least 5 entries are needed to
  // experience non-determinism
  it should "preserve Module declaration order" in {
    val input = """
                  |circuit Top :
                  |  module Top :
                  |    inst c1 of Child1
                  |    inst c2 of Child2
                  |  module Child1 :
                  |    inst a of Child1a
                  |    inst b of Child1b
                  |    skip
                  |  module Child1a :
                  |    skip
                  |  module Child1b :
                  |    skip
                  |  module Child2 :
                  |    skip
                  |""".stripMargin
    val circuit = parse(input)
    val instGraph = new InstanceKeyGraph(circuit)
    val childMap = instGraph.getChildInstances
    childMap.map(_._1) should equal (Seq("Top", "Child1", "Child1a", "Child1b", "Child2"))
  }

  // Note that due to optimized implementations of Map1-4, at least 5 entries are needed to
  // experience non-determinism
  it should "preserve Instance declaration order" in {
    val input = """
                  |circuit Top :
                  |  module Top :
                  |    inst a of Child
                  |    inst b of Child
                  |    inst c of Child
                  |    inst d of Child
                  |    inst e of Child
                  |    inst f of Child
                  |  module Child :
                  |    skip
                  |""".stripMargin
    val circuit = parse(input)
    val instGraph = new InstanceKeyGraph(circuit)
    val childMap = instGraph.getChildInstances.toMap
    val insts = childMap("Top").map(_.name)
    insts should equal (Seq("a", "b", "c", "d", "e", "f"))
  }

  it should "compute a correct and deterministic module order" in {
    val input = """
                  |circuit Top :
                  |  module Top :
                  |    inst c1 of Child1
                  |    inst c2 of Child2
                  |    inst c4 of Child4
                  |    inst c3 of Child3
                  |  module Child1 :
                  |    inst a of Child1a
                  |    inst b of Child1b
                  |    skip
                  |  module Child1a :
                  |    skip
                  |  module Child1b :
                  |    skip
                  |  module Child2 :
                  |    skip
                  |  module Child3 :
                  |    skip
                  |  module Child4 :
                  |    skip
                  |""".stripMargin
    val circuit = parse(input)
    val instGraph = new InstanceKeyGraph(circuit)
    val order = instGraph.moduleOrder.map(_.name)
    // Where it has freedom, the instance declaration order will be reversed.
    order should equal (Seq("Top", "Child3", "Child4", "Child2", "Child1", "Child1b", "Child1a"))
  }

  it should "find hierarchical instances correctly in disconnected hierarchies" in {
    val input =
      """circuit Top :
        |  module Top :
        |    inst c of Child1
        |  module Child1 :
        |    skip
        |
        |  module Top2 :
        |    inst a of Child2
        |    inst b of Child3
        |    skip
        |  module Child2 :
        |    inst a of Child2a
        |    inst b of Child2b
        |    skip
        |  module Child2a :
        |    skip
        |  module Child2b :
        |    skip
        |  module Child3 :
        |    skip
        |""".stripMargin

    val circuit = parse(input)
    val iGraph = new InstanceKeyGraph(circuit)
    iGraph.findInstancesInHierarchy("Top") shouldBe Seq(Seq(InstanceKey("Top", "Top")))
    iGraph.findInstancesInHierarchy("Child1") shouldBe Seq(Seq(InstanceKey("Top", "Top"), InstanceKey("c", "Child1")))
    iGraph.findInstancesInHierarchy("Top2") shouldBe Nil
    iGraph.findInstancesInHierarchy("Child2") shouldBe Nil
    iGraph.findInstancesInHierarchy("Child2a") shouldBe Nil
    iGraph.findInstancesInHierarchy("Child2b") shouldBe Nil
    iGraph.findInstancesInHierarchy("Child3") shouldBe Nil
  }

}
