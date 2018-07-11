package firrtlTests.analyses

import java.io._
import org.scalatest._
import org.scalatest.prop._
import org.scalatest.Matchers._
import firrtl.analyses.InstanceGraph
import firrtl.graph.DiGraph
import firrtl.Parser.parse
import firrtl.passes._
import firrtlTests._

class InstanceGraphTests extends FirrtlFlatSpec {
  private def getEdgeSet(graph: DiGraph[String]): collection.Map[String, collection.Set[String]] = {
    (graph.getVertices map {v => (v, graph.getEdges(v))}).toMap
  }

  behavior of "InstanceGraph"

  it should "recognize a simple hierarchy" in {
    val input = """
circuit Top :
  module Top :
    inst c1 of Child1
    inst c2 of Child2
  module Child1 :
    inst a of Child1a
    inst b of Child1b
    skip
  module Child1a :
    skip
  module Child1b :
    skip
  module Child2 :
    skip
"""
    val circuit = ToWorkingIR.run(parse(input))
    val graph = new InstanceGraph(circuit).graph.transformNodes(_.module)
    getEdgeSet(graph) shouldBe Map("Top" -> Set("Child1", "Child2"), "Child1" -> Set("Child1a", "Child1b"), "Child2" -> Set(), "Child1a" -> Set(), "Child1b" -> Set())
  }

  it should "recognize disconnected hierarchies" in {
    val input = """
circuit Top :
  module Top :
    inst c of Child1
  module Child1 :
    skip

  module Top2 :
    inst a of Child2
    inst b of Child3
    skip
  module Child2 :
    inst a of Child2a
    inst b of Child2b
    skip
  module Child2a :
    skip
  module Child2b :
    skip
  module Child3 :
    skip

"""
    val circuit = ToWorkingIR.run(parse(input))
    val graph = new InstanceGraph(circuit).graph.transformNodes(_.module)
    getEdgeSet(graph) shouldBe Map("Top" -> Set("Child1"), "Top2" -> Set("Child2", "Child3"), "Child2" -> Set("Child2a", "Child2b"), "Child1" -> Set(), "Child2a" -> Set(), "Child2b" -> Set(), "Child3" -> Set())
  }

  it should "not drop duplicate nodes when they collide as a result of transformNodes" in {
    val input =
"""circuit Top :
  module Buzz :
    skip
  module Fizz :
    inst b of Buzz
  module Foo :
    inst f1 of Fizz
  module Bar :
    inst f2 of Fizz
  module Top :
    inst f of Foo
    inst b of Bar
"""
    val circuit = ToWorkingIR.run(parse(input))
    val graph = (new InstanceGraph(circuit)).graph

    // Create graphs with edges from child to parent module
    // g1 has collisions on parents to children, ie. it combines:
    //   (f1, Fizz) -> (b, Buzz) and (f2, Fizz) -> (b, Buzz)
    val g1 = graph.transformNodes(_.module).reverse
    g1.getEdges("Fizz") shouldBe Set("Foo", "Bar")

    val g2 = graph.reverse.transformNodes(_.module)
    // g2 combines
    //   (f1, Fizz) -> (f, Foo) and (f2, Fizz) -> (b, Bar)
    g2.getEdges("Fizz") shouldBe Set("Foo", "Bar")
  }

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
    val circuit = ToWorkingIR.run(parse(input))
    val instGraph = new InstanceGraph(circuit)
    val childMap = instGraph.getChildrenInstances
    childMap.keys.toSeq should equal (Seq("Top", "Child1", "Child1a", "Child1b", "Child2"))
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
    val circuit = ToWorkingIR.run(parse(input))
    val instGraph = new InstanceGraph(circuit)
    val childMap = instGraph.getChildrenInstances
    val insts = childMap("Top").toSeq.map(_.name)
    insts should equal (Seq("a", "b", "c", "d", "e", "f"))
  }

  // Note that due to optimized implementations of Map1-4, at least 5 entries are needed to
  // experience non-determinism
  it should "have defined fullHierarchy order" in {
    val input = """
      |circuit Top :
      |  module Top :
      |    inst a of Child
      |    inst b of Child
      |    inst c of Child
      |    inst d of Child
      |    inst e of Child
      |  module Child :
      |    skip
      |""".stripMargin
    val circuit = ToWorkingIR.run(parse(input))
    val instGraph = new InstanceGraph(circuit)
    val hier = instGraph.fullHierarchy
    hier.keys.toSeq.map(_.name) should equal (Seq("Top", "a", "b", "c", "d", "e"))
  }
}
