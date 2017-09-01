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
  private def getEdgeSet(graph: DiGraph[String]): Map[String, Set[String]] = {
    (graph.getVertices map {v => (v, graph.getEdges(v))}).toMap
  }

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
}
