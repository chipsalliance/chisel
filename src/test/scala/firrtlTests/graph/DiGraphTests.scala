// See LICENSE for license details.

package firrtlTests.graph

import firrtl.graph._
import firrtl.testutils._

class DiGraphTests extends FirrtlFlatSpec {

  val acyclicGraph = DiGraph(Map(
    "a" -> Set("b","c"),
    "b" -> Set("d"),
    "c" -> Set("d"),
    "d" -> Set("e"),
    "e" -> Set.empty[String]))

  val reversedAcyclicGraph = DiGraph(Map(
    "a" -> Set.empty[String],
    "b" -> Set("a"),
    "c" -> Set("a"),
    "d" -> Set("b", "c"),
    "e" -> Set("d")))

  val cyclicGraph = DiGraph(Map(
    "a" -> Set("b","c"),
    "b" -> Set("d"),
    "c" -> Set("d"),
    "d" -> Set("a")))

  val tupleGraph = DiGraph(Map(
    ("a", 0) -> Set(("b", 2)),
    ("a", 1) -> Set(("c", 3)),
    ("b", 2) -> Set.empty[(String, Int)],
    ("c", 3) -> Set.empty[(String, Int)]
  ))

  val degenerateGraph = DiGraph(Map("a" -> Set.empty[String]))

  "A graph without cycles" should "have NOT SCCs" in {
    acyclicGraph.findSCCs.filter(_.length > 1) shouldBe empty
  }

  "A graph with cycles" should "have SCCs" in {
    cyclicGraph.findSCCs.filter(_.length > 1) should not be empty
  }

  "Asking a DiGraph for a path that exists" should "work" in {
    acyclicGraph.path("a","e") should not be empty
  }

  "Asking a DiGraph for a path from one node to another with no path" should "error" in {
    an [PathNotFoundException] should be thrownBy acyclicGraph.path("e","a")
  }

  "The first element in a linearized graph with a single root node" should "be the root" in {
    acyclicGraph.linearize.head should equal ("a")
  }

  "A DiGraph with a cycle" should "error when linearized" in {
    a [CyclicException] should be thrownBy cyclicGraph.linearize
  }

  "CyclicExceptions" should "contain information about the cycle" in {
    val c = the [CyclicException] thrownBy {
      cyclicGraph.linearize
    }
    c.getMessage.contains("found at a") should be (true)
    c.node.asInstanceOf[String] should be ("a")
  }

  "Reversing a graph" should "reverse all of the edges" in {
    acyclicGraph.reverse.getEdgeMap should equal (reversedAcyclicGraph.getEdgeMap)
  }

  "Reversing a graph with no edges" should "equal the graph itself" in {
    degenerateGraph.getEdgeMap should equal (degenerateGraph.reverse.getEdgeMap)
  }

  "transformNodes" should "combine vertices that collide, not drop them" in {
    tupleGraph.transformNodes(_._1).getEdgeMap should contain ("a" -> Set("b", "c"))
  }

  "Graph summation" should "be order-wise equivalent to original" in {
    val first = acyclicGraph.subgraph(Set("a", "b", "c"))
    val second = acyclicGraph.subgraph(Set("b", "c", "d", "e"))

    (first + second).getEdgeMap should equal (acyclicGraph.getEdgeMap)
  }

  it should "be idempotent" in {
    val first = acyclicGraph.subgraph(Set("a", "b", "c"))
    val second = acyclicGraph.subgraph(Set("b", "c", "d", "e"))

    (first + second + second + second).getEdgeMap should equal (acyclicGraph.getEdgeMap)
  }

  "linearize" should "not cause a stack overflow on very large graphs" in {
    // Graph of 0 -> 1, 1 -> 2, etc.
    val N = 10000
    val edges = (1 to N).zipWithIndex.map({ case (n, idx) => idx -> Set(n)}).toMap
    val bigGraph = DiGraph(edges + (N -> Set.empty[Int]))
    bigGraph.linearize should be (0 to N)
  }

  it should "work on multi-rooted graphs" in {
    val graph = DiGraph(Map("a" -> Set[String](), "b" -> Set[String]()))
    graph.linearize.toSet should be (graph.getVertices)
  }

  "acyclic graph" should "be rendered" in {
    val acyclicGraph2 = DiGraph(Map(
      "a" -> Set("b","c"),
      "b" -> Set("d", "x", "z"),
      "c" -> Set("d", "x"),
      "d" -> Set("e", "k", "l"),
      "x" -> Set("e"),
      "z" -> Set("e", "j"),
      "j" -> Set("k", "l", "c"),
      "k" -> Set("l"),
      "l" -> Set("e"),
      "e" -> Set.empty[String]
    ))
    val render = new RenderDiGraph(acyclicGraph2)
    val dotLines = render.toDotRanked.split("\n")

    dotLines.count(s => s.contains("rank=same")) should be (4)
    dotLines.exists(s => s.contains(""""b" -> { "d" "x" "z" };""")) should be (true)
    dotLines.exists(s => s.contains("""rankdir="LR";""")) should be (true)
  }

  "subgraphs containing cycles" should "be rendered with loop edges in red, can override orientation" in {
    val cyclicGraph2 = DiGraph(Map(
      "a" -> Set("b","c"),
      "b" -> Set("d", "x", "z"),
      "c" -> Set("d", "x"),
      "d" -> Set("e", "k", "l"),
      "x" -> Set("e"),
      "z" -> Set("e", "j"),
      "j" -> Set("k", "l", "c"),
      "k" -> Set("l"),
      "l" -> Set("e"),
      "e" -> Set("c")
    ))
    val render = new RenderDiGraph(cyclicGraph2, rankDir = "TB")
    val dotLines = render.showOnlyTheLoopAsDot.split("\n")

    dotLines.count(s => s.contains("rank=same")) should be (4)
    dotLines.count(s => s.contains("""[color=red,penwidth=3.0];""")) should be (3)
    dotLines.exists(s => s.contains(""""d" -> "k";""")) should be (true)
    dotLines.exists(s => s.contains("""rankdir="TB";""")) should be (true)
  }

  "reachableFrom" should "omit the queried node if no self-path exists" in {
    degenerateGraph.reachableFrom("a") shouldBe empty
    acyclicGraph.reachableFrom("b") should contain theSameElementsAs Vector("d", "e")
  }

  "reachableFrom" should "include the queried node if it is included in a cycle" in {
    cyclicGraph.reachableFrom("b") should contain theSameElementsAs Vector("a", "b", "c", "d")
  }
}
