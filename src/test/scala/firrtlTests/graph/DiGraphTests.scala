package firrtlTests.graph

import java.io._
import org.scalatest._
import org.scalatest.prop._
import org.scalatest.Matchers._
import firrtl.graph._
import firrtlTests._

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

  acyclicGraph.findSCCs.filter(_.length > 1) shouldBe empty

  cyclicGraph.findSCCs.filter(_.length > 1) should not be empty

  acyclicGraph.path("a","e") should not be empty

  an [PathNotFoundException] should be thrownBy acyclicGraph.path("e","a")

  acyclicGraph.linearize.head should equal ("a")

  a [CyclicException] should be thrownBy cyclicGraph.linearize

  acyclicGraph.reverse.getEdgeMap should equal (reversedAcyclicGraph.getEdgeMap)

  degenerateGraph.getEdgeMap should equal (degenerateGraph.reverse.getEdgeMap)

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

}
