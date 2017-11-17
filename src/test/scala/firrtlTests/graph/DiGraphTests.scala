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

  val degenerateGraph = DiGraph(Map("a" -> Set.empty[String]))

  acyclicGraph.findSCCs.filter(_.length > 1) shouldBe empty

  cyclicGraph.findSCCs.filter(_.length > 1) should not be empty

  acyclicGraph.path("a","e") should not be empty

  an [PathNotFoundException] should be thrownBy acyclicGraph.path("e","a")

  acyclicGraph.linearize.head should equal ("a")

  a [CyclicException] should be thrownBy cyclicGraph.linearize

  acyclicGraph.reverse.getEdgeMap should equal (reversedAcyclicGraph.getEdgeMap)

  degenerateGraph.getEdgeMap should equal (degenerateGraph.reverse.getEdgeMap)

}
