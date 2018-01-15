package firrtlTests.graph

import firrtl.graph._
import firrtlTests._

class EulerTourTests extends FirrtlFlatSpec {

  val top = "top"
  val first_layer = Set("1a", "1b", "1c")
  val second_layer = Set("2a", "2b", "2c")
  val third_layer = Set("3a", "3b", "3c")
  val last_null = Set.empty[String]

  val m = Map(top -> first_layer) ++ first_layer.map{
    case x => Map(x -> second_layer) }.flatten.toMap ++ second_layer.map{
    case x => Map(x -> third_layer) }.flatten.toMap ++ third_layer.map{
    case x => Map(x -> last_null) }.flatten.toMap

  val graph = DiGraph(m)
  val instances = graph.pathsInDAG(top).values.flatten
  val tour = EulerTour(graph, top)

  it should "show equivalency of Berkman--Vishkin and naive RMQs" in {
    instances.toSeq.combinations(2).toList.map { case Seq(a, b) =>
      tour.rmqNaive(a, b) should be (tour.rmqBV(a, b))
    }
  }

  it should "determine naive RMQs of itself correctly" in {
    instances.toSeq.map { case a => tour.rmqNaive(a, a) should be (a) }
  }

  it should "determine Berkman--Vishkin RMQs of itself correctly" in {
    instances.toSeq.map { case a => tour.rmqNaive(a, a) should be (a) }
  }
}
