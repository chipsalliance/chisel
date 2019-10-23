// See LICENSE for license details.

package firrtl.graph

import scala.collection.mutable

/**
  * An exception that indicates that an edge cannot be found in a graph with edge data.
  * 
  * @note the vertex type is not captured as a type parameter, as it would be erased.
  */
class EdgeNotFoundException(u: Any, v: Any)
    extends IllegalArgumentException(s"Edge (${u}, ${v}) does not exist!")

/**
  * Mixing this trait into a DiGraph indicates that each edge may be associated with an optional
  * data value. The EdgeData trait provides a minimal API for viewing edge data without mutation.
  *
  * @tparam V the vertex type (datatype) of the underlying DiGraph
  * @tparam E the type of each edge data value
  */
trait EdgeData[V, E] {
  this: DiGraph[V] =>
  protected val edgeDataMap: collection.Map[(V, V), E]

  protected def assertEdgeExists(u: V, v: V): Unit = {
    if (!contains(u) || !getEdges(u).contains(v)) {
      throw new EdgeNotFoundException(u, v)
    }
  }

  /**
    * @return the edge data associated with a given edge
    * @param u the source of the edge
    * @param v the destination of the edge
    * @throws EdgeNotFoundException if the edge does not exist
    * @throws NoSuchElementException if the edge has no data
    */
  def edgeData(u: V, v: V): E = {
    assertEdgeExists(u, v)
    edgeDataMap((u, v))
  }

  /**
    * Optionally return the edge data associated with a given edge.
    *
    * @return an option containing the edge data, if any, or None
    * @param u the source of the edge
    * @param v the destination of the edge
    */
  def getEdgeData(u: V, v: V): Option[E] = edgeDataMap.get((u, v))
}

/**
  * Mixing this trait into a DiGraph indicates that each edge may be associated with an optional
  * data value. The MutableEdgeData trait provides an API for viewing and mutating edge data.
  *
  * @tparam V the vertex type (datatype) of the underlying DiGraph
  * @tparam E the type of each edge data value
  */
trait MutableEdgeData[V, E] extends EdgeData[V, E] {
  this: MutableDiGraph[V] =>

  protected val edgeDataMap: mutable.Map[(V, V), E] = new mutable.LinkedHashMap[(V, V), E]

  /**
    * Associate an edge data value with a graph edge.
    *
    * @param u the source of the edge
    * @param v the destination of the edge
    * @param data the edge data to associate with the edge
    * @throws EdgeNotFoundException if the edge does not exist in the graph
    */
  def setEdgeData(u: V, v: V, data: E): Unit = {
    assertEdgeExists(u, v)
    edgeDataMap((u, v)) = data
  }

  /**
    * Add an edge (u,v) to the graph with associated edge data.
    *
    * @see [[DiGraph.addEdge]]
    * @param u the source of the edge
    * @param v the destination of the edge
    * @param data the edge data to associate with the edge
    * @throws IllegalArgumentException if u or v is not part of the graph
    */
  def addEdge(u: V, v: V, data: E): Unit = {
    addEdge(u, v)
    setEdgeData(u, v, data)
  }

  /**
    * Safely add an edge (u,v) to the graph with associated edge data. If on or more of the two
    * vertices is not present in the graph, add them before creating the edge.
    *
    * @see [[DiGraph.addPairWithEdge]]
    * @param u the source of the edge
    * @param v the destination of the edge
    * @param data the edge data to associate with the edge
    */
  def addPairWithEdge(u: V, v: V, data: E): Unit = {
    addPairWithEdge(u, v)
    setEdgeData(u, v, data)
  }

  /**
    * Safely add an edge (u,v) to the graph with associated edge data if and only if both vertices
    * are present in the graph. This is useful for preventing spurious edge creating when examining
    * a subset of possible nodes.
    *
    * @see [[DiGraph.addEdgeIfValid]]
    * @return a Boolean indicating whether the edge was added
    * @param u the source of the edge
    * @param v the destination of the edge
    * @param data the edge data to associate with the edge
    */
  def addEdgeIfValid(u: V, v: V, data: E): Boolean = {
    if (addEdgeIfValid(u, v)) {
      setEdgeData(u, v, data)
      true
    } else {
      false
    }
  }
}
