package firrtl.graph

import scala.collection.immutable.{Set, Map, HashSet, HashMap}
import scala.collection.mutable
import scala.collection.mutable.MultiMap

/** Represents common behavior of all directed graphs */
trait DiGraphLike[T] {
  /** Check whether the graph contains vertex v */
  def contains(v: T): Boolean

  /** Get all vertices in the graph
    * @return a Set[T] of all vertices in the graph
    */
  def getVertices: collection.Set[T]

  /** Get all edges of a node
    * @param v the specified node
    * @return a Set[T] of all vertices that v has edges to
    */
  def getEdges(v: T): collection.Set[T]
}

/** A class to represent a mutable directed graph with nodes of type T
  * 
  * @constructor Create a new graph with the provided edge data
  * @param edges a mutable.MultiMap[T,T] of edge data
  * 
  * For the edge data MultiMap, the values associated with each vertex
  * u in the graph are the vertices with inedges from u
  */
class MutableDiGraph[T](
  private[graph] val edgeData: MultiMap[T,T] =
    new mutable.HashMap[T, mutable.Set[T]] with MultiMap[T, T]) extends DiGraphLike[T] {

  // Inherited methods from DiGraphLike
  def contains(v: T) = edgeData.contains(v)
  def getVertices = edgeData.keySet
  def getEdges(v: T) = edgeData(v)

  /** Add vertex v to the graph
    * @return v, the added vertex
    */
  def addVertex(v: T): T = {
    edgeData.getOrElseUpdate(v,new mutable.HashSet[T])
    v
  }

  /** Add edge (u,v) to the graph */
  def addEdge(u: T, v: T) = {
    // Add v to keys to maintain invariant that all vertices are keys
    // of edge data
    edgeData.getOrElseUpdate(v, new mutable.HashSet[T])
    edgeData.addBinding(u,v)
  }
}

/** A companion to create immutable DiGraphs from mutable data */
object DiGraph {
  /** Create a DiGraph from a MutableDigraph, representing the same graph */
  def apply[T](mdg: MutableDiGraph[T]): DiGraph[T] =
    new DiGraph((mdg.edgeData mapValues { _.toSet }).toMap[T, Set[T]])

  /** Create a DiGraph from a MultiMap[T] of edge data */
  def apply[T](edgeData: MultiMap[T,T]): DiGraph[T] =
    new DiGraph((edgeData mapValues { _.toSet }).toMap[T, Set[T]])

  /** Create a DiGraph from a Map[T,Set[T]] of edge data */
  def apply[T](edgeData: Map[T,Set[T]]) = new DiGraph(edgeData)
}

/**
  * A class to represent an immutable directed graph with nodes of
  * type T
  * 
  * @constructor Create a new graph with the provided edge data
  * @param edges a Map[T,Set[T]] of edge data
  * 
  * For the edge data Map, the value associated with each vertex u in
  * the graph is a Set[T] of nodes where for each node v in the set,
  * the directed edge (u,v) exists in the graph.
  */
class DiGraph[T] (val edges: Map[T, Set[T]]) extends DiGraphLike[T] {

  /** An exception that is raised when an assumed DAG has a cycle */
  class CyclicException extends Exception("No valid linearization for cyclic graph")
  /** An exception that is raised when attempting to find an unreachable node */
  class PathNotFoundException extends Exception("Unreachable node")

  // Inherited methods from DiGraphLike
  def contains(v: T) = edges.contains(v)
  def getVertices = edges.keySet
  def getEdges(v: T) = edges.getOrElse(v, new HashSet[T])

  /** Find all sources in the graph
    * 
    * @return a Set[T] of source nodes
    */
  def findSources: Set[T] = edges.keySet -- edges.values.flatten.toSet

  /** Find all sinks in the graph
    * 
    * @return a Set[T] of sink nodes
    */
  def findSinks: Set[T] = reverse.findSources

  /** Linearizes (topologically sorts) a DAG
    * 
    * @param root the start node
    * @throws CyclicException if the graph is cyclic
    * @return a Map[T,T] from each visited node to its predecessor in the
    * traversal
    */
  def linearize: Seq[T] = {
    // permanently marked nodes are implicitly held in order
    val order = new mutable.ArrayBuffer[T]
    // invariant: no intersection between unmarked and tempMarked
    val unmarked = new mutable.HashSet[T]
    val tempMarked = new mutable.HashSet[T]

    def visit(n: T): Unit = {
      if (tempMarked.contains(n)) {
        throw new CyclicException
      }
      if (unmarked.contains(n)) {
        tempMarked += n
        unmarked -= n
        for (m <- getEdges(n)) {
          visit(m)
        }
        tempMarked -= n
        order.append(n)
      }
    }

    unmarked ++= getVertices
    while (!unmarked.isEmpty) {
      visit(unmarked.head)
    }

    // visited nodes are in post-traversal order, so must be reversed
    order.reverse.toSeq
  }

  /** Performs breadth-first search on the directed graph
    * 
    * @param root the start node
    * @return a Map[T,T] from each visited node to its predecessor in the
    * traversal
    */
  def BFS(root: T): Map[T,T] = {
    val prev = new mutable.HashMap[T,T]
    val queue = new mutable.Queue[T]
    queue.enqueue(root)
    while (!queue.isEmpty) {
      val u = queue.dequeue
      for (v <- getEdges(u)) {
        if (!prev.contains(v)) {
          prev(v) = u
          queue.enqueue(v)
        }
      }
    }
    prev.toMap
  }

  /** Finds the set of nodes reachable from a particular node
    * 
    * @param root the start node
    * @return a Set[T] of nodes reachable from the root
    */
  def reachableFrom(root: T): Set[T] = BFS(root).keys.toSet

  /** Finds a path (if one exists) from one node to another
    * 
    * @param start the start node
    * @param end the destination node
    * @throws PathNotFoundException
    * @return a Seq[T] of nodes defining an arbitrary valid path
    */
  def path(start: T, end: T) = {
    val nodePath = new mutable.ArrayBuffer[T]
    val prev = BFS(start)
    nodePath += end
    while (nodePath.last != start && prev.contains(nodePath.last)) {
      nodePath += prev(nodePath.last)
    }
    if (nodePath.last != start) {
      throw new PathNotFoundException
    }
    nodePath.toSeq.reverse
  }

  /** Finds the strongly connected components in the graph
    * 
    * @return a Seq of Seq[T], each containing nodes of an SCC in traversable order
    */
  def findSCCs: Seq[Seq[T]] = {
    var counter: BigInt = 0
    val stack = new mutable.Stack[T]
    val onstack = new mutable.HashSet[T]
    val indices = new mutable.HashMap[T, BigInt]
    val lowlinks = new mutable.HashMap[T, BigInt]
    val sccs = new mutable.ArrayBuffer[Seq[T]]

    /*
     * Recursive code is transformed to iterative code by representing
     * call stack info in an explicit structure. Here, the stack data
     * consists of the current vertex, its currently active edge, and
     * the position in the function. Because there is only one
     * recursive call site, remembering whether a child call was
     * created on the last iteration where the current frame was
     * active is sufficient to track the position.
     */
    class StrongConnectFrame[T](val v: T, val edgeIter: Iterator[T], var childCall: Option[T] = None)
    val callStack = new mutable.Stack[StrongConnectFrame[T]]

    for (node <- getVertices) {
      callStack.push(new StrongConnectFrame(node,getEdges(node).iterator))
      while (!callStack.isEmpty) {
        val frame = callStack.top
        val v = frame.v
        frame.childCall match {
          case None =>
            indices(v) = counter
            lowlinks(v) = counter
            counter = counter + 1
            stack.push(v)
            onstack += v
          case Some(w) =>
            lowlinks(v) = lowlinks(v).min(lowlinks(w))
        }
        frame.childCall = None
        while (frame.edgeIter.hasNext && frame.childCall.isEmpty) {
          val w = frame.edgeIter.next
          if (!indices.contains(w)) {
            frame.childCall = Some(w)
            callStack.push(new StrongConnectFrame(w,getEdges(w).iterator))
          } else if (onstack.contains(w)) {
            lowlinks(v) = lowlinks(v).min(indices(w))
          }
        }
        if (frame.childCall.isEmpty) {
          if (lowlinks(v) == indices(v)) {
            val scc = new mutable.ArrayBuffer[T]
            do {
              val w = stack.pop
              onstack -= w
              scc += w
            }
            while (scc.last != v);
            sccs.append(scc.toSeq)
          }
          callStack.pop
        }
      }
    }

    sccs.toSeq
  }

  /** Finds all paths starting at a particular node in a DAG
    * 
    * WARNING: This is an exponential time algorithm (as any algorithm
    * must be for this problem), but is useful for flattening circuit
    * graph hierarchies. Each path is represented by a Seq[T] of nodes
    * in a traversable order.
    * 
    * @param start the node to start at
    * @return a Map[T,Seq[Seq[T]]] where the value associated with v is the Seq of all paths from start to v
    */
  def pathsInDAG(start: T): Map[T,Seq[Seq[T]]] = {
    // paths(v) holds the set of paths from start to v
    val paths = new mutable.HashMap[T,mutable.Set[Seq[T]]] with mutable.MultiMap[T,Seq[T]]
    val queue = new mutable.Queue[T]
    val visited = new mutable.HashSet[T]
    paths.addBinding(start,Seq(start))
    queue.enqueue(start)
    visited += start
    while (!queue.isEmpty) {
      val current = queue.dequeue
      for (v <- getEdges(current)) {
        if (!visited.contains(v)) {
          queue.enqueue(v)
          visited += v
        }
        for (p <- paths(current)) {
          paths.addBinding(v, p :+ v)
        }
      }
    }
      (paths map { case (k,v) => (k,v.toSeq) }).toMap
  }

  /** Returns a graph with all edges reversed */
  def reverse: DiGraph[T] = {
    val mdg = new MutableDiGraph[T]
    edges foreach { case (u,edges) => edges.foreach({ v => mdg.addEdge(v,u) }) }
    DiGraph(mdg)
  }

  /** Return a graph with only a subset of the nodes
    *
    * Any edge including a deleted node will be deleted
    * 
    * @param vprime the Set[T] of desired vertices
    * @throws IllegalArgumentException if vprime is not a subset of V
    * @return the subgraph
    */
  def subgraph(vprime: Set[T]): DiGraph[T] = {
    require(vprime.subsetOf(edges.keySet))
    val eprime = vprime.map(v => (v,getEdges(v) & vprime)).toMap
    new DiGraph(eprime)
  }

  /** Return a graph with only a subset of the nodes
    *
    * Any path between two non-deleted nodes (u,v) that traverses only
    * deleted nodes will be transformed into an edge (u,v).
    * 
    * @param vprime the Set[T] of desired vertices
    * @throws IllegalArgumentException if vprime is not a subset of V
    * @return the simplified graph
    */
  def simplify(vprime: Set[T]): DiGraph[T] = {
    require(vprime.subsetOf(edges.keySet))
    val eprime = vprime.map( v => (v,reachableFrom(v) & (vprime-v)) ).toMap
    new DiGraph(eprime)
  }

  /** Return a graph with all the nodes of the current graph transformed
    * by a function. Edge connectivity will be the same as the current
    * graph.
    * 
    * @param f A function {(T) => Q} that transforms each node
    * @return a transformed DiGraph[Q]
    */
  def transformNodes[Q](f: (T) => Q): DiGraph[Q] = {
    val eprime = edges.map({ case (k,v) => (f(k),v.map(f(_))) })
    new DiGraph(eprime)
  }

}
