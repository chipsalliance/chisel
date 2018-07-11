// See LICENSE for license details.

package firrtl.graph

import scala.collection.{Set, Map}
import scala.collection.mutable
import scala.collection.mutable.{LinkedHashSet, LinkedHashMap}

/** An exception that is raised when an assumed DAG has a cycle */
class CyclicException(val node: Any) extends Exception(s"No valid linearization for cyclic graph, found at $node")

/** An exception that is raised when attempting to find an unreachable node */
class PathNotFoundException extends Exception("Unreachable node")

/** A companion to create DiGraphs from mutable data */
object DiGraph {
  /** Create a DiGraph from a MutableDigraph, representing the same graph */
  def apply[T](mdg: MutableDiGraph[T]): DiGraph[T] = mdg

  /** Create a DiGraph from a Map[T,Set[T]] of edge data */
  def apply[T](edgeData: Map[T,Set[T]]): DiGraph[T] = {
    val edgeDataCopy = new LinkedHashMap[T, LinkedHashSet[T]]
    for ((k, v) <- edgeData) {
      edgeDataCopy(k) = new LinkedHashSet[T]
    }
    for ((k, v) <- edgeData) {
      for (n <- v) {
        require(edgeDataCopy.contains(n), s"Does not contain $n")
        edgeDataCopy(k) += n
      }
    }
    new DiGraph(edgeDataCopy)
  }
}

/** Represents common behavior of all directed graphs */
class DiGraph[T] private[graph] (private[graph] val edges: LinkedHashMap[T, LinkedHashSet[T]]) {
  /** Check whether the graph contains vertex v */
  def contains(v: T): Boolean = edges.contains(v)

  /** Get all vertices in the graph
    * @return a Set[T] of all vertices in the graph
    */
  // The pattern of mapping map pairs to keys maintains LinkedHashMap ordering
  def getVertices: Set[T] = new LinkedHashSet ++ edges.map({ case (k, _) => k })

  /** Get all edges of a node
    * @param v the specified node
    * @return a Set[T] of all vertices that v has edges to
    */
  def getEdges(v: T): Set[T] = edges.getOrElse(v, Set.empty)

  def getEdgeMap: Map[T, Set[T]] = edges

  /** Find all sources in the graph
    *
    * @return a Set[T] of source nodes
    */
  def findSources: Set[T] = getVertices -- edges.values.flatten.toSet

  /** Find all sinks in the graph
    *
    * @return a Set[T] of sink nodes
    */
  def findSinks: Set[T] = reverse.findSources

  /** Linearizes (topologically sorts) a DAG
    *
    * @throws CyclicException if the graph is cyclic
    * @return a Map[T,T] from each visited node to its predecessor in the
    * traversal
    */
  def linearize: Seq[T] = {
    // permanently marked nodes are implicitly held in order
    val order = new mutable.ArrayBuffer[T]
    // invariant: no intersection between unmarked and tempMarked
    val unmarked = new mutable.LinkedHashSet[T]
    val tempMarked = new mutable.LinkedHashSet[T]

    case class LinearizeFrame[T](v: T, expanded: Boolean)
    val callStack = mutable.Stack[LinearizeFrame[T]]()

    unmarked ++= getVertices
    while (unmarked.nonEmpty) {
      callStack.push(LinearizeFrame(unmarked.head, false))
      while (callStack.nonEmpty) {
        val LinearizeFrame(n, expanded) = callStack.pop()
        if (!expanded) {
          if (tempMarked.contains(n)) {
            throw new CyclicException(n)
          }
          if (unmarked.contains(n)) {
            tempMarked += n
            unmarked -= n
            callStack.push(LinearizeFrame(n, true))
            // We want to visit the first edge first (so push it last)
            for (m <- edges.getOrElse(n, Set.empty).toSeq.reverse) {
              callStack.push(LinearizeFrame(m, false))
            }
          }
        } else {
          tempMarked -= n
          order.append(n)
        }
      }
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
  def BFS(root: T): Map[T,T] = BFS(root, Set.empty[T])

  /** Performs breadth-first search on the directed graph, with a blacklist of nodes
    *
    * @param root the start node
    * @param blacklist list of nodes to stop searching, if encountered
    * @return a Map[T,T] from each visited node to its predecessor in the
    * traversal
    */
  def BFS(root: T, blacklist: Set[T]): Map[T,T] = {
    val prev = new mutable.LinkedHashMap[T,T]
    val queue = new mutable.Queue[T]
    queue.enqueue(root)
    while (queue.nonEmpty) {
      val u = queue.dequeue
      for (v <- getEdges(u)) {
        if (!prev.contains(v) && !blacklist.contains(v)) {
          prev(v) = u
          queue.enqueue(v)
        }
      }
    }
    prev
  }

  /** Finds the set of nodes reachable from a particular node
    *
    * @param root the start node
    * @return a Set[T] of nodes reachable from the root
    */
  def reachableFrom(root: T): LinkedHashSet[T] = reachableFrom(root, Set.empty[T])

  /** Finds the set of nodes reachable from a particular node, with a blacklist
    *
    * @param root the start node
    * @param blacklist list of nodes to stop searching, if encountered
    * @return a Set[T] of nodes reachable from the root
    */
  def reachableFrom(root: T, blacklist: Set[T]): LinkedHashSet[T] = new LinkedHashSet[T] ++ BFS(root, blacklist).map({ case (k, v) => k })

  /** Finds a path (if one exists) from one node to another
    *
    * @param start the start node
    * @param end the destination node
    * @throws PathNotFoundException
    * @return a Seq[T] of nodes defining an arbitrary valid path
    */
  def path(start: T, end: T): Seq[T] = path(start, end, Set.empty[T])
  
  /** Finds a path (if one exists) from one node to another, with a blacklist
    *
    * @param start the start node
    * @param end the destination node
    * @param blacklist list of nodes which break path, if encountered
    * @throws PathNotFoundException
    * @return a Seq[T] of nodes defining an arbitrary valid path
    */
  def path(start: T, end: T, blacklist: Set[T]): Seq[T] = {
    val nodePath = new mutable.ArrayBuffer[T]
    val prev = BFS(start, blacklist)
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
    val onstack = new LinkedHashSet[T]
    val indices = new LinkedHashMap[T, BigInt]
    val lowlinks = new LinkedHashMap[T, BigInt]
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
  def pathsInDAG(start: T): LinkedHashMap[T,Seq[Seq[T]]] = {
    // paths(v) holds the set of paths from start to v
    val paths = new LinkedHashMap[T, mutable.Set[Seq[T]]]
    val queue = new mutable.Queue[T]
    val reachable = reachableFrom(start)
    def addBinding(n: T, p: Seq[T]): Unit = {
      paths.getOrElseUpdate(n, new LinkedHashSet[Seq[T]]) += p
    }
    addBinding(start,Seq(start))
    queue += start
    queue ++= linearize.filter(reachable.contains(_))
    while (!queue.isEmpty) {
      val current = queue.dequeue
      for (v <- getEdges(current)) {
        for (p <- paths(current)) {
          addBinding(v, p :+ v)
        }
      }
    }
    paths.map({ case (k,v) => (k,v.toSeq) })
  }

  /** Returns a graph with all edges reversed */
  def reverse: DiGraph[T] = {
    val mdg = new MutableDiGraph[T]
    edges.foreach({ case (u, edges) => mdg.addVertex(u) })
    edges.foreach({ case (u, edges) =>
      edges.foreach(v => mdg.addEdge(v,u))
    })
    DiGraph(mdg)
  }

  private def filterEdges(vprime: Set[T]): LinkedHashMap[T, LinkedHashSet[T]] = {
    def filterNodeSet(s: LinkedHashSet[T]): LinkedHashSet[T] = s.filter({ case (k) => vprime.contains(k) })
    def filterAdjacencyLists(m: LinkedHashMap[T, LinkedHashSet[T]]): LinkedHashMap[T, LinkedHashSet[T]] = m.map({ case (k, v) => (k, filterNodeSet(v)) })
    var eprime: LinkedHashMap[T, LinkedHashSet[T]] = edges.filter({ case (k, v) => vprime.contains(k) })
    filterAdjacencyLists(eprime)
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
    new DiGraph(filterEdges(vprime))
  }

  /** Return a simplified connectivity graph with only a subset of the nodes
    *
    * Any path between two non-deleted nodes (u,v) in the original graph will be
    * transformed into an edge (u,v).
    *
    * @param vprime the Set[T] of desired vertices
    * @throws IllegalArgumentException if vprime is not a subset of V
    * @return the simplified graph
    */
  def simplify(vprime: Set[T]): DiGraph[T] = {
    require(vprime.subsetOf(edges.keySet))
    val pathEdges = vprime.map( v => (v, reachableFrom(v) & (vprime-v)) )
    new DiGraph(new LinkedHashMap[T, LinkedHashSet[T]] ++ pathEdges)
  }

  /** Return a graph with all the nodes of the current graph transformed
    * by a function. Edge connectivity will be the same as the current
    * graph.
    *
    * @param f A function {(T) => Q} that transforms each node
    * @return a transformed DiGraph[Q]
    */
  def transformNodes[Q](f: (T) => Q): DiGraph[Q] = {
    val eprime = edges.map({ case (k, _) => (f(k), new LinkedHashSet[Q]) })
    edges.foreach({ case (k, v) => eprime(f(k)) ++= v.map(f(_)) })
    new DiGraph(eprime)
  }

  /** Graph sum of `this` and `that`
    *
    * @param that a second DiGraph[T]
    * @return a DiGraph[T] containing all vertices and edges of each graph
    */
  def +(that: DiGraph[T]): DiGraph[T] = {
    val eprime = edges.clone
    that.edges.map({ case (k, v) => eprime.getOrElseUpdate(k, new LinkedHashSet[T]) ++= v })
    new DiGraph(eprime)
  }
}

class MutableDiGraph[T] extends DiGraph[T](new LinkedHashMap[T, LinkedHashSet[T]]) {
  /** Add vertex v to the graph
    * @return v, the added vertex
    */
  def addVertex(v: T): T = {
    edges.getOrElseUpdate(v, new LinkedHashSet[T])
    v
  }

  /** Add edge (u,v) to the graph.
    * @throws IllegalArgumentException if u and/or v is not in the graph
    */
  def addEdge(u: T, v: T): Unit = {
    require(contains(u))
    require(contains(v))
    edges(u) += v
  }

  /** Add edge (u,v) to the graph, adding u and/or v if they are not
    * already in the graph.
    */
  def addPairWithEdge(u: T, v: T): Unit = {
    edges.getOrElseUpdate(v, new LinkedHashSet[T])
    edges.getOrElseUpdate(u, new LinkedHashSet[T]) += v
  }

  /** Add edge (u,v) to the graph if and only if both u and v are in
    * the graph prior to calling addEdgeIfValid.
    */
  def addEdgeIfValid(u: T, v: T): Boolean = {
    val valid = contains(u) && contains(v)
    if (contains(u) && contains(v)) {
      edges(u) += v
    }
    valid
  }
}
