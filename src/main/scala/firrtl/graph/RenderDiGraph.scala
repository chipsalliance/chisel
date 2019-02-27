// See LICENSE for license details.

package firrtl.graph

import scala.collection.mutable

/**
  * Implement a really simple graphviz dot renderer for a digraph
  * There are three main renderers currently
  * -
  *
  * @param diGraph   The DiGraph to be rendered
  * @param graphName Name of the graph, when shown in viewer
  * @param rankDir   Graph orientation, default is LR (left to right), most common alternative is TB (top to bottom)
  * @tparam T        The type of the Node.
  */
class RenderDiGraph[T <: Any](diGraph: DiGraph[T], graphName: String = "", rankDir: String = "LR") {


  /**
    * override this to change the default way a node is displayed. Default is toString surrounded by double quotes
    * {{{
    * val rend = new RenderDiGraph(graph, "alice") {
    *   override def renderNode(node: Symbol): String = s"\"${symbol.name}\""
    *   }
    * }}}
    */
  def renderNode(node: T): String = {
    s""""${node.toString}""""
  }

  /**
    * This finds a loop in a DiGraph if one exists and returns nodes
    * @note there is no way to currently to specify a particular loop
    * @return
    */
  def findOneLoop: Set[T] = {
    var path = Seq.empty[T]

    try {
      diGraph.linearize
    }
    catch {
      case cyclicException: CyclicException =>
        val node = cyclicException.node.asInstanceOf[T]
        path = diGraph.findLoopAtNode(node)

      case t: Throwable =>
        throw t

    }
    path.toSet
  }

  /**
    * Searches a DiGraph for a cycle. The first one found will be rendered as a graph that contains
    * only the nodes in the cycle plus the neighbors of those nodes.
    * @return a string that can be used as input to the dot command, string is empty if no loop
    */
  def showOnlyTheLoopAsDot: String = {
    // finds the loop

    val loop = findOneLoop

    if(loop.nonEmpty) {

      // Find all the children of the nodes in the loop
      val childrenFound = diGraph.getEdgeMap.flatMap {
        case (node, children) if loop.contains(node) => children
        case _ => Seq.empty
      }.toSet

      // Create a new DiGraph containing only loop and direct children or parents
      val edgeData = diGraph.getEdgeMap
      val newEdgeData = edgeData.flatMap { case (node, children) =>
        if(loop.contains(node)) {
          Some(node -> children)
        }
        else if(childrenFound.contains(node)) {
          Some(node -> children.intersect(loop))
        }
        else {
          val newChildren = children.intersect(loop)
          if(newChildren.nonEmpty) {
            Some(node -> newChildren)
          }
          else {
            None
          }
          }
      }

      val justLoop = DiGraph(newEdgeData)
      val newRenderer = new RenderDiGraph(justLoop, graphName, rankDir) {
        override def renderNode(node: T): String = {
          super.renderNode(node)
        }
      }
      newRenderer.toDotWithLoops(loop, getRankedNodes)
    }
    else {
      ""
    }
  }

  /**
    * Convert this graph into input for the graphviz dot program
    * @return A string representation of the digraph in dot notation
    */
  def toDot: String = {
    val s = new mutable.StringBuilder()

    s.append(s"digraph $graphName {\n")
    s.append(s""" rankdir="$rankDir";""" + "\n")

    val edges = diGraph.getEdgeMap

    edges.foreach { case (parent, children) =>
      children.foreach { child =>
        s.append(s"""  ${renderNode(parent)} -> ${renderNode(child)};""" + "\n")
      }
    }
    s.append("}\n")
    s.toString
  }

  /**
    * Convert this graph into input for the graphviz dot program, but with  a
    * loop,if present, highlighted in red.
    * @return string that is a graphviz digraph, but with loops highlighted
    */
  def toDotWithLoops(loopedNodes: Set[T], rankedNodes: mutable.ArrayBuffer[Seq[T]]): String = {
    val s = new mutable.StringBuilder()
    val allNodes = new mutable.HashSet[T]

    s.append(s"digraph $graphName {\n")
    s.append(s""" rankdir="$rankDir";""" + "\n")

    val edges = diGraph.getEdgeMap

    edges.foreach { case (parent, children) =>
      allNodes += parent
      allNodes ++= children

      children.foreach { child =>
        val highlight = if(loopedNodes.contains(parent) && loopedNodes.contains(child)) {
          "[color=red,penwidth=3.0]"
        }
        else {
          ""
        }
        s.append(s"""  ${renderNode(parent)} -> ${renderNode(child)}$highlight;""" + "\n")
      }
    }

    val paredRankedNodes = rankedNodes.flatMap { nodes =>
      val newNodes = nodes.filter(allNodes.contains)
      if(newNodes.nonEmpty) { Some(newNodes) } else { None }
    }

    paredRankedNodes.foreach { nodesAtRank =>
      s.append(s"""  { rank=same; ${nodesAtRank.map(renderNode).mkString(" ")} };""" + "\n")
    }

    s.append("}\n")
    s.toString
  }

  /**
    * Creates a series of Seq of nodes for each minimum depth that those
    * are from the sources of this graph.
    * @return
    */
  private def getRankedNodes: mutable.ArrayBuffer[Seq[T]] = {
    val alreadyVisited = new mutable.HashSet[T]()
    val rankNodes = new mutable.ArrayBuffer[Seq[T]]()

    def walkByRank(nodes: Seq[T], rankNumber: Int = 0): Unit = {
      rankNodes.append(nodes)

      alreadyVisited ++= nodes

      val nextNodes = nodes.flatMap { node =>
        diGraph.getEdges(node)
      }.filterNot(alreadyVisited.contains).distinct

      if(nextNodes.nonEmpty) {
        walkByRank(nextNodes, rankNumber + 1)
      }
    }

    walkByRank(diGraph.findSources.toSeq)
    rankNodes
  }
  /**
    * Convert this graph into input for the graphviz dot program.
    * It tries to align nodes in columns based
    * on their minimum distance to a source.
    * Can also be faster and better behaved on large graphs
    * @return string that is a graphviz digraph
    */
  def toDotRanked: String = {
    val s = new mutable.StringBuilder()
    val alreadyVisited = new mutable.HashSet[T]()
    val rankNodes = new mutable.ArrayBuffer[Seq[T]]()

    def walkByRank(nodes: Seq[T], rankNumber: Int = 0): Unit = {
      rankNodes.append(nodes)

      alreadyVisited ++= nodes

      val nextNodes = nodes.flatMap { node =>
        val children = diGraph.getEdges(node)

        s.append(s"""  ${renderNode(node)} -> { ${children.map(renderNode).mkString(" ")} };""" + "\n")

        children
      }.filterNot(alreadyVisited.contains).distinct

      if(nextNodes.nonEmpty) {
        walkByRank(nextNodes, rankNumber + 1)
      }
    }

    s.append(s"digraph {\n")
    s.append(s""" rankdir="$rankDir";""" + "\n")

    walkByRank(diGraph.findSources.toSeq)
    rankNodes.foreach { nodesAtRank =>
      s.append(s"""  { rank=same; ${nodesAtRank.map(renderNode).mkString(" ")} };""" + "\n")
    }
    s.append("}\n")
    s.toString
  }
}
