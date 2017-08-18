// See LICENSE for license details.

package firrtl
package analyses

import ir._
import scala.annotation.tailrec
import scala.collection.JavaConverters._

/** This is not intended to be used as a metric for the size of an actual
  *   circuit, but rather to debug the compiler itself
  * Works because all FirrtlNodes implement Product (case classes do)
  */
class NodeCount private (node: FirrtlNode) {

  // Counts the number of unique objects in a Firrtl graph
  // There is no IdentityHashSet
  private val identityMap = new java.util.IdentityHashMap[Any, Boolean]()
  // This is a strict subset of the keys of identityMap
  private val regularSet = new collection.mutable.HashSet[Any]

  @tailrec
  private final def rec(xs: List[Any]): Unit =
    if (xs.isEmpty) { }
    else {
      val node = xs.head
      require(node.isInstanceOf[Product] || !node.isInstanceOf[FirrtlNode],
        "Unexpected FirrtlNode that does not implement Product!")
      val moreToVisit =
        if (identityMap.containsKey(node)) List.empty
        else { // Haven't seen yet
          identityMap.put(node, true)
          regularSet += node
          node match { // FirrtlNodes are Products
            case p: Product => p.productIterator
            case i: Iterable[Any] => i
            case _ => List.empty
          }
        }
      rec(moreToVisit ++: xs.tail)
    }
  rec(List(node))

  /** Number of nodes that are referentially unique
    *
    * !(a eq b)
    */
  def unique: Long = identityMap.size

  /** Number of nodes that are different
    *
    * !(a == b)
    */
  def nonequivalent: Long = regularSet.size

  /** Number of nodes in this NodeCount that are NOT present in that NodeCount */
  def uniqueFrom(that: NodeCount): Long =
    this.identityMap.keySet.asScala.count(!that.identityMap.containsKey(_))
}

object NodeCount {
  def apply(node: FirrtlNode) = new NodeCount(node)
}
