package firrtl.altIR
import firrtl.ir._
import scala.collection.mutable

/**
 * Base class for the graph nodes in the graph representation of firrtl AST tree.
 */
abstract class FirrtlGraphNode {
  def neighbors: Seq[FirrtlGraphNode]
}

/**
 * Graph node equivalent of Statement AST tree nodes in IR.scala.
 */
abstract class StatementGraphNode extends FirrtlGraphNode

/**
 * Graph node representation of nodes that can be refered to by name in the tree form of the IR
 */
abstract class NamedGraphNode extends StatementGraphNode {
  val name: String
  val references = mutable.ArrayBuffer.empty[ReferenceGraphNode]
  def addReference(ref: ReferenceGraphNode): Unit
}

/**
 * Graph node representation of nodes that can be assigned to by connect statements in the tree form
 * of firrtl.
 */
abstract class AssignableGraphNode extends NamedGraphNode {
  def neighbors = references
  def addReference(ref: ReferenceGraphNode): Unit = {
    references += ref
    ref.namedNode = Some(this)
  }
}

/**
 * Graph node equivalent to Expression nodes in the tree form of the IR.
 */
abstract class ExpressionGraphNode extends FirrtlGraphNode {
  val tpe: Type
  var parent: Option[FirrtlGraphNode]
  def addParent(node: ExpressionGraphNode): Unit = {
    parent = Some(node)
    node match {
      case subField: SubFieldGraphNode =>
        subField.expr = Some(this)
      case subIndex: SubIndexGraphNode =>
        subIndex.expr = Some(this)
    }
  }
}

/**
 * Graph node equivalent to Reference nodes in the tree form of the IR
 */
class ReferenceGraphNode(
  val name: String, val tpe: Type
) extends ExpressionGraphNode {
  var namedNode: Option[NamedGraphNode] = None
  var parent: Option[FirrtlGraphNode] = None
  def neighbors = namedNode.toList ++ parent.toList
}

/**
 * Graph node equivalent to SubField nodes in the tree form of the IR
 */
class SubFieldGraphNode(
  val name: String, val tpe: Type
) extends ExpressionGraphNode {
  var expr: Option[ExpressionGraphNode] = None
  var parent: Option[FirrtlGraphNode] = None
  def neighbors = expr.toList ++ parent.toList
}

/**
 * Graph node equivalent to SubIndex nodes in the tree form of the IR
 */
class SubIndexGraphNode(
  val value: Int, val tpe: Type
) extends ExpressionGraphNode {
  var expr: Option[ExpressionGraphNode] = None
  var parent: Option[FirrtlGraphNode] = None
  def neighbors = expr.toList ++ parent.toList
}

/**
 * Graph node equivalent to DefWire nodes in the tree form of the IR
 */
class DefWireGraphNode(
  val info: Info,
  val name: String,
  val tpe: Type
) extends AssignableGraphNode

/**
 * Graph node equivalent to DefInstance nodes in the tree form of the IR
 */
class DefInstanceGraphNode(
  val info: Info,
  val name: String,
  val module: String
) extends AssignableGraphNode

/**
 * Graph nodes that represent the IO ports of a module not present in the tree form of the IR.
 */
class PortGraphNode(
  val info: Info,
  val name: String,
  val direction: Direction,
  val tpe: Type
) extends AssignableGraphNode

/**
 * Graph node equivalent to Connect nodes in the tree form of the IR
 */
class ConnectGraphNode(val info: Info) extends StatementGraphNode {
  var loc: Option[ExpressionGraphNode] = None
  var expr: Option[ExpressionGraphNode] = None
  def neighbors = loc.toList ++ expr.toList
  def addLoc(node: ExpressionGraphNode): Unit = {
    loc = Some(node)
    node.parent = Some(this)
  }
  def addExpr(node: ExpressionGraphNode): Unit = {
    expr = Some(node)
    node.parent = Some(this)
  }
}

/**
 * Graph node equivalent to IsInvalid nodes in the tree form of the IR
 */
class IsInvalidGraphNode(val info: Info) extends StatementGraphNode {
  var expr: Option[ExpressionGraphNode] = None
  def neighbors = expr.toList
}

object getGraphNode {
  def apply[T](pointer: Option[T]): T = {
    pointer.getOrElse(
      throw new Exception(
        s"InsertWrapperModules pass encountered unexpectedly unconnected "
        + s"graph node pointer"
      )
    )
  }
}
