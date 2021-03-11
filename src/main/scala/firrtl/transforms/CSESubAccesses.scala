// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.ir._
import firrtl.traversals.Foreachers._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.WrappedExpression._
import firrtl.options.Dependency
import firrtl.passes._
import firrtl.Utils.{distinctBy, flow, getAllRefs, get_info, niceName}

import scala.collection.mutable

object CSESubAccesses {

  // Get all SubAccesses used on the right-hand side along with the info from the outer Statement
  private def collectRValueSubAccesses(mod: Module): Seq[(SubAccess, Info)] = {
    val acc = new mutable.ListBuffer[(SubAccess, Info)]
    def onExpr(outer: Statement)(expr: Expression): Unit = {
      // Need postorder because we want to visit inner SubAccesses first
      // Stop recursing on any non-Source because flips can make the SubAccess a Source despite the
      //   overall Expression being a Sink
      if (flow(expr) == SourceFlow) expr.foreach(onExpr(outer))
      expr match {
        case e: SubAccess if flow(e) == SourceFlow => acc += e -> get_info(outer)
        case _ => // Do nothing
      }
    }
    def onStmt(stmt: Statement): Unit = {
      stmt.foreach(onStmt)
      stmt match {
        // Don't record SubAccesses that are already assigned to a Node, but *do* record any nested
        // inside of the SubAccess. This makes the transform idempotent and avoids unnecessary work.
        case DefNode(_, _, acc: SubAccess) => acc.foreach(onExpr(stmt))
        case other => other.foreach(onExpr(stmt))
      }
    }
    onStmt(mod.body)
    distinctBy(acc.toList)(_._1)
  }

  // Replaces all right-hand side SubAccesses with References
  private def replaceOnSourceExpr(replace: SubAccess => Reference)(expr: Expression): Expression = expr match {
    // Stop is we ever see a non-SourceFlow
    case e if flow(e) != SourceFlow => e
    // Don't traverse children of SubAccess, just replace it
    // Nested SubAccesses are handled during creation of the nodes that the references refer to
    case acc: SubAccess if flow(acc) == SourceFlow => replace(acc)
    case other => other.map(replaceOnSourceExpr(replace))
  }

  private def hoistSubAccesses(
    hoist:   String => List[DefNode],
    replace: SubAccess => Reference
  )(stmt:    Statement
  ): Statement = {
    val onExpr = replaceOnSourceExpr(replace) _
    def onStmt(s: Statement): Statement = s.map(onExpr).map(onStmt) match {
      case decl: IsDeclaration =>
        val nodes = hoist(decl.name)
        if (nodes.isEmpty) decl else Block(decl :: nodes)
      case other => other
    }
    onStmt(stmt)
  }

  // Given some nodes, determine after which String declaration each node should be inserted
  // This function is *mutable*, it keeps track of which declarations each node is sensitive to and
  // returns nodes in groups once the last declaration they depend on is seen
  private def getSensitivityLookup(nodes: Iterable[DefNode]): String => List[DefNode] = {
    case class ReferenceCount(var n: Int, node: DefNode)
    // Gather names of declarations each node depends on
    val nodeDeps = nodes.map(node => getAllRefs(node.value).view.map(_.name).toSet -> node)
    // Map from declaration names to the indices of nodeDeps that depend on it
    val lookup = new mutable.HashMap[String, mutable.ArrayBuffer[Int]]
    for (((decls, _), idx) <- nodeDeps.zipWithIndex) {
      for (d <- decls) {
        val indices = lookup.getOrElseUpdate(d, new mutable.ArrayBuffer[Int])
        indices += idx
      }
    }
    // Now we can just associate each List of nodes with how many declarations they need to see
    // We use an Array because we're mutating anyway and might as well be quick about it
    val nodeLists: Array[ReferenceCount] =
      nodeDeps.view.map { case (deps, node) => ReferenceCount(deps.size, node) }.toArray

    // Must be a def because it's recursive
    def func(decl: String): List[DefNode] = {
      if (lookup.contains(decl)) {
        val indices = lookup(decl)
        val result = new mutable.ListBuffer[DefNode]
        lookup -= decl
        for (i <- indices) {
          val refCount = nodeLists(i)
          refCount.n -= 1
          assert(refCount.n >= 0, "Internal Error!")
          if (refCount.n == 0) result += refCount.node
        }
        // DefNodes can depend on each other, recurse
        result.toList.flatMap { node => node :: func(node.name) }
      } else {
        Nil
      }
    }
    func _
  }

  /** Performs [[CSESubAccesses]] on a single [[ir.Module Module]] */
  def onMod(mod: Module): Module = {
    // ***** Pre-Analyze (do we even need to do anything) *****
    val accesses = collectRValueSubAccesses(mod)
    if (accesses.isEmpty) mod
    else {
      // ***** Analyze *****
      val namespace = Namespace(mod)
      val replace = new mutable.HashMap[SubAccess, Reference]
      val nodes = new mutable.ArrayBuffer[DefNode]
      for ((acc, info) <- accesses) {
        val name = namespace.newName(niceName(acc))
        // SubAccesses can be nested, so replace any nested ones with prior references
        // This is why post-order traversal in collectRValueSubAccesses is important
        val accx = acc.map(replaceOnSourceExpr(replace))
        val node = DefNode(info, name, accx)
        val ref = Reference(node)
        // Record in replace
        replace(acc) = ref
        // Record node
        nodes += node
      }
      val hoist = getSensitivityLookup(nodes)

      // ***** Transform *****
      val portStmts = mod.ports.flatMap(x => hoist(x.name))
      val bodyx = hoistSubAccesses(hoist, replace)(mod.body)
      mod.copy(body = if (portStmts.isEmpty) bodyx else Block(Block(portStmts), bodyx))
    }
  }
}

/** Performs Common Subexpression Elimination (CSE) on right-hand side [[ir.SubAccess SubAccess]]es
  *
  * This avoids quadratic node creation behavior in [[passes.RemoveAccesses RemoveAccesses]]. For
  * simplicity of implementation, all SubAccesses on the right-hand side are also split into
  * individual nodes.
  */
class CSESubAccesses extends Transform with DependencyAPIMigration {

  override def prerequisites = Dependency(ResolveFlows) :: Dependency(CheckHighForm) :: Nil

  // Faster to run after these
  override def optionalPrerequisites = Dependency(ReplaceAccesses) :: Dependency[DedupModules] :: Nil

  // Running before ExpandConnects is an optimization
  override def optionalPrerequisiteOf = Dependency(ExpandConnects) :: Nil

  override def invalidates(a: Transform) = false

  def execute(state: CircuitState): CircuitState = {
    val modulesx = state.circuit.modules.map {
      case ext: ExtModule => ext
      case mod: Module    => CSESubAccesses.onMod(mod)
    }
    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
