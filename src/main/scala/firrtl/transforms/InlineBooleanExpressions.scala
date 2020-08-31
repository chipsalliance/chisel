// See LICENSE for license details.

package firrtl
package transforms

import firrtl.annotations.{NoTargetAnnotation, Target}
import firrtl.annotations.TargetToken.{fromStringToTargetToken, OfModule, Ref}
import firrtl.ir._
import firrtl.passes.{InferTypes, LowerTypes, SplitExpressions}
import firrtl.options.Dependency
import firrtl.stage.Forms
import firrtl.PrimOps._
import firrtl.WrappedExpression._

import scala.collection.mutable

case class InlineBooleanExpressionsMax(max: Int) extends NoTargetAnnotation

object InlineBooleanExpressions {
  val defaultMax = 30
}

/** Inline Bool expressions
  *
  * The following conditions must be satisfied to inline
  * 1. has type [[Utils.BoolType]]
  * 2. is bound to a [[firrtl.ir.DefNode DefNode]] with name starting with '_'
  * 3. is bound to a [[firrtl.ir.DefNode DefNode]] with a source locator that
  *    points at the same file and line number. If it is a MultiInfo source
  *    locator, the set of file and line number pairs must be the same. Source
  *    locators may point to different column numbers.
  * 4. [[InlineBooleanExpressionsMax]] has not been exceeded
  * 5. is not a [[firrtl.ir.Mux Mux]]
  */
class InlineBooleanExpressions extends Transform with DependencyAPIMigration {

  override def prerequisites = Seq(
    Dependency(InferTypes),
    Dependency(LowerTypes)
  )

  override def optionalPrerequisites = Seq(
    Dependency(SplitExpressions)
  )

  override def optionalPrerequisiteOf = Forms.BackendEmitters

  override def invalidates(a: Transform) = a match {
    case _: DeadCodeElimination => true // this transform does not remove nodes that are unused after inlining
    case _ => false
  }

  type Netlist = mutable.HashMap[WrappedExpression, (Expression, Info)]

  private def isArgN(outerExpr: DoPrim, subExpr: Expression, n: Int): Boolean = {
    outerExpr.args.lift(n) match {
      case Some(arg) => arg eq subExpr
      case _         => false
    }
  }

  private val fileLineRegex = """(.*) ([0-9]+):[0-9]+""".r
  private def sameFileAndLineInfo(info1: Info, info2: Info): Boolean = {
    (info1, info2) match {
      case (FileInfo(fileLineRegex(file1, line1)), FileInfo(fileLineRegex(file2, line2))) =>
        (file1 == file2) && (line1 == line2)
      case (MultiInfo(infos1), MultiInfo(infos2)) if infos1.size == infos2.size =>
        infos1.zip(infos2).forall {
          case (i1, i2) =>
            sameFileAndLineInfo(i1, i2)
        }
      case (NoInfo, NoInfo) => true
      case _                => false
    }
  }

  /** A helper class to initialize and store mutable state that the expression
    * and statement map functions need access to. This makes it easier to pass
    * information around without having to plump arguments through the onExpr
    * and onStmt methods.
    */
  private class MapMethods(maxInlineCount: Int, dontTouches: Set[Ref]) {
    val netlist: Netlist = new Netlist
    val inlineCounts = mutable.Map.empty[Ref, Int]
    var inlineCount: Int = 1

    /** Whether or not an can be inlined
      * @param refExpr the expression to check for inlining
      * @param outerExpr the parent expression of refExpr, if any
      */
    def canInline(refExpr: Expression, outerExpr: Option[Expression]): Boolean = {
      val contextInsensitiveDetOps: Set[PrimOp] = Set(Lt, Leq, Gt, Geq, Eq, Neq, Andr, Orr, Xorr)
      outerExpr match {
        case None => true
        case Some(o) if (o.tpe == Utils.BoolType) =>
          refExpr match {
            case _: Mux => false
            case e => e.tpe == Utils.BoolType
          }
        case Some(o) =>
          refExpr match {
            case DoPrim(op, _, _, Utils.BoolType) => contextInsensitiveDetOps(op)
            case _                                => false
          }
      }
    }

    /** Inlines [[Wref]]s if they are Boolean, have matching file line numbers,
      * and would not raise inlineCounts past the maximum.
      *
      * @param info the [[Info]] of the enclosing [[Statement]]
      * @param outerExpr the direct parent [[Expression]] of the current [[Expression]]
      * @param expr the [[Expression]] to apply inlining to
      */
    def onExpr(info: Info, outerExpr: Option[Expression])(expr: Expression): Expression = {
      expr match {
        case ref: WRef if !dontTouches.contains(ref.name.Ref) && ref.name.head == '_' =>
          val refKey = ref.name.Ref
          netlist.get(we(ref)) match {
            case Some((refExpr, refInfo)) if sameFileAndLineInfo(info, refInfo) =>
              val inlineNum = inlineCounts.getOrElse(refKey, 1)
              val notTooDeep = !outerExpr.isDefined || ((inlineNum + inlineCount) <= maxInlineCount)
              if (canInline(refExpr, outerExpr) && notTooDeep) {
                inlineCount += inlineNum
                refExpr
              } else {
                ref
              }
            case other => ref
          }
        case other => other.mapExpr(onExpr(info, Some(other)))
      }
    }

    /** Applies onExpr and records metadata for every [[HasInfo]] in a [[Statement]]
      *
      * This resets inlineCount before inlining and records the resulting
      * inline counts and inlined values in the inlineCounts and netlist maps
      * after inlining.
      */
    def onStmt(stmt: Statement): Statement = {
      stmt.mapStmt(onStmt) match {
        case hasInfo: HasInfo =>
          inlineCount = 1
          val stmtx = hasInfo.mapExpr(onExpr(hasInfo.info, None))
          stmtx match {
            case node: DefNode => inlineCounts(node.name.Ref) = inlineCount
            case _ =>
          }
          stmtx match {
            case node @ DefNode(info, name, value) =>
              netlist(we(WRef(name))) = (value, info)
            case _ =>
          }
          stmtx
        case other => other
      }
    }
  }

  def execute(state: CircuitState): CircuitState = {
    val dontTouchMap: Map[OfModule, Set[Ref]] = {
      val refTargets = state.annotations.flatMap {
        case anno: HasDontTouches => anno.dontTouches
        case o => Nil
      }
      val dontTouches: Seq[(OfModule, Ref)] = refTargets.map {
        case r => Target.referringModule(r).module.OfModule -> r.ref.Ref
      }
      dontTouches.groupBy(_._1).mapValues(_.map(_._2).toSet).toMap
    }

    val maxInlineCount = state.annotations.collectFirst {
      case InlineBooleanExpressionsMax(max) => max
    }.getOrElse(InlineBooleanExpressions.defaultMax)

    val modulesx = state.circuit.modules.map { m =>
      val mapMethods = new MapMethods(maxInlineCount, dontTouchMap.getOrElse(m.name.OfModule, Set.empty[Ref]))
      m.mapStmt(mapMethods.onStmt(_))
    }

    state.copy(circuit = state.circuit.copy(modules = modulesx))
  }
}
