// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.stage.PrettyNoExprInlining
import firrtl.annotations.{NoTargetAnnotation, Target}
import firrtl.annotations.TargetToken.{fromStringToTargetToken, OfModule, Ref}
import firrtl.ir._
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

  override def prerequisites = Seq.empty

  override def optionalPrerequisites = Seq.empty

  override def optionalPrerequisiteOf = Forms.BackendEmitters

  override def invalidates(a: Transform) = false

  type Netlist = mutable.HashMap[WrappedExpression, (Expression, Info)]

  private def isArgN(outerExpr: DoPrim, subExpr: Expression, n: Int): Boolean = {
    outerExpr.args.lift(n) match {
      case Some(arg) => arg eq subExpr
      case _         => false
    }
  }

  private val fileLineRegex = """(.*) ([0-9]+):[0-9]+""".r
  private def getFileAndLineNumbers(info: Info): Set[(String, String)] = {
    info match {
      case FileInfo(fileLineRegex(file, line)) => Set(file -> line)
      case FileInfo(file)                      => Set(file -> "0")
      case MultiInfo(infos)                    => infos.flatMap(getFileAndLineNumbers).toSet
      case NoInfo                              => Set.empty[(String, String)]
    }
  }

  private def sameFileAndLineInfo(info1: Info, info2: Info): Boolean = {
    val set1 = getFileAndLineNumbers(info1)
    val set2 = getFileAndLineNumbers(info2)
    set1.subsetOf(set2)
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
      * @param ref the WRef that references refExpr
      * @param refExpr the expression to check for inlining
      * @param outerExpr the parent expression of refExpr, if any
      */
    def canInline(ref: WRef, refExpr: Expression, outerExpr: Option[Expression]): Boolean = {
      val contextInsensitiveDetOps: Set[PrimOp] = Set(Lt, Leq, Gt, Geq, Eq, Neq, Andr, Orr, Xorr)
      outerExpr match {
        case None => true
        case Some(o) =>
          if ((refExpr.tpe != Utils.BoolType) || refExpr.isInstanceOf[Mux]) {
            false
          } else {
            o match {
              // if outer expression is also boolean context does not affect width
              case o if o.tpe == Utils.BoolType => true

              // mux condition argument is self-determined
              case m: Mux if m.cond eq ref => true

              // dshl/dshr second argument is self-determined
              case DoPrim(Dshl | Dshlw | Dshr, Seq(_, shamt), _, _) if shamt eq ref => true

              case o =>
                refExpr match {
                  case DoPrim(op, _, _, _) => contextInsensitiveDetOps(op)
                  case _                   => false
                }
            }
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
            case Some((refExpr, refInfo)) if sameFileAndLineInfo(refInfo, info) =>
              val inlineNum = inlineCounts.getOrElse(refKey, 1)
              val notTooDeep = !outerExpr.isDefined || ((inlineNum + inlineCount) <= maxInlineCount)
              if (canInline(ref, refExpr, outerExpr) && notTooDeep) {
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
    val run = !state.annotations.contains(PrettyNoExprInlining)

    if (run) {
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
    } else {
      logger.info(s"--${PrettyNoExprInlining.longOption} specified, skipping...")
      state
    }
  }
}
