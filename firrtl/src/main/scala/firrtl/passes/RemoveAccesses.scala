// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.{Namespace, Transform, WRef, WSubAccess, WSubField, WSubIndex}
import firrtl.PrimOps.{And, Eq}
import firrtl.ir._
import firrtl.Mappers._
import firrtl.Utils._
import firrtl.WrappedExpression._
import firrtl.options.Dependency

import scala.collection.mutable

/** Removes all [[firrtl.WSubAccess]] from circuit
  */
object RemoveAccesses extends Pass {

  override def prerequisites =
    Seq(
      Dependency(PullMuxes),
      Dependency(ZeroLengthVecs),
      Dependency(ReplaceAccesses),
      Dependency(ExpandConnects)
    ) ++ firrtl.stage.Forms.Deduped

  override def invalidates(a: Transform): Boolean = a match {
    case ResolveFlows => true
    case _            => false
  }

  private def AND(e1: Expression, e2: Expression) =
    if (e1 == one) e2
    else if (e2 == one) e1
    else DoPrim(And, Seq(e1, e2), Nil, BoolType)

  private def EQV(e1: Expression, e2: Expression): Expression =
    DoPrim(Eq, Seq(e1, e2), Nil, BoolType)

  /** Container for a base expression and its corresponding guard
    */
  private case class Location(base: Expression, guard: Expression)

  /** Walks a referencing expression and returns a list of valid references
    * (base) and the corresponding guard which, if true, returns that base.
    * E.g. if called on a[i] where a: UInt[2], we would return:
    *   Seq(Location(a[0], UIntLiteral(0)), Location(a[1], UIntLiteral(1)))
    */
  private def getLocations(e: Expression): Seq[Location] = e match {
    case e: SubIndex =>
      val ls = getLocations(e.expr)
      val start = get_point(e)
      val end = start + get_size(e.tpe)
      val stride = get_size(e.expr.tpe)
      for (
        (l, i) <- ls.zipWithIndex
        if ((i % stride) >= start) & ((i % stride) < end)
      ) yield l
    case e: SubField =>
      val ls = getLocations(e.expr)
      val start = get_point(e)
      val end = start + get_size(e.tpe)
      val stride = get_size(e.expr.tpe)
      for (
        (l, i) <- ls.zipWithIndex
        if ((i % stride) >= start) & ((i % stride) < end)
      ) yield l
    case SubAccess(expr, index, tpe, _) =>
      getLocations(expr).zipWithIndex.flatMap {
        case (Location(exprBase, exprGuard), exprIndex) =>
          getLocations(index).map {
            case Location(indexBase, indexGuard) =>
              Location(
                exprBase,
                AND(
                  AND(
                    indexGuard,
                    exprGuard
                  ),
                  EQV(
                    UIntLiteral((exprIndex / get_size(tpe)) % expr.tpe.asInstanceOf[VectorType].size),
                    indexBase
                  )
                )
              )
          }
      }
    case e => create_exps(e).map(Location(_, one))
  }

  /** Returns true if e contains a [[firrtl.WSubAccess]]
    */
  private def hasAccess(e: Expression): Boolean = {
    var ret: Boolean = false
    def rec_has_access(e: Expression): Expression = {
      e match {
        case _: WSubAccess => ret = true
        case _ =>
      }
      e.map(rec_has_access)
    }
    rec_has_access(e)
    ret
  }

  // This improves the performance of this pass
  private val createExpsCache = mutable.HashMap[Expression, Seq[Expression]]()
  private def create_exps(e: Expression) =
    createExpsCache.getOrElseUpdate(e, firrtl.Utils.create_exps(e))

  def run(c: Circuit): Circuit = {
    def remove_m(m: Module): Module = {
      val namespace = Namespace(m)
      def onStmt(s: Statement): Statement = {
        def create_temp(e: Expression): (Statement, Expression) = {
          val n = namespace.newName(niceName(e))
          (DefWire(get_info(s), n, e.tpe), WRef(n, e.tpe, kind(e), flow(e)))
        }

        /** Replaces a subaccess in a given source expression
          */
        val stmts = mutable.ArrayBuffer[Statement]()
        // Only called on RefLikes that definitely have a SubAccess
        // Must accept Expression because that's the output type of fixIndices
        def removeSource(e: Expression): Expression = {
          val rs = getLocations(e)
          rs.find(x => x.guard != one) match {
            case None => throwInternalError(s"removeSource: shouldn't be here - $e")
            case Some(_) =>
              val (wire, temp) = create_temp(e)
              val temps = create_exps(temp)
              def getTemp(i: Int) = temps(i % temps.size)
              stmts += wire
              rs.zipWithIndex.foreach {
                case (x, i) if i < temps.size =>
                  stmts += IsInvalid(get_info(s), getTemp(i))
                  stmts += Conditionally(get_info(s), x.guard, Connect(get_info(s), getTemp(i), x.base), EmptyStmt)
                case (x, i) =>
                  stmts += Conditionally(get_info(s), x.guard, Connect(get_info(s), getTemp(i), x.base), EmptyStmt)
              }
              temp
          }
        }

        /** Replaces a subaccess in a given sink expression
          */
        def removeSink(info: Info, loc: Expression): Expression = loc match {
          case (_: WSubAccess | _: WSubField | _: WSubIndex | _: WRef) if hasAccess(loc) =>
            val ls = getLocations(loc)
            if (ls.size == 1 & weq(ls.head.guard, one)) loc
            else {
              val (wire, temp) = create_temp(loc)
              stmts += wire
              ls.foreach(x =>
                stmts +=
                  Conditionally(info, x.guard, Connect(info, x.base, temp), EmptyStmt)
              )
              temp
            }
          case _ => loc
        }

        /** Recurse until find SubAccess and call fixSource on its index
          * @note this only accepts [[RefLikeExpression]]s but we can't enforce it because map
          *       requires Expression => Expression
          */
        def fixIndices(e: Expression): Expression = e match {
          case e: SubAccess => e.copy(index = fixSource(e.index))
          case other => other.map(fixIndices)
        }

        /** Recursively walks a source expression and fixes all subaccesses
          *
          * If we see a RefLikeExpression that contains a SubAccess, we recursively remove
          * subaccesses from the indices of any SubAccesses, then process modified RefLikeExpression
          */
        def fixSource(e: Expression): Expression = e match {
          case ref: RefLikeExpression =>
            if (hasAccess(ref)) removeSource(fixIndices(ref)) else ref
          case x => x.map(fixSource)
        }

        /** Recursively walks a sink expression and fixes all subaccesses
          * If we see a sub-access, its index is a source expression, and we must replace it.
          * Otherwise, map to children.
          */
        def fixSink(e: Expression): Expression = e match {
          case w: WSubAccess => WSubAccess(fixSink(w.expr), fixSource(w.index), w.tpe, w.flow)
          case x => x.map(fixSink)
        }

        val sx = s match {
          case Connect(info, loc, exp) =>
            Connect(info, removeSink(info, fixSink(loc)), fixSource(exp))
          case sxx => sxx.map(fixSource).map(onStmt)
        }
        stmts += sx
        if (stmts.size != 1) Block(stmts.toSeq) else stmts(0)
      }
      Module(m.info, m.name, m.ports, squashEmpty(onStmt(m.body)))
    }

    c.copy(modules = c.modules.map {
      case m: ExtModule => m
      case m: Module    => remove_m(m)
    })
  }
}
