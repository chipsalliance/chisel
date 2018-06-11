// See LICENSE for license details.

package firrtl.passes

import firrtl.{WRef, WSubAccess, WSubIndex, WSubField, Namespace}
import firrtl.PrimOps.{And, Eq}
import firrtl.ir._
import firrtl.Mappers._
import firrtl.Utils._
import firrtl.WrappedExpression._
import scala.collection.mutable


/** Removes all [[firrtl.WSubAccess]] from circuit
  */
object RemoveAccesses extends Pass {
  private def AND(e1: Expression, e2: Expression) =
    if(e1 == one) e2
    else if(e2 == one) e1
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
    case e: WRef => create_exps(e).map(Location(_,one))
    case e: WSubIndex =>
      val ls = getLocations(e.expr)
      val start = get_point(e)
      val end = start + get_size(e.tpe)
      val stride = get_size(e.expr.tpe)
      for ((l, i) <- ls.zipWithIndex
        if ((i % stride) >= start) & ((i % stride) < end)) yield l
    case e: WSubField =>
      val ls = getLocations(e.expr)
      val start = get_point(e)
      val end = start + get_size(e.tpe)
      val stride = get_size(e.expr.tpe)
      for ((l, i) <- ls.zipWithIndex
        if ((i % stride) >= start) & ((i % stride) < end)) yield l
    case e: WSubAccess =>
      val ls = getLocations(e.expr)
      val stride = get_size(e.tpe)
      val wrap = e.expr.tpe.asInstanceOf[VectorType].size
      ls.zipWithIndex map {case (l, i) =>
        val c = (i / stride) % wrap
        val basex = l.base
        val guardx = AND(l.guard,EQV(UIntLiteral(c),e.index))
        Location(basex,guardx)
      }
  }

  /** Returns true if e contains a [[firrtl.WSubAccess]]
    */
  private def hasAccess(e: Expression): Boolean = {
    var ret: Boolean = false
    def rec_has_access(e: Expression): Expression = {
      e match {
        case _ : WSubAccess => ret = true
        case _ =>
      }
      e map rec_has_access
    }
    rec_has_access(e)
    ret
  }

  // This improves the performance of this pass
  private val createExpsCache = mutable.HashMap[Expression, Seq[Expression]]()
  private def create_exps(e: Expression) =
    createExpsCache getOrElseUpdate (e, firrtl.Utils.create_exps(e))

  def run(c: Circuit): Circuit = {
    def remove_m(m: Module): Module = {
      val namespace = Namespace(m)
      def onStmt(s: Statement): Statement = {
        def create_temp(e: Expression): (Statement, Expression) = {
          val n = namespace.newName(niceName(e))
          (DefWire(get_info(s), n, e.tpe), WRef(n, e.tpe, kind(e), gender(e)))
        }

        /** Replaces a subaccess in a given male expression
          */
        val stmts = mutable.ArrayBuffer[Statement]()
        def removeMale(e: Expression): Expression = e match {
          case (_:WSubAccess| _: WSubField| _: WSubIndex| _: WRef) if hasAccess(e) =>
            val rs = getLocations(e)
            rs find (x => x.guard != one) match {
              case None => throwInternalError(s"removeMale: shouldn't be here - $e")
              case Some(_) =>
                val (wire, temp) = create_temp(e)
                val temps = create_exps(temp)
                def getTemp(i: Int) = temps(i % temps.size)
                stmts += wire
                rs.zipWithIndex foreach {
                  case (x, i) if i < temps.size =>
                    stmts += IsInvalid(get_info(s),getTemp(i))
                    stmts += Conditionally(get_info(s),x.guard,Connect(get_info(s),getTemp(i),x.base),EmptyStmt)
                  case (x, i) =>
                    stmts += Conditionally(get_info(s),x.guard,Connect(get_info(s),getTemp(i),x.base),EmptyStmt)
                }
                temp
            }
          case _ => e
        }

        /** Replaces a subaccess in a given female expression
          */
        def removeFemale(info: Info, loc: Expression): Expression = loc match {
          case (_: WSubAccess| _: WSubField| _: WSubIndex| _: WRef) if hasAccess(loc) =>
            val ls = getLocations(loc)
            if (ls.size == 1 & weq(ls.head.guard,one)) loc
            else {
              val (wire, temp) = create_temp(loc)
              stmts += wire
              ls foreach (x => stmts +=
                Conditionally(info,x.guard,Connect(info,x.base,temp),EmptyStmt))
              temp
            }
          case _ => loc
        }

        /** Recursively walks a male expression and fixes all subaccesses
          * If we see a sub-access, replace it.
          * Otherwise, map to children.
          */
        def fixMale(e: Expression): Expression = e match {
          case w: WSubAccess => removeMale(WSubAccess(w.expr, fixMale(w.index), w.tpe, w.gender))
          //case w: WSubIndex => removeMale(w)
          //case w: WSubField => removeMale(w)
          case x => x map fixMale
        }

        /** Recursively walks a female expression and fixes all subaccesses
          * If we see a sub-access, its index is a male expression, and we must replace it.
          * Otherwise, map to children.
          */
        def fixFemale(e: Expression): Expression = e match {
          case w: WSubAccess => WSubAccess(fixFemale(w.expr), fixMale(w.index), w.tpe, w.gender)
          case x => x map fixFemale
        }

        val sx = s match {
          case Connect(info, loc, exp) =>
            Connect(info, removeFemale(info, fixFemale(loc)), fixMale(exp))
          case sxx => sxx map fixMale map onStmt
        }
        stmts += sx
        if (stmts.size != 1) Block(stmts) else stmts(0)
      }
      Module(m.info, m.name, m.ports, squashEmpty(onStmt(m.body)))
    }
  
    c copy (modules = c.modules map {
      case m: ExtModule => m
      case m: Module => remove_m(m)
    })
  }
}
