// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._

import annotation.tailrec

/** Reports errors for any references that are not fully initialized
  *
  * @note This pass looks for [[firrtl.WVoid]]s left behind by [[ExpandWhens]]
  * @note Assumes single connection (ie. no last connect semantics)
  */
object CheckInitialization extends Pass {
  private case class VoidExpr(stmt: Statement, voidDeps: Seq[Expression])

  class RefNotInitializedException(info: Info, mname: String, name: String, trace: Seq[Statement]) extends PassException(
      s"$info : [module $mname]  Reference $name is not fully initialized.\n" + 
      trace.map(s => s"  ${get_info(s)} : ${s.serialize}").mkString("\n")
    )

  private def getTrace(expr: WrappedExpression, voidExprs: Map[WrappedExpression, VoidExpr]): Seq[Statement] = {
    @tailrec
    def rec(e: WrappedExpression, map: Map[WrappedExpression, VoidExpr], trace: Seq[Statement]): Seq[Statement] = {
      val voidExpr = map(e) 
      val newTrace = voidExpr.stmt +: trace
      if (voidExpr.voidDeps.nonEmpty) rec(voidExpr.voidDeps.head, map, newTrace) else newTrace
    }
    rec(expr, voidExprs, Seq())
  }

  def run(c: Circuit): Circuit = {
    val errors = new Errors()

    def checkInitM(m: Module): Unit = {
      val voidExprs = collection.mutable.HashMap[WrappedExpression, VoidExpr]()

      def hasVoidExpr(e: Expression): (Boolean, Seq[Expression]) = {
        var void = false
        val voidDeps = collection.mutable.ArrayBuffer[Expression]()
        def hasVoid(e: Expression): Expression = e match {
          case WVoid =>
            void = true
            e
          case (_: WRef | _: WSubField) =>
            if (voidExprs.contains(e)) {
              void = true
              voidDeps += e
            }
            e
          case _ => e map hasVoid
        }
        hasVoid(e)
        (void, voidDeps)
      }
      def checkInitS(s: Statement): Statement = {
        s match {
          case con: Connect =>
            val (hasVoid, voidDeps) = hasVoidExpr(con.expr)
            if (hasVoid) voidExprs(con.loc) = VoidExpr(con, voidDeps)
            con
          case node: DefNode =>
            val (hasVoid, voidDeps) = hasVoidExpr(node.value)
            if (hasVoid) {
              val nodeRef = WRef(node.name, node.value.tpe, NodeKind, MALE)
              voidExprs(nodeRef) = VoidExpr(node, voidDeps)
            }
            node
          case sx => sx map checkInitS
        }
      }
      checkInitS(m.body)

      // Build Up Errors
      for ((expr, _) <- voidExprs) {
        getDeclaration(m, expr.e1) match {
          case node: DefNode => // Ignore nodes
          case decl: IsDeclaration =>
            val trace = getTrace(expr, voidExprs.toMap)
            errors append new RefNotInitializedException(decl.info, m.name, decl.name, trace)
        }
      }
    }

    c.modules foreach {
      case m: Module => checkInitM(m)
      case m => // Do nothing
    }
    errors.trigger()
    c
  }
}
