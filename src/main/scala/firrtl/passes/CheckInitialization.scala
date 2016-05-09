/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

package firrtl.passes

import firrtl._
import firrtl.Utils._
import firrtl.Mappers._

import annotation.tailrec

/** Reports errors for any references that are not fully initialized
  *
  * @note This pass looks for [[firrtl.WVoid]]s left behind by [[ExpandWhens]]
  * @note Assumes single connection (ie. no last connect semantics)
  */
object CheckInitialization extends Pass {
  def name = "Check Initialization"

  private case class VoidExpr(stmt: Stmt, voidDeps: Seq[Expression])

  class RefNotInitializedException(info: Info, mname: String, name: String, trace: Seq[Stmt]) extends PassException(
      s"$info : [module $mname]  Reference $name is not fully initialized.\n" + 
      trace.map(s => s"  ${get_info(s)} : ${s.serialize}").mkString("\n")
    )

  private def getTrace(expr: WrappedExpression, voidExprs: Map[WrappedExpression, VoidExpr]): Seq[Stmt] = {
    @tailrec
    def rec(e: WrappedExpression, map: Map[WrappedExpression, VoidExpr], trace: Seq[Stmt]): Seq[Stmt] = {
      val voidExpr = map(e) 
      val newTrace = voidExpr.stmt +: trace
      if (voidExpr.voidDeps.nonEmpty) rec(voidExpr.voidDeps.head, map, newTrace) else newTrace
    }
    rec(expr, voidExprs, Seq())
  }

  def run(c: Circuit): Circuit = {
    val errors = collection.mutable.ArrayBuffer[PassException]()


    def checkInitM(m: Module): Unit = {
      val voidExprs = collection.mutable.HashMap[WrappedExpression, VoidExpr]()

      def hasVoidExpr(e: Expression): (Boolean, Seq[Expression]) = {
        var void = false
        val voidDeps = collection.mutable.ArrayBuffer[Expression]()
        def hasVoid(e: Expression): Expression = {
          e match {
            case e: WVoid =>
              void = true
              e
            case (_: WRef | _: WSubField) =>
              if (voidExprs.contains(e)) {
                void = true
                voidDeps += e
              }
              e
            case e => e map hasVoid
          }
        }
        hasVoid(e)
        (void, voidDeps)
      }
      def checkInitS(s: Stmt): Stmt = {
        s match {
          case con: Connect =>
            val (hasVoid, voidDeps) = hasVoidExpr(con.exp)
            if (hasVoid) voidExprs(con.loc) = VoidExpr(con, voidDeps)
            con
          case node: DefNode =>
            val (hasVoid, voidDeps) = hasVoidExpr(node.value)
            if (hasVoid) {
              val nodeRef = WRef(node.name, node.value.tpe, NodeKind(), MALE)
              voidExprs(nodeRef) = VoidExpr(node, voidDeps)
            }
            node
          case s => s map checkInitS
        }
      }
      checkInitS(m.body)

      // Build Up Errors
      for ((expr, _) <- voidExprs) {
        getDeclaration(m, expr.e1) match {
          case node: DefNode => // Ignore nodes
          case decl: IsDeclaration =>
            val trace = getTrace(expr, voidExprs.toMap)
            errors += new RefNotInitializedException(decl.info, m.name, decl.name, trace)
        }
      }
    }

    c.modules foreach { m =>
      m match {
        case m: Module => checkInitM(m)
        case m => // Do nothing
      }
    }

    if (errors.nonEmpty) throw new PassExceptions(errors)
    c
  }
}
