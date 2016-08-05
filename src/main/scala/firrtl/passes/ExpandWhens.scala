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
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.WrappedExpression._

// Datastructures
import scala.collection.mutable.HashMap
import scala.collection.mutable.LinkedHashMap
import scala.collection.mutable.ArrayBuffer

import annotation.tailrec

/** Expand Whens
*
* @note This pass does three things: remove last connect semantics,
* remove conditional blocks, and eliminate concept of scoping.
* @note Assumes bulk connects and isInvalids have been expanded
* @note Assumes all references are declared
*/
object ExpandWhens extends Pass {
  def name = "Expand Whens"

  // ========== Expand When Utilz ==========
  private def getEntries(
      hash: LinkedHashMap[WrappedExpression, Expression],
      exps: Seq[Expression]): LinkedHashMap[WrappedExpression, Expression] = {
    val hashx = LinkedHashMap[WrappedExpression, Expression]()
    exps foreach (e => if (hash.contains(e)) hashx(e) = hash(e))
    hashx
  }
  private def getFemaleRefs(n: String, t: Type, g: Gender): Seq[Expression] = {
    def getGender(t: Type, i: Int, g: Gender): Gender = times(g, get_flip(t, i, Default))
    val exps = create_exps(WRef(n, t, ExpKind(), g))
    val expsx = ArrayBuffer[Expression]()
    for (j <- 0 until exps.size) {
      getGender(t, j, g) match {
        case (BIGENDER | FEMALE) => expsx += exps(j)
        case _ =>
      }
    }
    expsx
  }
  private def expandNetlist(netlist: LinkedHashMap[WrappedExpression, Expression]) =
    netlist map { case (k, v) =>
      v match {
        case WInvalid() => IsInvalid(NoInfo, k.e1)
        case _ => Connect(NoInfo, k.e1, v)
      }
    }
  // Searches nested scopes of defaults for lvalue
  // defaults uses mutable Map because we are searching LinkedHashMaps and conversion to immutable is VERY slow
  @tailrec
  private def getDefault(
      lvalue: WrappedExpression, 
      defaults: Seq[collection.mutable.Map[WrappedExpression, Expression]]): Option[Expression] = {
    if (defaults.isEmpty) None
    else if (defaults.head.contains(lvalue)) defaults.head.get(lvalue)
    else getDefault(lvalue, defaults.tail)
  }

  // ------------ Pass -------------------
  def run(c: Circuit): Circuit = {
    def expandWhens(m: Module): (LinkedHashMap[WrappedExpression, Expression], ArrayBuffer[Statement], Statement) = {
      val namespace = Namespace(m)
      val simlist = ArrayBuffer[Statement]()

      // defaults ideally would be immutable.Map but conversion from mutable.LinkedHashMap to mutable.Map is VERY slow
      def expandWhens(
          netlist: LinkedHashMap[WrappedExpression, Expression],
          defaults: Seq[collection.mutable.Map[WrappedExpression, Expression]],
          p: Expression)
          (s: Statement): Statement = {
        s match {
          case w: DefWire =>
            getFemaleRefs(w.name, w.tpe, BIGENDER) foreach (ref => netlist(ref) = WVoid())
            w
          case r: DefRegister =>
            getFemaleRefs(r.name, r.tpe, BIGENDER) foreach (ref => netlist(ref) = ref)
            r
          case c: Connect =>
            netlist(c.loc) = c.expr
            EmptyStmt
          case c: IsInvalid =>
            netlist(c.expr) = WInvalid()
            EmptyStmt
          case s: Conditionally =>
            val memos = ArrayBuffer[Statement]()

            val conseqNetlist = LinkedHashMap[WrappedExpression, Expression]()
            val altNetlist = LinkedHashMap[WrappedExpression, Expression]()
            val conseqStmt = expandWhens(conseqNetlist, netlist +: defaults, AND(p, s.pred))(s.conseq)
            val altStmt = expandWhens(altNetlist, netlist +: defaults, AND(p, NOT(s.pred)))(s.alt)

            (conseqNetlist.keySet ++ altNetlist.keySet) foreach { lvalue =>
              // Defaults in netlist get priority over those in defaults
              val default = if (netlist.contains(lvalue)) netlist.get(lvalue) else getDefault(lvalue, defaults)
              val res = default match {
                case Some(defaultValue) =>
                  val trueValue = conseqNetlist.getOrElse(lvalue, defaultValue)
                  val falseValue = altNetlist.getOrElse(lvalue, defaultValue)
                  (trueValue, falseValue) match {
                    case (WInvalid(), WInvalid()) => WInvalid()
                    case (WInvalid(), fv) => ValidIf(NOT(s.pred), fv, tpe(fv))
                    case (tv, WInvalid()) => ValidIf(s.pred, tv, tpe(tv))
                    case (tv, fv) => Mux(s.pred, tv, fv, mux_type_and_widths(tv, fv))
                  }
                case None =>
                  // Since not in netlist, lvalue must be declared in EXACTLY one of conseq or alt
                  conseqNetlist.getOrElse(lvalue, altNetlist(lvalue))
              }

              val memoNode = DefNode(s.info, namespace.newTemp, res)
              val memoExpr = WRef(memoNode.name, res.tpe, NodeKind(), MALE)
              memos += memoNode
              netlist(lvalue) = memoExpr
            }
            Block(Seq(conseqStmt, altStmt) ++ memos)

          case s: Print =>
            if(weq(p, one)) {
              simlist += s
            } else {
              simlist += Print(s.info, s.string, s.args, s.clk, AND(p, s.en))
            }
            EmptyStmt
          case s: Stop =>
            if (weq(p, one)) {
              simlist += s
            } else {
              simlist += Stop(s.info, s.ret, s.clk, AND(p, s.en))
            }
            EmptyStmt
          case s => s map expandWhens(netlist, defaults, p)
        }
      }
      val netlist = LinkedHashMap[WrappedExpression, Expression]()

      // Add ports to netlist
      m.ports foreach { port =>
        getFemaleRefs(port.name, port.tpe, to_gender(port.direction)) foreach (ref => netlist(ref) = WVoid())
      }
      val bodyx = expandWhens(netlist, Seq(netlist), one)(m.body)

      (netlist, simlist, bodyx)
    }
    val modulesx = c.modules map { m =>
      m match {
        case m: ExtModule => m
        case m: Module =>
        val (netlist, simlist, bodyx) = expandWhens(m)
        val newBody = Block(Seq(squashEmpty(bodyx)) ++ expandNetlist(netlist) ++ simlist)
        Module(m.info, m.name, m.ports, newBody)
      }
    }
    Circuit(c.info, modulesx, c.main)
  }
}

