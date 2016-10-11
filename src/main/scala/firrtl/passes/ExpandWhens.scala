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
  type NodeMap = collection.mutable.HashMap[MemoizedHash[Expression], String]
  type Netlist = collection.mutable.LinkedHashMap[WrappedExpression, Expression]
  type Simlist = collection.mutable.ArrayBuffer[Statement]
  type Attachlist = collection.mutable.ArrayBuffer[Statement]
  type Defaults = Seq[collection.mutable.Map[WrappedExpression, Expression]]

  // ========== Expand When Utilz ==========
  private def getFemaleRefs(n: String, t: Type, g: Gender): Seq[Expression] = {
    def getGender(t: Type, i: Int, g: Gender): Gender = times(g, get_flip(t, i, Default))
    val exps = create_exps(WRef(n, t, ExpKind, g))
    (exps.zipWithIndex foldLeft Seq[Expression]()){
      case (expsx, (exp, j)) => exp.tpe match {
        case AnalogType(w) => expsx
        case _ => getGender(t, j, g) match {
          case (BIGENDER | FEMALE) => expsx :+ exp
          case _ => expsx
        }
      }
    }
  }
  private def expandNetlist(netlist: Netlist) =
    netlist map {
      case (k, WInvalid) => IsInvalid(NoInfo, k.e1)
      case (k, v) => Connect(NoInfo, k.e1, v)
    }
  // Searches nested scopes of defaults for lvalue
  // defaults uses mutable Map because we are searching LinkedHashMaps and conversion to immutable is VERY slow
  @tailrec
  private def getDefault(lvalue: WrappedExpression, defaults: Defaults): Option[Expression] = {
    defaults match {
      case Nil => None
      case head :: tail => head get lvalue match {
        case Some(p) => Some(p)
        case None => getDefault(lvalue, tail)
      }
    }
  }

  private def AND(e1: Expression, e2: Expression) =
    DoPrim(And, Seq(e1, e2), Nil, BoolType)
  private def NOT(e: Expression) =
    DoPrim(Eq, Seq(e, zero), Nil, BoolType)

  // ------------ Pass -------------------
  def run(c: Circuit): Circuit = {
    def expandWhens(m: Module): (Netlist, Simlist, Statement) = {
      val namespace = Namespace(m)
      val simlist = new Simlist
      val nodes = new NodeMap

      // defaults ideally would be immutable.Map but conversion from mutable.LinkedHashMap to mutable.Map is VERY slow
      def expandWhens(netlist: Netlist,
                      defaults: Defaults,
                      p: Expression)
                      (s: Statement): Statement = s match {
        case w: DefWire =>
          netlist ++= (getFemaleRefs(w.name, w.tpe, BIGENDER) map (ref => we(ref) -> WVoid))
          w
        case r: DefRegister =>
          netlist ++= (getFemaleRefs(r.name, r.tpe, BIGENDER) map (ref => we(ref) -> ref))
          r
        case c: Connect =>
          netlist(c.loc) = c.expr
          EmptyStmt
        case c: IsInvalid =>
          netlist(c.expr) = WInvalid
          EmptyStmt
        case c: Attach => c
        case sx: Conditionally =>
          val conseqNetlist = new Netlist
          val altNetlist = new Netlist
          val conseqStmt = expandWhens(conseqNetlist, netlist +: defaults, AND(p, sx.pred))(sx.conseq)
          val altStmt = expandWhens(altNetlist, netlist +: defaults, AND(p, NOT(sx.pred)))(sx.alt)

          // Process combined maps because we only want to create 1 mux for each node
          //   present in the conseq and/or alt
          val memos = (conseqNetlist ++ altNetlist) map { case (lvalue, _) =>
            // Defaults in netlist get priority over those in defaults
            val default = netlist get lvalue match {
              case Some(v) => Some(v)
              case None => getDefault(lvalue, defaults)
            }
            val res = default match {
              case Some(defaultValue) =>
                val trueValue = conseqNetlist getOrElse (lvalue, defaultValue)
                val falseValue = altNetlist getOrElse (lvalue, defaultValue)
                (trueValue, falseValue) match {
                  case (WInvalid, WInvalid) => WInvalid
                  case (WInvalid, fv) => ValidIf(NOT(sx.pred), fv, fv.tpe)
                  case (tv, WInvalid) => ValidIf(sx.pred, tv, tv.tpe)
                  case (tv, fv) => Mux(sx.pred, tv, fv, mux_type_and_widths(tv, fv))
                }
              case None =>
                // Since not in netlist, lvalue must be declared in EXACTLY one of conseq or alt
                conseqNetlist getOrElse (lvalue, altNetlist(lvalue))
            }

            res match {
              case _: ValidIf | _: Mux | _: DoPrim => nodes get res match {
                case Some(name) =>
                  netlist(lvalue) = WRef(name, res.tpe, NodeKind, MALE)
                  EmptyStmt
                case None =>
                  val name = namespace.newTemp
                  nodes(res) = name
                  netlist(lvalue) = WRef(name, res.tpe, NodeKind, MALE)
                  DefNode(sx.info, name, res)
              }
              case _ =>
                netlist(lvalue) = res
                EmptyStmt
            }
          }
          Block(Seq(conseqStmt, altStmt) ++ memos)
        case sx: Print =>
          simlist += (if (weq(p, one)) sx else Print(sx.info, sx.string, sx.args, sx.clk, AND(p, sx.en)))
          EmptyStmt
        case sx: Stop =>
          simlist += (if (weq(p, one)) sx else Stop(sx.info, sx.ret, sx.clk, AND(p, sx.en)))
          EmptyStmt
        case sx => sx map expandWhens(netlist, defaults, p)
      }
      val netlist = new Netlist
      // Add ports to netlist
      netlist ++= (m.ports flatMap { case Port(_, name, dir, tpe) =>
        getFemaleRefs(name, tpe, to_gender(dir)) map (ref => we(ref) -> WVoid)
      })
      (netlist, simlist, expandWhens(netlist, Seq(netlist), one)(m.body))
    }
    val modulesx = c.modules map {
      case m: ExtModule => m
      case m: Module =>
      val (netlist, simlist, bodyx) = expandWhens(m)
      val newBody = Block(Seq(squashEmpty(bodyx)) ++ expandNetlist(netlist) ++ simlist)
      Module(m.info, m.name, m.ports, newBody)
    }
    Circuit(c.info, modulesx, c.main)
  }
}

