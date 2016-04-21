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
import firrtl.PrimOps._
import firrtl.WrappedExpression._

// Datastructures
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

/** Expand Whens
*
* @note This pass does three things: remove last connect semantics,
* remove conditional blocks, and eliminate concept of scoping.
*/
object ExpandWhens extends Pass {
  def name = "Expand Whens"
  var mname = ""
  // ========== Expand When Utilz ==========
  def getEntries(
      hash: HashMap[WrappedExpression, Expression],
      exps: Seq[Expression]): HashMap[WrappedExpression, Expression] = {
    val hashx = HashMap[WrappedExpression, Expression]()
    exps foreach (e => if (hash.contains(e)) hashx(e) = hash(e))
    hashx
  }
  def getFemaleRefs(n: String, t: Type, g: Gender): Seq[Expression] = {
    def getGender(t: Type, i: Int, g: Gender): Gender = times(g, get_flip(t, i, DEFAULT))
    val exps = create_exps(WRef(n, t, ExpKind(), g))
    val expsx = ArrayBuffer[Expression]()
    for (i <- 0 until exps.size) {
      getGender(t, i, g) match {
        case (BIGENDER | FEMALE) => expsx += exps(i)
        case _ =>
      }
    }
    expsx
  }

  // ------------ Pass -------------------
  def run(c: Circuit): Circuit = {
    def voidAll(m: InModule): InModule = {
      mname = m.name
      def voidAllStmt(s: Stmt): Stmt = s match {
        case (_: DefWire | _: DefRegister | _: WDefInstance |_: DefMemory) =>
          val voids = ArrayBuffer[Stmt]()
          for (e <- getFemaleRefs(get_name(s),get_type(s),get_gender(s))) {
            voids += Connect(get_info(s),e,WVoid())
          }
          Begin(Seq(s,Begin(voids)))
        case s => s map voidAllStmt
      }
      val voids = ArrayBuffer[Stmt]()
      for (p <- m.ports) {
        for (e <- getFemaleRefs(p.name,p.tpe,get_gender(p))) {
          voids += Connect(p.info,e,WVoid())
        }
      }
      val bodyx = voidAllStmt(m.body)
      InModule(m.info, m.name, m.ports, Begin(Seq(Begin(voids),bodyx)))
    }
    def expandWhens(m: InModule): (HashMap[WrappedExpression, Expression], ArrayBuffer[Stmt]) = {
      val simlist = ArrayBuffer[Stmt]()
      mname = m.name
      def expandWhens(netlist: HashMap[WrappedExpression, Expression], p: Expression)(s: Stmt): Stmt = {
        s match {
          case s: Connect => netlist(s.loc) = s.exp
          case s: IsInvalid => netlist(s.exp) = WInvalid()
          case s: Conditionally =>
            val exps = ArrayBuffer[Expression]()
            def prefetch(s: Stmt): Stmt = s match {
              case s: Connect => exps += s.loc; s
              case s => s map prefetch
            }
            prefetch(s.conseq)
            val c_netlist = getEntries(netlist,exps)
            expandWhens(c_netlist, AND(p, s.pred))(s.conseq)
            expandWhens(netlist, AND(p, NOT(s.pred)))(s.alt)
            for (lvalue <- c_netlist.keys) {
              val value = netlist.get(lvalue)
              value match {
                case value: Some[Expression] =>
                  val tv = c_netlist(lvalue)
                  val fv = value.get
                  val res = (tv, fv) match {
                    case (tv:WInvalid, fv:WInvalid) => WInvalid()
                    case (tv:WInvalid, fv) => ValidIf(NOT(s.pred), fv,tpe(fv))
                    case (tv, fv:WInvalid) => ValidIf(s.pred, tv, tpe(tv))
                    case (tv, fv) => Mux(s.pred, tv, fv, mux_type_and_widths(tv, fv))
                  }
                  netlist(lvalue) = res
                case None => netlist(lvalue) = c_netlist(lvalue)
              }
            }
          case s: Print =>
            if(weq(p, one)) {
              simlist += s
            } else {
              simlist += Print(s.info, s.string, s.args, s.clk, AND(p, s.en))
            }
          case s: Stop =>
            if (weq(p, one)) {
              simlist += s
            } else {
              simlist += Stop(s.info, s.ret, s.clk, AND(p, s.en))
            }
          case s => s map expandWhens(netlist, p)
        }
        s
      }
      val netlist = HashMap[WrappedExpression, Expression]()
      expandWhens(netlist, one)(m.body)

      (netlist, simlist)
    }

    def createModule(netlist: HashMap[WrappedExpression,Expression], simlist: ArrayBuffer[Stmt], m: InModule): InModule = {
      mname = m.name
      val stmts = ArrayBuffer[Stmt]()
      val connections = ArrayBuffer[Stmt]()
      def replace_void(e: Expression)(rvalue: Expression): Expression = rvalue match {
        case rv: WVoid => e
        case rv => rv map replace_void(e)
      }
      def create(s: Stmt): Stmt = {
        s match {
          case (_: DefWire | _: WDefInstance | _: DefMemory) =>
            stmts += s
            for (e <- getFemaleRefs(get_name(s), get_type(s), get_gender(s))) {
              val rvalue = netlist(e)
              val con = rvalue match {
                case rvalue: WInvalid => IsInvalid(get_info(s), e)
                case rvalue => Connect(get_info(s), e, rvalue)
              }
              connections += con
            }
          case s: DefRegister =>
            stmts += s
            for (e <- getFemaleRefs(get_name(s), get_type(s), get_gender(s))) {
              val rvalue = replace_void(e)(netlist(e))
              val con = rvalue match {
                case rvalue: WInvalid => IsInvalid(get_info(s), e)
                case rvalue => Connect(get_info(s), e, rvalue)
              }
              connections += con
            }
          case (_: DefPoison | _: DefNode) => stmts += s
          case s => s map create
        }
        s
      }
      create(m.body)
      for (p <- m.ports) {
        for (e <- getFemaleRefs(p.name, p.tpe, get_gender(p))) {
          val rvalue = netlist(e)
          val con = rvalue match {
            case rvalue: WInvalid => IsInvalid(p.info, e)
            case rvalue => Connect(p.info, e, rvalue)
          }
          connections += con
        }
      }
      for (x <- simlist) { stmts += x }
      InModule(m.info, m.name, m.ports, Begin(Seq(Begin(stmts), Begin(connections))))
    }

    val voided_modules = c.modules map { m =>
      m match {
        case m: ExModule => m
        case m: InModule => voidAll(m)
      }
    }

    val modulesx = voided_modules map { m =>
      m match {
        case m: ExModule => m
        case m: InModule =>
        val (netlist, simlist) = expandWhens(m)
        createModule(netlist, simlist, m)

      }
    }
    Circuit(c.info, modulesx, c.main)
  }
}

