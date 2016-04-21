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
import scala.collection.mutable.LinkedHashMap
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
// ; ========== Expand When Utilz ==========
   def add (hash:LinkedHashMap[WrappedExpression,Expression],key:WrappedExpression,value:Expression) = {
      hash += (key -> value)
   }

   def get_entries (hash:LinkedHashMap[WrappedExpression,Expression],exps:Seq[Expression]) : LinkedHashMap[WrappedExpression,Expression] = {
      val hashx = LinkedHashMap[WrappedExpression,Expression]()
      exps.foreach { e => {
         val value = hash.get(e)
         value match {
            case (value:Some[Expression]) => add(hashx,e,value.get)
            case (None) => {}
         }
      }}
      hashx
   }
   def get_female_refs (n:String,t:Type,g:Gender) : Seq[Expression] = {
      val exps = create_exps(WRef(n,t,ExpKind(),g))
      val expsx = ArrayBuffer[Expression]()
      def get_gender (t:Type, i:Int, g:Gender) : Gender = {
         val f = get_flip(t,i,DEFAULT)
         times(g, f)
      }
      for (i <- 0 until exps.size) {
         get_gender(t,i,g) match {
            case BIGENDER => expsx += exps(i)
            case FEMALE => expsx += exps(i)
            case _ => false
         }
      }
      expsx
   }

   // ------------ Pass -------------------
   def run (c:Circuit): Circuit = {
      def void_all (m:InModule) : InModule = {
         mname = m.name
         def void_all_s (s:Stmt) : Stmt = {
            (s) match {
               case (_:DefWire|_:DefRegister|_:WDefInstance|_:DefMemory) => {
                  val voids = ArrayBuffer[Stmt]()
                  for (e <- get_female_refs(get_name(s),get_type(s),get_gender(s))) {
                     voids += Connect(get_info(s),e,WVoid())
                  }
                  Begin(Seq(s,Begin(voids)))
               }
               case (s) => s map (void_all_s)
            }
         }
         val voids = ArrayBuffer[Stmt]()
         for (p <- m.ports) {
            for (e <- get_female_refs(p.name,p.tpe,get_gender(p))) {
               voids += Connect(p.info,e,WVoid())
            }
         }
         val bodyx = void_all_s(m.body)
         InModule(m.info,m.name,m.ports,Begin(Seq(Begin(voids),bodyx)))
      }
      def expand_whens (m:InModule) : Tuple2[LinkedHashMap[WrappedExpression,Expression],ArrayBuffer[Stmt]] = {
         val simlist = ArrayBuffer[Stmt]()
         mname = m.name
         def expand_whens (netlist:LinkedHashMap[WrappedExpression,Expression],p:Expression)(s:Stmt) : Stmt = {
            (s) match {
               case (s:Connect) => netlist(s.loc) = s.exp
               case (s:IsInvalid) => netlist(s.exp) = WInvalid()
               case (s:Conditionally) => {
                  val exps = ArrayBuffer[Expression]()
                  def prefetch (s:Stmt) : Stmt = {
                     (s) match {
                        case (s:Connect) => exps += s.loc; s
                        case (s) => s map(prefetch)
                     }
                  }
                  prefetch(s.conseq)
                  val c_netlist = get_entries(netlist,exps)
                  expand_whens(c_netlist,AND(p,s.pred))(s.conseq)
                  expand_whens(netlist,AND(p,NOT(s.pred)))(s.alt)
                  for (lvalue <- c_netlist.keys) {
                     val value = netlist.get(lvalue)
                     (value) match {
                        case (value:Some[Expression]) => {
                           val tv = c_netlist(lvalue)
                           val fv = value.get
                           val res = (tv,fv) match {
                              case (tv:WInvalid,fv:WInvalid) => WInvalid()
                              case (tv:WInvalid,fv) => ValidIf(NOT(s.pred),fv,tpe(fv))
                              case (tv,fv:WInvalid) => ValidIf(s.pred,tv,tpe(tv))
                              case (tv,fv) => Mux(s.pred,tv,fv,mux_type_and_widths(tv,fv))
                           }
                           netlist(lvalue) = res
                        }
                        case (None) => add(netlist,lvalue,c_netlist(lvalue))
                     }
                  }
               }
               case (s:Print) => {
                  if (weq(p,one)) {
                     simlist += s
                  } else {
                     simlist += Print(s.info,s.string,s.args,s.clk,AND(p,s.en))
                  }
               }
               case (s:Stop) => {
                  if (weq(p,one)) {
                     simlist += s
                  } else {
                     simlist += Stop(s.info,s.ret,s.clk,AND(p,s.en))
                  }
               }
               case (s) => s map(expand_whens(netlist,p))
            }
            s
         }
         val netlist = LinkedHashMap[WrappedExpression,Expression]()
         expand_whens(netlist,one)(m.body)

         //println("Netlist:")
         //println(netlist)
         //println("Simlist:")
         //println(simlist)
         ( netlist, simlist )
      }

      def create_module (netlist:LinkedHashMap[WrappedExpression,Expression],simlist:ArrayBuffer[Stmt],m:InModule) : InModule = {
         mname = m.name
         val stmts = ArrayBuffer[Stmt]()
         val connections = ArrayBuffer[Stmt]()
         def replace_void (e:Expression)(rvalue:Expression) : Expression = {
            (rvalue) match {
               case (rv:WVoid) => e
               case (rv) => rv map (replace_void(e))
            }
         }
         def create (s:Stmt) : Stmt = {
            (s) match {
               case (_:DefWire|_:WDefInstance|_:DefMemory) => {
                  stmts += s
                  for (e <- get_female_refs(get_name(s),get_type(s),get_gender(s))) {
                     val rvalue = netlist(e)
                     val con = (rvalue) match {
                        case (rvalue:WInvalid) => IsInvalid(get_info(s),e)
                        case (rvalue) => Connect(get_info(s),e,rvalue)
                     }
                     connections += con
                  }
               }
               case (s:DefRegister) => {
                  stmts += s
                  for (e <- get_female_refs(get_name(s),get_type(s),get_gender(s))) {
                     val rvalue = replace_void(e)(netlist(e))
                     val con = (rvalue) match {
                        case (rvalue:WInvalid) => IsInvalid(get_info(s),e)
                        case (rvalue) => Connect(get_info(s),e,rvalue)
                     }
                     connections += con
                  }
               }
               case (_:DefPoison|_:DefNode) => stmts += s
               case (s) => s map(create)
            }
            s
         }
         create(m.body)
         for (p <- m.ports) {
            for (e <- get_female_refs(p.name,p.tpe,get_gender(p))) {
               val rvalue = netlist(e)
               val con = (rvalue) match {
                  case (rvalue:WInvalid) => IsInvalid(p.info,e)
                  case (rvalue) => Connect(p.info,e,rvalue)
               }
               connections += con
            }
         }
         for (x <- simlist) { stmts += x }
         InModule(m.info,m.name,m.ports,Begin(Seq(Begin(stmts),Begin(connections))))
      }

      val voided_modules = c.modules.map{ m => {
            (m) match {
               case (m:ExModule) => m
               case (m:InModule) => void_all(m)
            } } }
      val modulesx = voided_modules.map{ m => {
            (m) match {
               case (m:ExModule) => m
               case (m:InModule) => {
                  val (netlist, simlist) = expand_whens(m)
                  create_module(netlist,simlist,m)
               }
            }}}
      Circuit(c.info,modulesx,c.main)
   }
}

