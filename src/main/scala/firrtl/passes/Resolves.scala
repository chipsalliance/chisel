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

import com.typesafe.scalalogging.LazyLogging
import java.nio.file.{Paths, Files}

// Datastructures
import scala.collection.mutable.LinkedHashMap
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.WrappedExpression._

object ResolveKinds extends Pass {
  private var mname = ""
  def name = "Resolve Kinds"
  def run (c:Circuit): Circuit = {
    def resolve_kinds (m:DefModule, c:Circuit):DefModule = {
      val kinds = LinkedHashMap[String,Kind]()
      def resolve (body:Statement) = {
        def resolve_expr (e:Expression):Expression = {
          e match {
             case e:WRef => WRef(e.name,e.tpe,kinds(e.name),e.gender)
             case e => e map (resolve_expr)
          }
        }
        def resolve_stmt (s:Statement):Statement = s map (resolve_stmt) map (resolve_expr)
        resolve_stmt(body)
      }
 
      def find (m:DefModule) = {
        def find_stmt (s:Statement):Statement = {
          s match {
            case s:DefWire => kinds(s.name) = WireKind()
            case s:DefNode => kinds(s.name) = NodeKind()
            case s:DefRegister => kinds(s.name) = RegKind()
            case s:WDefInstance => kinds(s.name) = InstanceKind()
            case s:DefMemory => kinds(s.name) = MemKind(s.readers ++ s.writers ++ s.readwriters)
            case s => false
          }
          s map (find_stmt)
        }
        m.ports.foreach { p => kinds(p.name) = PortKind() }
        m match {
          case m:Module => find_stmt(m.body)
          case m:ExtModule => false
        }
      }
      
      mname = m.name
      find(m)   
      m match {
         case m:Module => {
            val bodyx = resolve(m.body)
            Module(m.info,m.name,m.ports,bodyx)
         }
         case m:ExtModule => ExtModule(m.info,m.name,m.ports)
      }
    }
    val modulesx = c.modules.map(m => resolve_kinds(m,c))
    Circuit(c.info,modulesx,c.main)
  }
}

object ResolveGenders extends Pass {
  private var mname = ""
  def name = "Resolve Genders"
  def run (c:Circuit): Circuit = {
    def resolve_e (g:Gender)(e:Expression) : Expression = {
      e match {
        case e:WRef => WRef(e.name,e.tpe,e.kind,g)
        case e:WSubField => {
          val expx = 
            field_flip(e.exp.tpe,e.name) match {
               case Default => resolve_e(g)(e.exp)
               case Flip => resolve_e(swap(g))(e.exp)
            }
          WSubField(expx,e.name,e.tpe,g)
        }
        case e:WSubIndex => {
          val expx = resolve_e(g)(e.exp)
          WSubIndex(expx,e.value,e.tpe,g)
        }
        case e:WSubAccess => {
          val expx = resolve_e(g)(e.exp)
          val indexx = resolve_e(MALE)(e.index)
          WSubAccess(expx,indexx,e.tpe,g)
        }
        case e => e map (resolve_e(g))
      }
    }
          
    def resolve_s (s:Statement) : Statement = {
      s match {
        case s:IsInvalid => {
          val expx = resolve_e(FEMALE)(s.expr)
          IsInvalid(s.info,expx)
        }
        case s:Connect => {
          val locx = resolve_e(FEMALE)(s.loc)
          val expx = resolve_e(MALE)(s.expr)
          Connect(s.info,locx,expx)
        }
        case s:PartialConnect => {
          val locx = resolve_e(FEMALE)(s.loc)
          val expx = resolve_e(MALE)(s.expr)
          PartialConnect(s.info,locx,expx)
        }
        case s => s map (resolve_e(MALE)) map (resolve_s)
      }
    }
    val modulesx = c.modules.map { 
      m => {
        mname = m.name
        m match {
          case m:Module => {
            val bodyx = resolve_s(m.body)
            Module(m.info,m.name,m.ports,bodyx)
          }
          case m:ExtModule => m
        }
      }
    }
    Circuit(c.info,modulesx,c.main)
  }
}

object CInferMDir extends Pass {
  def name = "CInfer MDir"
  var mname = ""
  def run (c:Circuit) : Circuit = {
    def infer_mdir (m:DefModule) : DefModule = {
       val mports = LinkedHashMap[String,MPortDir]()
       def infer_mdir_e (dir:MPortDir)(e:Expression) : Expression = {
         (e map (infer_mdir_e(dir))) match { 
           case (e:Reference) => {
             if (mports.contains(e.name)) {
                val new_mport_dir = {
                  (mports(e.name),dir) match {
                    case (MInfer,MInfer) => error("Shouldn't be here")
                    case (MInfer,MWrite) => MWrite
                    case (MInfer,MRead) => MRead
                    case (MInfer,MReadWrite) => MReadWrite
                    case (MWrite,MInfer) => error("Shouldn't be here")
                    case (MWrite,MWrite) => MWrite
                    case (MWrite,MRead) => MReadWrite
                    case (MWrite,MReadWrite) => MReadWrite
                    case (MRead,MInfer) => error("Shouldn't be here")
                    case (MRead,MWrite) => MReadWrite
                    case (MRead,MRead) => MRead
                    case (MRead,MReadWrite) => MReadWrite
                    case (MReadWrite,MInfer) => error("Shouldn't be here")
                    case (MReadWrite,MWrite) => MReadWrite
                    case (MReadWrite,MRead) => MReadWrite
                    case (MReadWrite,MReadWrite) => MReadWrite
                  }
                }
                mports(e.name) = new_mport_dir
              }
              e
            }
            case (e) => e
          }
       }
      def infer_mdir_s (s:Statement) : Statement = {
        (s) match { 
          case (s:CDefMPort) => {
            mports(s.name) = s.direction
            s map (infer_mdir_e(MRead))
          }
          case (s:Connect) => {
            infer_mdir_e(MRead)(s.expr)
            infer_mdir_e(MWrite)(s.loc)
            s
          }
          case (s:PartialConnect) => {
            infer_mdir_e(MRead)(s.expr)
            infer_mdir_e(MWrite)(s.loc)
            s
          }
          case (s) => s map (infer_mdir_s) map (infer_mdir_e(MRead))
        }
      }
      def set_mdir_s (s:Statement) : Statement = {
        (s) match { 
          case (s:CDefMPort) => 
            CDefMPort(s.info,s.name,s.tpe,s.mem,s.exps,mports(s.name))
          case (s) => s map (set_mdir_s)
        }
      }
      (m) match { 
        case (m:Module) => {
          infer_mdir_s(m.body)
          Module(m.info,m.name,m.ports,set_mdir_s(m.body))
        }
        case (m:ExtModule) => m
      }
    }
 
    //; MAIN
    Circuit(c.info, c.modules.map(m => infer_mdir(m)), c.main)
  }
}
