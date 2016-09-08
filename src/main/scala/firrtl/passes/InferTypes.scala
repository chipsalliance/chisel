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

object InferTypes extends Pass {
  private var mname = ""
  def name = "Infer Types"
  def set_type (s:Statement, t:Type) : Statement = {
    s match {
       case s:DefWire => DefWire(s.info,s.name,t)
       case s:DefRegister => DefRegister(s.info,s.name,t,s.clock,s.reset,s.init)
       case s:DefMemory => DefMemory(s.info,s.name,t,s.depth,s.writeLatency,s.readLatency,s.readers,s.writers,s.readwriters)
       case s:DefNode => s
    }
  }
  def remove_unknowns_w (w:Width)(implicit namespace: Namespace):Width = {
    w match {
       case UnknownWidth => VarWidth(namespace.newName("w"))
       case w => w
    }
  }
  def remove_unknowns (t:Type)(implicit n: Namespace): Type = mapr(remove_unknowns_w _,t)
  def run (c:Circuit): Circuit = {
    val module_types = LinkedHashMap[String,Type]()
    implicit val wnamespace = Namespace()
    def infer_types (m:DefModule) : DefModule = {
      val types = LinkedHashMap[String,Type]()
      def infer_types_e (e:Expression) : Expression = {
        e map (infer_types_e) match {
          case e:ValidIf => ValidIf(e.cond,e.value,e.value.tpe)
          case e:WRef => WRef(e.name, types(e.name),e.kind,e.gender)
          case e:WSubField => WSubField(e.exp,e.name,field_type(e.exp.tpe,e.name),e.gender)
          case e:WSubIndex => WSubIndex(e.exp,e.value,sub_type(e.exp.tpe),e.gender)
          case e:WSubAccess => WSubAccess(e.exp,e.index,sub_type(e.exp.tpe),e.gender)
          case e:DoPrim => set_primop_type(e)
          case e:Mux => Mux(e.cond,e.tval,e.fval,mux_type_and_widths(e.tval,e.fval))
          case e:UIntLiteral => e
          case e:SIntLiteral => e
        }
      }
      def infer_types_s (s:Statement) : Statement = {
        s match {
          case s:DefRegister => {
            val t = remove_unknowns(get_type(s))
            types(s.name) = t
            set_type(s,t) map (infer_types_e)
          }
          case s:DefWire => {
            val sx = s map(infer_types_e)
            val t = remove_unknowns(get_type(sx))
            types(s.name) = t
            set_type(sx,t)
          }
          case s:DefNode => {
            val sx = s map (infer_types_e)
            val t = remove_unknowns(get_type(sx))
            types(s.name) = t
            set_type(sx,t)
          }
          case s:DefMemory => {
            val t = remove_unknowns(get_type(s))
            types(s.name) = t
            val dt = remove_unknowns(s.dataType)
            set_type(s,dt)
          }
          case s:WDefInstance => {
            types(s.name) = module_types(s.module)
            WDefInstance(s.info,s.name,s.module,module_types(s.module))
          }
          case s => s map (infer_types_s) map (infer_types_e)
        }
      }

      mname = m.name
      m.ports.foreach(p => types(p.name) = p.tpe)
      m match {
         case m:Module => Module(m.info,m.name,m.ports,infer_types_s(m.body))
         case m:ExtModule => m
      }
    }

    val modulesx = c.modules.map { 
      m => {
        mname = m.name
        val portsx = m.ports.map(p => Port(p.info,p.name,p.direction,remove_unknowns(p.tpe)))
        m match {
          case m:Module => Module(m.info,m.name,portsx,m.body)
          case m:ExtModule => ExtModule(m.info,m.name,portsx)
        }
      }
    }
    modulesx.foreach(m => module_types(m.name) = module_type(m))
    Circuit(c.info,modulesx.map({m => mname = m.name; infer_types(m)}) , c.main )
  }
}

object CInferTypes extends Pass {
  def name = "CInfer Types"
  var mname = ""
  def set_type (s:Statement, t:Type) : Statement = {
    (s) match { 
      case (s:DefWire) => DefWire(s.info,s.name,t)
      case (s:DefRegister) => DefRegister(s.info,s.name,t,s.clock,s.reset,s.init)
      case (s:CDefMemory) => CDefMemory(s.info,s.name,t,s.size,s.seq)
      case (s:CDefMPort) => CDefMPort(s.info,s.name,t,s.mem,s.exps,s.direction)
      case (s:DefNode) => s
    }
  }
  
  def to_field (p:Port) : Field = {
    if (p.direction == Output) Field(p.name,Default,p.tpe)
    else if (p.direction == Input) Field(p.name,Flip,p.tpe)
    else error("Shouldn't be here"); Field(p.name,Flip,p.tpe)
  }
  def module_type (m:DefModule) : Type = BundleType(m.ports.map(p => to_field(p)))
  def field_type (v:Type,s:String) : Type = {
    (v) match { 
      case (v:BundleType) => {
        val ft = v.fields.find(p => p.name == s)
        if (ft != None) ft.get.tpe
        else  UnknownType
      }
      case (v) => UnknownType
    }
  }
  def sub_type (v:Type) : Type =
    (v) match { 
      case (v:VectorType) => v.tpe
      case (v) => UnknownType
    }
  def run (c:Circuit) : Circuit = {
    val module_types = LinkedHashMap[String,Type]()
    def infer_types (m:DefModule) : DefModule = {
      val types = LinkedHashMap[String,Type]()
      def infer_types_e (e:Expression) : Expression = {
        e map infer_types_e match {
          case (e:Reference) => Reference(e.name, types.getOrElse(e.name,UnknownType))
          case (e:SubField) => SubField(e.expr,e.name,field_type(e.expr.tpe,e.name))
          case (e:SubIndex) => SubIndex(e.expr,e.value,sub_type(e.expr.tpe))
          case (e:SubAccess) => SubAccess(e.expr,e.index,sub_type(e.expr.tpe))
          case (e:DoPrim) => set_primop_type(e)
          case (e:Mux) => Mux(e.cond,e.tval,e.fval,mux_type(e.tval,e.tval))
          case (e:ValidIf) => ValidIf(e.cond,e.value,e.value.tpe)
          case (_:UIntLiteral | _:SIntLiteral) => e
        }
      }
      def infer_types_s (s:Statement) : Statement = {
        s match {
          case (s:DefRegister) => {
             types(s.name) = s.tpe
             s map infer_types_e
             s
          }
          case (s:DefWire) => {
             types(s.name) = s.tpe
             s
          }
          case (s:DefNode) => {
             val sx = s map infer_types_e
             val t = get_type(sx)
             types(s.name) = t
             sx
          }
          case (s:DefMemory) => {
             types(s.name) = get_type(s)
             s
          }
          case (s:CDefMPort) => {
             val t = types.getOrElse(s.mem,UnknownType)
             types(s.name) = t
             CDefMPort(s.info,s.name,t,s.mem,s.exps,s.direction)
          }
          case (s:CDefMemory) => {
             types(s.name) = s.tpe
             s
          }
          case (s:DefInstance) => {
             types(s.name) = module_types.getOrElse(s.module,UnknownType)
             s
          }
          case (s) => s map infer_types_s map infer_types_e
        }
      }
      for (p <- m.ports) {
        types(p.name) = p.tpe
      }
      m match {
         case (m:Module) => Module(m.info,m.name,m.ports,infer_types_s(m.body))
         case (m:ExtModule) => m
      }
    }
  
    //; MAIN
    for (m <- c.modules) {
      module_types(m.name) = module_type(m)
    }
    val modulesx = c.modules.map(m => infer_types(m))
    Circuit(c.info, modulesx, c.main)
  }
}
