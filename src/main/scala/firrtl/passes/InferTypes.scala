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

object InferTypes extends Pass {
  def name = "Infer Types"
  type TypeMap = collection.mutable.LinkedHashMap[String, Type]

  def run(c: Circuit): Circuit = {
    val namespace = Namespace()
    val mtypes = (c.modules map (m => m.name -> module_type(m))).toMap

    def remove_unknowns_w(w: Width): Width = w match {
      case UnknownWidth => VarWidth(namespace.newName("w"))
      case w => w
    }

    def remove_unknowns(t: Type): Type =
      t map remove_unknowns map remove_unknowns_w

    def infer_types_e(types: TypeMap)(e: Expression): Expression =
      e map infer_types_e(types) match {
        case e: WRef => e copy (tpe = types(e.name))
        case e: WSubField => e copy (tpe = field_type(e.exp.tpe, e.name))
        case e: WSubIndex => e copy (tpe = sub_type(e.exp.tpe))
        case e: WSubAccess => e copy (tpe = sub_type(e.exp.tpe))
        case e: DoPrim => PrimOps.set_primop_type(e)
        case e: Mux => e copy (tpe = mux_type_and_widths(e.tval, e.fval))
        case e: ValidIf => e copy (tpe = e.value.tpe)
        case e @ (_: UIntLiteral | _: SIntLiteral) => e
      }

    def infer_types_s(types: TypeMap)(s: Statement): Statement = s match {
      case s: WDefInstance =>
        val t = mtypes(s.module)
        types(s.name) = t
        s copy (tpe = t)
      case s: DefWire =>
        val t = remove_unknowns(s.tpe)
        types(s.name) = t
        s copy (tpe = t)
      case s: DefNode =>
        val sx = (s map infer_types_e(types)).asInstanceOf[DefNode]
        val t = remove_unknowns(sx.value.tpe)
        types(s.name) = t
        sx map infer_types_e(types)
      case s: DefRegister =>
        val t = remove_unknowns(s.tpe)
        types(s.name) = t
        s copy (tpe = t) map infer_types_e(types)
      case s: DefMemory =>
        val t = remove_unknowns(MemPortUtils.memType(s))
        types(s.name) = t
        s copy (dataType = remove_unknowns(s.dataType))
      case s => s map infer_types_s(types) map infer_types_e(types)
    }

    def infer_types_p(types: TypeMap)(p: Port): Port = {
      val t = remove_unknowns(p.tpe)
      types(p.name) = t
      p copy (tpe = t)
    }

    def infer_types(m: DefModule): DefModule = {
      val types = new TypeMap
      m map infer_types_p(types) map infer_types_s(types)
    }
 
    c copy (modules = (c.modules map infer_types))
  }
}

object CInferTypes extends Pass {
  def name = "CInfer Types"
  type TypeMap = collection.mutable.LinkedHashMap[String, Type]

  def run(c: Circuit): Circuit = {
    val namespace = Namespace()
    val mtypes = (c.modules map (m => m.name -> module_type(m))).toMap

    def infer_types_e(types: TypeMap)(e: Expression) : Expression =
      e map infer_types_e(types) match {
         case (e: Reference) => e copy (tpe = (types getOrElse (e.name, UnknownType)))
         case (e: SubField) => e copy (tpe = field_type(e.expr.tpe, e.name))
         case (e: SubIndex) => e copy (tpe = sub_type(e.expr.tpe))
         case (e: SubAccess) => e copy (tpe = sub_type(e.expr.tpe))
         case (e: DoPrim) => PrimOps.set_primop_type(e)
         case (e: Mux) => e copy (tpe = mux_type(e.tval,e.tval))
         case (e: ValidIf) => e copy (tpe = e.value.tpe)
         case e @ (_: UIntLiteral | _: SIntLiteral) => e
      }

    def infer_types_s(types: TypeMap)(s: Statement): Statement = s match {
      case (s: DefRegister) =>
        types(s.name) = s.tpe
        s map infer_types_e(types)
      case (s: DefWire) =>
        types(s.name) = s.tpe
        s
      case (s: DefNode) =>
        types(s.name) = s.value.tpe
        s
      case (s: DefMemory) =>
        types(s.name) = MemPortUtils.memType(s)
        s
      case (s: CDefMPort) =>
        val t = types getOrElse(s.mem, UnknownType)
        types(s.name) = t
        s copy (tpe = t)
      case (s: CDefMemory) =>
        types(s.name) = s.tpe
        s
      case (s: DefInstance) =>
        types(s.name) = mtypes(s.module)
        s
      case (s) => s map infer_types_s(types) map infer_types_e(types)
    }

    def infer_types_p(types: TypeMap)(p: Port): Port = {
      types(p.name) = p.tpe
      p
    }
 
    def infer_types(m: DefModule): DefModule = {
      val types = new TypeMap
      m map infer_types_p(types) map infer_types_s(types)
    }
   
    c copy (modules = (c.modules map infer_types))
  }
}
