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
import firrtl.Mappers._

object ResolveKinds extends Pass {
  def name = "Resolve Kinds"
  type KindMap = collection.mutable.LinkedHashMap[String, Kind]

  def find_port(kinds: KindMap)(p: Port): Port = {
    kinds(p.name) = PortKind ; p
  }

  def find_stmt(kinds: KindMap)(s: Statement):Statement = {
    s match {
      case s: DefWire => kinds(s.name) = WireKind
      case s: DefNode => kinds(s.name) = NodeKind
      case s: DefRegister => kinds(s.name) = RegKind
      case s: WDefInstance => kinds(s.name) = InstanceKind
      case s: DefMemory => kinds(s.name) = MemKind
      case s =>
    } 
    s map find_stmt(kinds)
  }

  def resolve_expr(kinds: KindMap)(e: Expression): Expression = e match {
    case e: WRef => e copy (kind = kinds(e.name))
    case e => e map resolve_expr(kinds)
  }

  def resolve_stmt(kinds: KindMap)(s: Statement): Statement =
    s map resolve_stmt(kinds) map resolve_expr(kinds)

  def resolve_kinds(m: DefModule): DefModule = {
    val kinds = new KindMap
    (m map find_port(kinds)
       map find_stmt(kinds)
       map resolve_stmt(kinds))
  }
 
  def run(c: Circuit): Circuit =
    c copy (modules = c.modules map resolve_kinds)
}

object ResolveGenders extends Pass {
  def name = "Resolve Genders"
  def resolve_e(g: Gender)(e: Expression): Expression = e match {
    case e: WRef => e copy (gender = g)
    case WSubField(exp, name, tpe, _) => WSubField(
      Utils.field_flip(exp.tpe, name) match {
        case Default => resolve_e(g)(exp)
        case Flip => resolve_e(Utils.swap(g))(exp)
      }, name, tpe, g)
    case WSubIndex(exp, value, tpe, _) =>
      WSubIndex(resolve_e(g)(exp), value, tpe, g)
    case WSubAccess(exp, index, tpe, _) =>
      WSubAccess(resolve_e(g)(exp), resolve_e(MALE)(index), tpe, g)
    case e => e map resolve_e(g)
  }
        
  def resolve_s(s: Statement): Statement = s match {
    //TODO(azidar): pretty sure don't need to do anything for Attach, but not positive...
    case IsInvalid(info, expr) =>
      IsInvalid(info, resolve_e(FEMALE)(expr))
    case Connect(info, loc, expr) =>
      Connect(info, resolve_e(FEMALE)(loc), resolve_e(MALE)(expr))
    case PartialConnect(info, loc, expr) =>
      PartialConnect(info, resolve_e(FEMALE)(loc), resolve_e(MALE)(expr))
    case s => s map resolve_e(MALE) map resolve_s
  }

  def resolve_gender(m: DefModule): DefModule = m map resolve_s

  def run(c: Circuit): Circuit =
    c copy (modules = c.modules map resolve_gender)
}

object CInferMDir extends Pass {
  def name = "CInfer MDir"
  type MPortDirMap = collection.mutable.LinkedHashMap[String, MPortDir]

  def infer_mdir_e(mports: MPortDirMap, dir: MPortDir)(e: Expression): Expression = {
    e map infer_mdir_e(mports, dir) match {
      case e: Reference => mports get e.name match {
        case Some(p) => mports(e.name) = (p, dir) match {
          case (MInfer, MInfer) => Utils.error("Shouldn't be here")
          case (MInfer, MWrite) => MWrite
          case (MInfer, MRead) => MRead
          case (MInfer, MReadWrite) => MReadWrite
          case (MWrite, MInfer) => Utils.error("Shouldn't be here")
          case (MWrite, MWrite) => MWrite
          case (MWrite, MRead) => MReadWrite
          case (MWrite, MReadWrite) => MReadWrite
          case (MRead, MInfer) => Utils.error("Shouldn't be here")
          case (MRead, MWrite) => MReadWrite
          case (MRead, MRead) => MRead
          case (MRead, MReadWrite) => MReadWrite
          case (MReadWrite, MInfer) => Utils.error("Shouldn't be here")
          case (MReadWrite, MWrite) => MReadWrite
          case (MReadWrite, MRead) => MReadWrite
          case (MReadWrite, MReadWrite) => MReadWrite
        } ; e
        case None => e
      }
      case _ => e
    }
  }

  def infer_mdir_s(mports: MPortDirMap)(s: Statement): Statement = s match { 
    case s: CDefMPort =>
       mports(s.name) = s.direction
       s map infer_mdir_e(mports, MRead)
    case s: Connect =>
       infer_mdir_e(mports, MRead)(s.expr)
       infer_mdir_e(mports, MWrite)(s.loc)
       s
    case s: PartialConnect =>
       infer_mdir_e(mports, MRead)(s.expr)
       infer_mdir_e(mports, MWrite)(s.loc)
       s
    case s => s map infer_mdir_s(mports) map infer_mdir_e(mports, MRead)
  }
        
  def set_mdir_s(mports: MPortDirMap)(s: Statement): Statement = s match { 
    case s: CDefMPort => s copy (direction = mports(s.name))
    case s => s map set_mdir_s(mports)
  }
  
  def infer_mdir(m: DefModule): DefModule = {
    val mports = new MPortDirMap
    m map infer_mdir_s(mports) map set_mdir_s(mports)
  }
     
  def run(c: Circuit): Circuit =
    c copy (modules = c.modules map infer_mdir)
}
