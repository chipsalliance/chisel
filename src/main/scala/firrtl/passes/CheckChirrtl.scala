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

object CheckChirrtl extends Pass {
  def name = "Chirrtl Check"
  type NameSet = collection.mutable.HashSet[String]

  class NotUniqueException(info: Info, mname: String, name: String) extends PassException(
    s"${info}: [module ${mname}] Reference ${name} does not have a unique name.")
  class InvalidLOCException(info: Info, mname: String) extends PassException(
    s"${info}: [module ${mname}] Invalid connect to an expression that is not a reference or a WritePort.")
  class UndeclaredReferenceException(info: Info, mname: String, name: String) extends PassException(
    s"${info}: [module ${mname}] Reference ${name} is not declared.")
  class MemWithFlipException(info: Info, mname: String, name: String) extends PassException(
    s"${info}: [module ${mname}] Memory ${name} cannot be a bundle type with flips.")
  class InvalidAccessException(info: Info, mname: String) extends PassException(
    s"${info}: [module ${mname}] Invalid access to non-reference.")
  class ModuleNotDefinedException(info: Info, mname: String, name: String) extends PassException(
    s"${info}: Module ${name} is not defined.")
  class NegWidthException(info: Info, mname: String) extends PassException(
    s"${info}: [module ${mname}] Width cannot be negative or zero.")
  class NegVecSizeException(info: Info, mname: String) extends PassException(
    s"${info}: [module ${mname}] Vector type size cannot be negative.")
  class NegMemSizeException(info: Info, mname: String) extends PassException(
    s"${info}: [module ${mname}] Memory size cannot be negative or zero.")
  class NoTopModuleException(info: Info, name: String) extends PassException(
    s"${info}: A single module must be named ${name}.")

  // TODO FIXME
  // - Do we need to check for uniquness on port names?
  def run (c: Circuit): Circuit = {
    val errors = new Errors()
    val moduleNames = (c.modules map (_.name)).toSet

    def checkValidLoc(info: Info, mname: String, e: Expression) = e match {
      case _: UIntLiteral | _: SIntLiteral | _: DoPrim =>
        errors append new InvalidLOCException(info, mname)
      case _ => // Do Nothing
    }

    def checkChirrtlW(info: Info, mname: String)(w: Width): Width = w match {
      case w: IntWidth if w.width <= 0 =>
        errors append new NegWidthException(info, mname)
        w
      case _ => w
    }

    def checkChirrtlT(info: Info, mname: String)(t: Type): Type = {
      t map checkChirrtlT(info, mname) match {
        case t: VectorType if t.size < 0 =>
          errors append new NegVecSizeException(info, mname)
        case _ => // Do nothing
      }
      t map checkChirrtlW(info, mname) map checkChirrtlT(info, mname)
    }

    def validSubexp(info: Info, mname: String)(e: Expression): Expression = {
      e match {
        case _: Reference | _: SubField | _: SubIndex | _: SubAccess |
             _: Mux | _: ValidIf => // No error
        case _ => errors append new InvalidAccessException(info, mname)
      }
      e
    }

    def checkChirrtlE(info: Info, mname: String, names: NameSet)(e: Expression): Expression = {
      e match {
        case _: DoPrim | _:Mux | _:ValidIf | _: UIntLiteral =>
        case e: Reference if !names(e.name) =>
          errors append new UndeclaredReferenceException(info, mname, e.name)
        case e: SubAccess => validSubexp(info, mname)(e.expr)
        case e => e map validSubexp(info, mname)
      }
      (e map checkChirrtlW(info, mname)
         map checkChirrtlT(info, mname)
         map checkChirrtlE(info, mname, names))
    }

    def checkName(info: Info, mname: String, names: NameSet)(name: String): String = {
      if (names(name))
        errors append (new NotUniqueException(info, mname, name))
      names += name
      name 
    }

    def checkChirrtlS(minfo: Info, mname: String, names: NameSet)(s: Statement): Statement = {
      val info = get_info(s) match {case NoInfo => minfo case x => x}
      (s map checkName(info, mname, names)) match {
        case s: DefMemory =>
          if (hasFlip(s.dataType)) errors append new MemWithFlipException(info, mname, s.name)
          if (s.depth <= 0) errors append new NegMemSizeException(info, mname)
        case s: DefInstance if !moduleNames(s.module) =>
          errors append new ModuleNotDefinedException(info, mname, s.module)
        case s: Connect => checkValidLoc(info, mname, s.loc)
        case s: PartialConnect => checkValidLoc(info, mname, s.loc)
        case _ => // Do Nothing
      }
      (s map checkChirrtlT(info, mname)
         map checkChirrtlE(info, mname, names)
         map checkChirrtlS(info, mname, names))
    }

    def checkChirrtlP(mname: String, names: NameSet)(p: Port): Port = {
      names += p.name
      (p.tpe map checkChirrtlT(p.info, mname)
             map checkChirrtlW(p.info, mname))
      p
    }

    def checkChirrtlM(m: DefModule) {
      val names = new NameSet
      (m map checkChirrtlP(m.name, names)
         map checkChirrtlS(m.info, m.name, names))
    }
    
    c.modules foreach checkChirrtlM
    (c.modules filter (_.name == c.main)).size match {
      case 1 =>
      case _ => errors append new NoTopModuleException(c.info, c.main)
    }
    errors.trigger
    c
  }
}
