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

// Datastructures
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.WrappedType._


object CheckChirrtl extends Pass with LazyLogging {
  def name = "Chirrtl Check"

  // TODO FIXME
  // - Do we need to check for uniquness on port names?
  def run (c: Circuit): Circuit = {
    var mname: String = ""
    var sinfo: Info = NoInfo

    class NotUniqueException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Reference ${name} does not have a unique name.")
    class InvalidLOCException extends PassException(s"${sinfo}: [module ${mname}] Invalid connect to an expression that is not a reference or a WritePort.")
    class UndeclaredReferenceException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Reference ${name} is not declared.")
    class MemWithFlipException(name: String) extends PassException(s"${sinfo}: [module ${mname}] Memory ${name} cannot be a bundle type with flips.")
    class InvalidAccessException extends PassException(s"${sinfo}: [module ${mname}] Invalid access to non-reference.")
    class NoTopModuleException(name: String) extends PassException(s"${sinfo}: A single module must be named ${name}.")
    class ModuleNotDefinedException(name: String) extends PassException(s"${sinfo}: Module ${name} is not defined.")
    class NegWidthException extends PassException(s"${sinfo}: [module ${mname}] Width cannot be negative or zero.")
    class NegVecSizeException extends PassException(s"${sinfo}: [module ${mname}] Vector type size cannot be negative.")
    class NegMemSizeException extends PassException(s"${sinfo}: [module ${mname}] Memory size cannot be negative or zero.")

    val errors = new Errors()
    def checkValidLoc(e: Expression) = e match {
      case e @ (_: UIntLiteral | _: SIntLiteral | _: DoPrim ) => errors.append(new InvalidLOCException)
      case _ => // Do Nothing
    }
    def checkChirrtlW(w: Width): Width = w match {
      case w: IntWidth if (w.width <= BigInt(0)) =>
        errors.append(new NegWidthException)
        w
      case _ => w
    }
    def checkChirrtlT(t: Type): Type = {
      t map (checkChirrtlT) match {
        case t: VectorType if (t.size < 0) => errors.append(new NegVecSizeException)
        case _ => // Do nothing
      }
      t map (checkChirrtlW)
    }

    def checkChirrtlM(m: DefModule): DefModule = {
      val names = HashMap[String, Boolean]()
      val mnames = HashMap[String, Boolean]()
      def checkChirrtlE(e: Expression): Expression = {
        def validSubexp(e: Expression): Expression = e match {
          case (_:Reference|_:SubField|_:SubIndex|_:SubAccess|_:Mux|_:ValidIf) => e // No error
          case _ => 
            errors.append(new InvalidAccessException)
            e
        }
        e map (checkChirrtlE) match {
          case e: Reference if (!names.contains(e.name)) => errors.append(new UndeclaredReferenceException(e.name))
          case e: DoPrim => {}
          case (_:Mux|_:ValidIf) => {}
          case e: SubAccess =>
            validSubexp(e.expr)
            e
          case e: UIntLiteral => {}
          case e => e map (validSubexp)
        }
        e map (checkChirrtlW)
        e map (checkChirrtlT)
        e
      }
      def checkChirrtlS(s: Statement): Statement = {
        sinfo = s.getInfo
        def checkName(name: String): String = {
          if (names.contains(name)) errors.append(new NotUniqueException(name))
          else names(name) = true
          name 
        }

        s map (checkName)
        s map (checkChirrtlT)
        s map (checkChirrtlE)
        s match {
          case s: DefMemory =>
            if (hasFlip(s.dataType)) errors.append(new MemWithFlipException(s.name))
            if (s.depth <= 0) errors.append(new NegMemSizeException)
          case s: DefInstance =>
            if (!c.modules.map(_.name).contains(s.module))
              errors.append(new ModuleNotDefinedException(s.module))
          case s: Connect => checkValidLoc(s.loc)
          case s: PartialConnect => checkValidLoc(s.loc)
          case s: Print => {}
          case _ => // Do Nothing
        }

        s map (checkChirrtlS)
      }

      mname = m.name
      for (m <- c.modules) {
        mnames(m.name) = true
      }
      for (p <- m.ports) {
        sinfo = p.info
        names(p.name) = true
        val tpe = p.getType
        tpe map (checkChirrtlT)
        tpe map (checkChirrtlW)
      }

      m match {
        case m: Module => checkChirrtlS(m.body)
        case m: ExtModule => // Do Nothing
      }
      m
    }
    
    var numTopM = 0
    for (m <- c.modules) {
      if (m.name == c.main) numTopM = numTopM + 1
      checkChirrtlM(m)
    }
    sinfo = c.info
    if (numTopM != 1) errors.append(new NoTopModuleException(c.main))
    errors.trigger
    c
  }
}

// vim: set ts=4 sw=4 et:
