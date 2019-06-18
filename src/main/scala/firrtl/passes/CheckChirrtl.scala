// See LICENSE for license details.

package firrtl.passes

import firrtl.ir._
import firrtl.Utils._
import firrtl.traversals.Foreachers._

object CheckChirrtl extends Pass {
  type NameSet = collection.mutable.HashSet[String]

  class NotUniqueException(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname] Reference $name does not have a unique name.")
  class InvalidLOCException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Invalid connect to an expression that is not a reference or a WritePort.")
  class UndeclaredReferenceException(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname] Reference $name is not declared.")
  class MemWithFlipException(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname] Memory $name cannot be a bundle type with flips.")
  class InvalidAccessException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Invalid access to non-reference.")
  class ModuleNotDefinedException(info: Info, mname: String, name: String) extends PassException(
    s"$info: Module $name is not defined.")
  class NegWidthException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Width cannot be negative or zero.")
  class NegVecSizeException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Vector type size cannot be negative.")
  class NegMemSizeException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Memory size cannot be negative or zero.")
  class NoTopModuleException(info: Info, name: String) extends PassException(
    s"$info: A single module must be named $name.")

  def run (c: Circuit): Circuit = {
    val errors = new Errors()
    val moduleNames = (c.modules map (_.name)).toSet

    def checkValidLoc(info: Info, mname: String, e: Expression) = e match {
      case _: UIntLiteral | _: SIntLiteral | _: DoPrim =>
        errors append new InvalidLOCException(info, mname)
      case _ => // Do Nothing
    }
    def checkChirrtlW(info: Info, mname: String)(w: Width): Unit = w match {
      case w: IntWidth if (w.width < BigInt(0)) => errors.append(new NegWidthException(info, mname))
      case _ =>
    }

    def checkChirrtlT(info: Info, mname: String)(t: Type): Unit = {
      t.foreach(checkChirrtlT(info, mname))
      t match {
        case t: VectorType if t.size < 0 =>
          errors append new NegVecSizeException(info, mname)
          t.foreach(checkChirrtlW(info, mname))
        //case FixedType(width, point) => FixedType(checkChirrtlW(width), point)
        case _ => t.foreach(checkChirrtlW(info, mname))
      }
    }

    def validSubexp(info: Info, mname: String)(e: Expression): Unit = e match {
      case _: Reference | _: SubField | _: SubIndex | _: SubAccess |
           _: Mux | _: ValidIf => // No error
      case _ => errors append new InvalidAccessException(info, mname)
    }

    def checkChirrtlE(info: Info, mname: String, names: NameSet)(e: Expression): Unit = {
      e match {
        case _: DoPrim | _:Mux | _:ValidIf | _: UIntLiteral =>
        case ex: Reference if !names(ex.name) =>
          errors append new UndeclaredReferenceException(info, mname, ex.name)
        case ex: SubAccess => validSubexp(info, mname)(ex.expr)
        case ex => ex.foreach(validSubexp(info, mname))
      }
      e.foreach(checkChirrtlW(info, mname))
      e.foreach(checkChirrtlT(info, mname))
      e.foreach(checkChirrtlE(info, mname, names))
    }

    def checkName(info: Info, mname: String, names: NameSet)(name: String): Unit = {
      if (names(name))
        errors append new NotUniqueException(info, mname, name)
      names += name
    }

    def checkChirrtlS(minfo: Info, mname: String, names: NameSet)(s: Statement): Unit = {
      val info = get_info(s) match {case NoInfo => minfo case x => x}
      s.foreach(checkName(info, mname, names))
      s match {
        case sx: DefMemory =>
          if (hasFlip(sx.dataType)) errors append new MemWithFlipException(info, mname, sx.name)
          if (sx.depth <= 0) errors append new NegMemSizeException(info, mname)
        case sx: DefInstance if !moduleNames(sx.module) =>
          errors append new ModuleNotDefinedException(info, mname, sx.module)
        case sx: Connect => checkValidLoc(info, mname, sx.loc)
        case sx: PartialConnect => checkValidLoc(info, mname, sx.loc)
        case _ => // Do Nothing
      }
      s.foreach(checkChirrtlT(info, mname))
      s.foreach(checkChirrtlE(info, mname, names))
      s.foreach(checkChirrtlS(info, mname, names))
    }

    def checkChirrtlP(mname: String, names: NameSet)(p: Port): Unit = {
      if (names(p.name))
        errors append new NotUniqueException(NoInfo, mname, p.name)
      names += p.name
      p.tpe.foreach(checkChirrtlT(p.info, mname))
      p.tpe.foreach(checkChirrtlW(p.info, mname))
    }

    def checkChirrtlM(m: DefModule): Unit = {
      val names = new NameSet
      m.foreach(checkChirrtlP(m.name, names))
      m.foreach(checkChirrtlS(m.info, m.name, names))
    }
    
    c.modules.foreach(checkChirrtlM)
    c.modules count (_.name == c.main) match {
      case 1 =>
      case _ => errors append new NoTopModuleException(c.info, c.main)
    }
    errors.trigger()
    c
  }
}
