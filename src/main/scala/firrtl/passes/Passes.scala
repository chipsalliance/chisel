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

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._

trait Pass extends LazyLogging {
  def name: String
  def run(c: Circuit): Circuit
}

// Error handling
class PassException(message: String) extends Exception(message)
class PassExceptions(exceptions: Seq[PassException]) extends Exception("\n" + exceptions.mkString("\n"))
class Errors {
  val errors = collection.mutable.ArrayBuffer[PassException]()
  def append(pe: PassException) = errors.append(pe)
  def trigger = errors.size match {
    case 0 =>
    case 1 => throw errors.head
    case _ =>
      append(new PassException(s"${errors.length} errors detected!"))
      throw new PassExceptions(errors)
  }
}

// These should be distributed into separate files
object ToWorkingIR extends Pass {
  def name = "Working IR"

  def toExp(e: Expression): Expression = e map (toExp) match {
    case e: Reference => WRef(e.name, e.tpe, NodeKind, UNKNOWNGENDER)
    case e: SubField => WSubField(e.expr, e.name, e.tpe, UNKNOWNGENDER)
    case e: SubIndex => WSubIndex(e.expr, e.value, e.tpe, UNKNOWNGENDER)
    case e: SubAccess => WSubAccess(e.expr, e.index, e.tpe, UNKNOWNGENDER)
    case e => e
  }

  def toStmt(s: Statement): Statement = s map (toExp) match {
    case s: DefInstance => WDefInstance(s.info, s.name, s.module, UnknownType)
    case s => s map (toStmt)
  }

  def run (c:Circuit): Circuit =
    c copy (modules = (c.modules map (_ map toStmt)))
}

object PullMuxes extends Pass {
   def name = "Pull Muxes"
   def run(c: Circuit): Circuit = {
     def pull_muxes_e(e: Expression): Expression = {
       val ex = e map (pull_muxes_e) match {
         case (e: WSubField) => e.exp match {
           case (ex: Mux) => Mux(ex.cond,
              WSubField(ex.tval, e.name, e.tpe, e.gender),
              WSubField(ex.fval, e.name, e.tpe, e.gender), e.tpe)
           case (ex: ValidIf) => ValidIf(ex.cond,
              WSubField(ex.value, e.name, e.tpe, e.gender), e.tpe)
           case (ex) => e
         }
         case (e: WSubIndex) => e.exp match {
           case (ex: Mux) => Mux(ex.cond,
              WSubIndex(ex.tval, e.value, e.tpe, e.gender),
              WSubIndex(ex.fval, e.value, e.tpe, e.gender), e.tpe)
           case (ex: ValidIf) => ValidIf(ex.cond,
              WSubIndex(ex.value, e.value, e.tpe, e.gender), e.tpe)
           case (ex) => e
         }
         case (e: WSubAccess) => e.exp match {
           case (ex: Mux) => Mux(ex.cond,
              WSubAccess(ex.tval, e.index, e.tpe, e.gender),
              WSubAccess(ex.fval, e.index, e.tpe, e.gender), e.tpe)
           case (ex: ValidIf) => ValidIf(ex.cond,
              WSubAccess(ex.value, e.index, e.tpe, e.gender), e.tpe)
           case (ex) => e
         }
         case (e) => e
       }
       ex map (pull_muxes_e)
     }
     def pull_muxes(s: Statement): Statement = s map (pull_muxes) map (pull_muxes_e)
     val modulesx = c.modules.map {
       case (m:Module) => Module(m.info, m.name, m.ports, pull_muxes(m.body))
       case (m:ExtModule) => m
     }
     Circuit(c.info, modulesx, c.main)
   }
}

object ExpandConnects extends Pass {
  def name = "Expand Connects"
  def run(c: Circuit): Circuit = {
    def expand_connects(m: Module): Module = {
      val genders = collection.mutable.LinkedHashMap[String,Gender]()
      def expand_s(s: Statement): Statement = {
        def set_gender(e: Expression): Expression = e map (set_gender) match {
          case (e: WRef) => WRef(e.name, e.tpe, e.kind, genders(e.name))
          case (e: WSubField) =>
            val f = get_field(e.exp.tpe, e.name)
            val genderx = times(gender(e.exp), f.flip)
            WSubField(e.exp, e.name, e.tpe, genderx)
          case (e: WSubIndex) => WSubIndex(e.exp, e.value, e.tpe, gender(e.exp))
          case (e: WSubAccess) => WSubAccess(e.exp, e.index, e.tpe, gender(e.exp))
          case (e) => e
        }
        s match {
          case (s: DefWire) => genders(s.name) = BIGENDER; s
          case (s: DefRegister) => genders(s.name) = BIGENDER; s
          case (s: WDefInstance) => genders(s.name) = MALE; s
          case (s: DefMemory) => genders(s.name) = MALE; s
          case (s: DefNode) => genders(s.name) = MALE; s
          case (s: IsInvalid) =>
            val invalids = (create_exps(s.expr) foldLeft Seq[Statement]())(
               (invalids,  expx) => gender(set_gender(expx)) match {
                  case BIGENDER => invalids :+ IsInvalid(s.info, expx)
                  case FEMALE => invalids :+ IsInvalid(s.info, expx)
                  case _ => invalids
               }
            )
            invalids.size match {
               case 0 => EmptyStmt
               case 1 => invalids.head
               case _ => Block(invalids)
            }
          case (s: Connect) =>
            val locs = create_exps(s.loc)
            val exps = create_exps(s.expr)
            Block((locs zip exps).zipWithIndex map {case ((locx, expx), i) =>
               get_flip(s.loc.tpe, i, Default) match {
                  case Default => Connect(s.info, locx, expx)
                  case Flip => Connect(s.info, expx, locx)
               }
            })
          case (s: PartialConnect) =>
            val ls = get_valid_points(s.loc.tpe, s.expr.tpe, Default, Default)
            val locs = create_exps(s.loc)
            val exps = create_exps(s.expr)
            Block(ls map {case (x, y) =>
              get_flip(s.loc.tpe, x, Default) match {
                 case Default => Connect(s.info, locs(x), exps(y))
                 case Flip => Connect(s.info, exps(y), locs(x))
              }
            })
          case (s) => s map (expand_s)
        }
      }

      m.ports.foreach { p => genders(p.name) = to_gender(p.direction) }
      Module(m.info, m.name, m.ports, expand_s(m.body))
    }

    val modulesx = c.modules.map {
       case (m: ExtModule) => m
       case (m: Module) => expand_connects(m)
    }
    Circuit(c.info, modulesx, c.main)
  }
}


// Replace shr by amount >= arg width with 0 for UInts and MSB for SInts
// TODO replace UInt with zero-width wire instead
object Legalize extends Pass {
  def name = "Legalize"
  private def legalizeShiftRight(e: DoPrim): Expression = {
    require(e.op == Shr)
    val amount = e.consts.head.toInt
    val width = bitWidth(e.args.head.tpe)
    lazy val msb = width - 1
    if (amount >= width) {
      e.tpe match {
        case UIntType(_) => zero
        case SIntType(_) =>
          val bits = DoPrim(Bits, e.args, Seq(msb, msb), BoolType)
          DoPrim(AsSInt, Seq(bits), Seq.empty, SIntType(IntWidth(1)))
        case t => error(s"Unsupported type ${t} for Primop Shift Right")
      }
    } else {
      e
    }
  }
  private def legalizeBits(expr: DoPrim): Expression = {
    lazy val (hi, low) = (expr.consts.head, expr.consts(1))
    lazy val mask = (BigInt(1) << (hi - low + 1).toInt) - 1
    lazy val width = IntWidth(hi - low + 1)
    expr.args.head match {
      case UIntLiteral(value, _) => UIntLiteral((value >> low.toInt) & mask, width)
      case SIntLiteral(value, _) => SIntLiteral((value >> low.toInt) & mask, width)
      case _ => expr
    }
  }
  private def legalizePad(expr: DoPrim): Expression = expr.args.head match {
    case UIntLiteral(value, IntWidth(width)) if (width < expr.consts.head) =>
      UIntLiteral(value, IntWidth(expr.consts.head))
    case SIntLiteral(value, IntWidth(width)) if (width < expr.consts.head) =>
      SIntLiteral(value, IntWidth(expr.consts.head))
    case _ => expr
  }
  private def legalizeConnect(c: Connect): Statement = {
    val t = c.loc.tpe
    val w = bitWidth(t)
    if (w >= bitWidth(c.expr.tpe)) {
      c
    } else {
      val bits = DoPrim(Bits, Seq(c.expr), Seq(w - 1, 0), UIntType(IntWidth(w)))
      val expr = t match {
        case UIntType(_) => bits
        case SIntType(_) => DoPrim(AsSInt, Seq(bits), Seq(), SIntType(IntWidth(w)))
      }
      Connect(c.info, c.loc, expr)
    }
  }
  def run (c: Circuit): Circuit = {
    def legalizeE(expr: Expression): Expression = expr map legalizeE match {
      case prim: DoPrim => prim.op match {
        case Shr => legalizeShiftRight(prim)
        case Pad => legalizePad(prim)
        case Bits => legalizeBits(prim)
        case _ => prim
      }
      case e => e // respect pre-order traversal
    }
    def legalizeS (s: Statement): Statement = {
      val legalizedStmt = s match {
        case c: Connect => legalizeConnect(c)
        case _ => s
      }
      legalizedStmt map legalizeS map legalizeE
    }
    c copy (modules = (c.modules map (_ map legalizeS)))
  }
}

object VerilogWrap extends Pass {
  def name = "Verilog Wrap"
  def vWrapE(e: Expression): Expression = e map vWrapE match {
    case e: DoPrim => e.op match {
      case Tail => e.args.head match {
        case e0: DoPrim => e0.op match {
          case Add => DoPrim(Addw, e0.args, Nil, e.tpe)
          case Sub => DoPrim(Subw, e0.args, Nil, e.tpe)
          case _ => e
        }
        case _ => e
      }
      case _ => e
    }
    case _ => e
  }
  def vWrapS(s: Statement): Statement = {
    s map vWrapS map vWrapE match {
      case s: Print => s copy (string = VerilogStringLitHandler.format(s.string))
      case s => s
    }
  }

  def run(c: Circuit): Circuit =
    c copy (modules = (c.modules map (_ map vWrapS)))
}

object VerilogRename extends Pass {
  def name = "Verilog Rename"
  def verilogRenameN(n: String): String =
    if (v_keywords(n)) "%s$".format(n) else n

  def verilogRenameE(e: Expression): Expression = e match {
    case e: WRef => e copy (name = verilogRenameN(e.name))
    case e => e map verilogRenameE
  }

  def verilogRenameS(s: Statement): Statement =
    s map verilogRenameS map verilogRenameE map verilogRenameN

  def verilogRenameP(p: Port): Port =
    p copy (name = verilogRenameN(p.name))

  def run(c: Circuit): Circuit =
    c copy (modules = (c.modules map (_ map verilogRenameP map verilogRenameS)))
}
