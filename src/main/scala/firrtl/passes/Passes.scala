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
  def trigger() = errors.size match {
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

  def toExp(e: Expression): Expression = e map toExp match {
    case ex: Reference => WRef(ex.name, ex.tpe, NodeKind, UNKNOWNGENDER)
    case ex: SubField => WSubField(ex.expr, ex.name, ex.tpe, UNKNOWNGENDER)
    case ex: SubIndex => WSubIndex(ex.expr, ex.value, ex.tpe, UNKNOWNGENDER)
    case ex: SubAccess => WSubAccess(ex.expr, ex.index, ex.tpe, UNKNOWNGENDER)
    case ex => ex // This might look like a case to use case _ => e, DO NOT!
  }

  def toStmt(s: Statement): Statement = s map toExp match {
    case sx: DefInstance => WDefInstance(sx.info, sx.name, sx.module, UnknownType)
    case sx => sx map toStmt
  }

  def run (c:Circuit): Circuit =
    c copy (modules = c.modules map (_ map toStmt))
}

object PullMuxes extends Pass {
   def name = "Pull Muxes"
   def run(c: Circuit): Circuit = {
     def pull_muxes_e(e: Expression): Expression = {
       val exxx = e map pull_muxes_e match {
         case ex: WSubField => ex.exp match {
           case exx: Mux => Mux(exx.cond,
              WSubField(exx.tval, ex.name, ex.tpe, ex.gender),
              WSubField(exx.fval, ex.name, ex.tpe, ex.gender), ex.tpe)
           case exx: ValidIf => ValidIf(exx.cond,
              WSubField(exx.value, ex.name, ex.tpe, ex.gender), ex.tpe)
           case _ => ex  // case exx => exx causes failed tests
         }
         case ex: WSubIndex => ex.exp match {
           case exx: Mux => Mux(exx.cond,
              WSubIndex(exx.tval, ex.value, ex.tpe, ex.gender),
              WSubIndex(exx.fval, ex.value, ex.tpe, ex.gender), ex.tpe)
           case exx: ValidIf => ValidIf(exx.cond,
              WSubIndex(exx.value, ex.value, ex.tpe, ex.gender), ex.tpe)
           case _ => ex  // case exx => exx causes failed tests
         }
         case ex: WSubAccess => ex.exp match {
           case exx: Mux => Mux(exx.cond,
              WSubAccess(exx.tval, ex.index, ex.tpe, ex.gender),
              WSubAccess(exx.fval, ex.index, ex.tpe, ex.gender), ex.tpe)
           case exx: ValidIf => ValidIf(exx.cond,
              WSubAccess(exx.value, ex.index, ex.tpe, ex.gender), ex.tpe)
           case _ => ex  // case exx => exx causes failed tests
         }
         case ex => ex
       }
       exxx map pull_muxes_e
     }
     def pull_muxes(s: Statement): Statement = s map pull_muxes map pull_muxes_e
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
        def set_gender(e: Expression): Expression = e map set_gender match {
          case ex: WRef => WRef(ex.name, ex.tpe, ex.kind, genders(ex.name))
          case ex: WSubField =>
            val f = get_field(ex.exp.tpe, ex.name)
            val genderx = times(gender(ex.exp), f.flip)
            WSubField(ex.exp, ex.name, ex.tpe, genderx)
          case ex: WSubIndex => WSubIndex(ex.exp, ex.value, ex.tpe, gender(ex.exp))
          case ex: WSubAccess => WSubAccess(ex.exp, ex.index, ex.tpe, gender(ex.exp))
          case ex => ex
        }
        s match {
          case sx: DefWire => genders(sx.name) = BIGENDER; sx
          case sx: DefRegister => genders(sx.name) = BIGENDER; sx
          case sx: WDefInstance => genders(sx.name) = MALE; sx
          case sx: DefMemory => genders(sx.name) = MALE; sx
          case sx: DefNode => genders(sx.name) = MALE; sx
          case sx: IsInvalid =>
            val invalids = (create_exps(sx.expr) foldLeft Seq[Statement]())(
               (invalids,  expx) => gender(set_gender(expx)) match {
                  case BIGENDER => invalids :+ IsInvalid(sx.info, expx)
                  case FEMALE => invalids :+ IsInvalid(sx.info, expx)
                  case _ => invalids
               }
            )
            invalids.size match {
               case 0 => EmptyStmt
               case 1 => invalids.head
               case _ => Block(invalids)
            }
          case sx: Connect =>
            val locs = create_exps(sx.loc)
            val exps = create_exps(sx.expr)
            Block((locs zip exps).zipWithIndex map {case ((locx, expx), i) =>
               get_flip(sx.loc.tpe, i, Default) match {
                  case Default => Connect(sx.info, locx, expx)
                  case Flip => Connect(sx.info, expx, locx)
               }
            })
          case sx: PartialConnect =>
            val ls = get_valid_points(sx.loc.tpe, sx.expr.tpe, Default, Default)
            val locs = create_exps(sx.loc)
            val exps = create_exps(sx.expr)
            Block(ls map {case (x, y) =>
              get_flip(sx.loc.tpe, x, Default) match {
                 case Default => Connect(sx.info, locs(x), exps(y))
                 case Flip => Connect(sx.info, exps(y), locs(x))
              }
            })
          case sx => sx map expand_s
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
        case t => error(s"Unsupported type $t for Primop Shift Right")
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
    case UIntLiteral(value, IntWidth(width)) if width < expr.consts.head =>
      UIntLiteral(value, IntWidth(expr.consts.head))
    case SIntLiteral(value, IntWidth(width)) if width < expr.consts.head =>
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
    c copy (modules = c.modules map (_ map legalizeS))
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
      case sx: Print => sx copy (string = VerilogStringLitHandler.format(sx.string))
      case sx => sx
    }
  }

  def run(c: Circuit): Circuit =
    c copy (modules = c.modules map (_ map vWrapS))
}

object VerilogRename extends Pass {
  def name = "Verilog Rename"
  def verilogRenameN(n: String): String =
    if (v_keywords(n)) "%s$".format(n) else n

  def verilogRenameE(e: Expression): Expression = e match {
    case ex: WRef => ex copy (name = verilogRenameN(ex.name))
    case ex => ex map verilogRenameE
  }

  def verilogRenameS(s: Statement): Statement =
    s map verilogRenameS map verilogRenameE map verilogRenameN

  def verilogRenameP(p: Port): Port =
    p copy (name = verilogRenameN(p.name))

  def run(c: Circuit): Circuit =
    c copy (modules = c.modules map (_ map verilogRenameP map verilogRenameS))
}


object VerilogPrep extends Pass {
  def name = "Verilog Prep"
  type InstAttaches = collection.mutable.HashMap[String, Expression]
  def run(c: Circuit): Circuit = {
    def buildS(attaches: InstAttaches)(s: Statement): Statement = s match {
      case Attach(_, source, exps) => 
        exps foreach { e => attaches(e.serialize) = source }
        s
      case _ => s map buildS(attaches)
    }
    def lowerE(e: Expression): Expression = e match {
      case _: WRef|_: WSubField if kind(e) == InstanceKind =>
        WRef(LowerTypes.loweredName(e), e.tpe, kind(e), gender(e))
      case _ => e map lowerE
    }
    def lowerS(attaches: InstAttaches)(s: Statement): Statement = s match {
      case WDefInstance(info, name, module, tpe) =>
        val exps = create_exps(WRef(name, tpe, ExpKind, MALE))
        val wcon = WDefInstanceConnector(info, name, module, tpe, exps.map( e => e.tpe match {
          case AnalogType(w) => attaches(e.serialize)
          case _ => WRef(LowerTypes.loweredName(e), e.tpe, WireKind, MALE)
        }))
        val wires = exps.map ( e => e.tpe match {
          case AnalogType(w) => EmptyStmt
          case _ => DefWire(info, LowerTypes.loweredName(e), e.tpe)
        })
        Block(Seq(wcon) ++ wires)
      case Attach(info, source, exps) => EmptyStmt
      case _ => s map lowerS(attaches) map lowerE
    }
    def prepModule(m: DefModule): DefModule = {
      val attaches = new InstAttaches
      m map buildS(attaches)
      m map lowerS(attaches)
    }
    c.copy(modules = c.modules.map(prepModule))
  }
}
