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
import firrtl.PrimOps._

import annotation.tailrec

object ConstProp extends Pass {
  def name = "Constant Propagation"

  trait FoldLogicalOp {
    def fold(c1: UIntLiteral, c2: UIntLiteral): UIntLiteral
    def simplify(e: Expression, lhs: UIntLiteral, rhs: Expression): Expression

    def apply(e: DoPrim): Expression = (e.args(0), e.args(1)) match {
      case (lhs: UIntLiteral, rhs: UIntLiteral) => fold(lhs, rhs)
      case (lhs: UIntLiteral, rhs) => simplify(e, lhs, rhs)
      case (lhs, rhs: UIntLiteral) => simplify(e, rhs, lhs)
      case _ => e
    }
  }

  object FoldAND extends FoldLogicalOp {
    def fold(c1: UIntLiteral, c2: UIntLiteral) = UIntLiteral(c1.value & c2.value, c1.width max c2.width)
    def simplify(e: Expression, lhs: UIntLiteral, rhs: Expression) = lhs.width match {
      case IntWidth(w) if long_BANG(tpe(rhs)) == w =>
        if (lhs.value == 0) lhs // and(x, 0) => 0
        else if (lhs.value == (BigInt(1) << w.toInt) - 1) rhs // and(x, 1) => x
        else e
      case _ => e
    }
  }

  object FoldOR extends FoldLogicalOp {
    def fold(c1: UIntLiteral, c2: UIntLiteral) = UIntLiteral(c1.value | c2.value, c1.width max c2.width)
    def simplify(e: Expression, lhs: UIntLiteral, rhs: Expression) = lhs.width match {
      case IntWidth(w) if long_BANG(tpe(rhs)) == w =>
        if (lhs.value == 0) rhs // or(x, 0) => x
        else if (lhs.value == (BigInt(1) << w.toInt) - 1) lhs // or(x, 1) => 1
        else e
      case _ => e
    }
  }

  object FoldXOR extends FoldLogicalOp {
    def fold(c1: UIntLiteral, c2: UIntLiteral) = UIntLiteral(c1.value ^ c2.value, c1.width max c2.width)
    def simplify(e: Expression, lhs: UIntLiteral, rhs: Expression) = lhs.width match {
      case IntWidth(w) if long_BANG(tpe(rhs)) == w =>
        if (lhs.value == 0) rhs // xor(x, 0) => x
        else e
      case _ => e
    }
  }

  object FoldEqual extends FoldLogicalOp {
    def fold(c1: UIntLiteral, c2: UIntLiteral) = UIntLiteral(if (c1.value == c2.value) 1 else 0, IntWidth(1))
    def simplify(e: Expression, lhs: UIntLiteral, rhs: Expression) = lhs.width match {
      case IntWidth(w) if w == 1 && long_BANG(tpe(rhs)) == 1 =>
        if (lhs.value == 1) rhs // eq(x, 1) => x
        else e
      case _ => e
    }
  }

  object FoldNotEqual extends FoldLogicalOp {
    def fold(c1: UIntLiteral, c2: UIntLiteral) = UIntLiteral(if (c1.value != c2.value) 1 else 0, IntWidth(1))
    def simplify(e: Expression, lhs: UIntLiteral, rhs: Expression) = lhs.width match {
      case IntWidth(w) if w == 1 && long_BANG(tpe(rhs)) == w =>
        if (lhs.value == 0) rhs // neq(x, 0) => x
        else e
      case _ => e
    }
  }

  private def foldConcat(e: DoPrim) = (e.args(0), e.args(1)) match {
    case (UIntLiteral(xv, IntWidth(xw)), UIntLiteral(yv, IntWidth(yw))) => UIntLiteral(xv << yw.toInt | yv, IntWidth(xw + yw))
    case _ => e
  }

  private def foldShiftLeft(e: DoPrim) = e.consts(0).toInt match {
    case 0 => e.args(0)
    case x => e.args(0) match {
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v << x, IntWidth(w + x))
      case SIntLiteral(v, IntWidth(w)) => SIntLiteral(v << x, IntWidth(w + x))
      case _ => e
    }
  }

  private def foldShiftRight(e: DoPrim) = e.consts(0).toInt match {
    case 0 => e.args(0)
    case x => e.args(0) match {
      // TODO when amount >= x.width, return a zero-width wire
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v >> x, IntWidth((w - x) max 1))
      // take sign bit if shift amount is larger than arg width
      case SIntLiteral(v, IntWidth(w)) => SIntLiteral(v >> x, IntWidth((w - x) max 1))
      case _ => e
    }
  }

  private def foldComparison(e: DoPrim) = {
    def foldIfZeroedArg(x: Expression): Expression = {
      def isUInt(e: Expression): Boolean = tpe(e) match {
        case UIntType(_) => true
        case _ => false
      }
      def isZero(e: Expression) = e match {
          case UIntLiteral(value, _) => value == BigInt(0)
          case SIntLiteral(value, _) => value == BigInt(0)
          case _ => false
        }
      x match {
        case DoPrim(Lt,  Seq(a,b),_,_) if(isUInt(a) && isZero(b)) => zero
        case DoPrim(Leq, Seq(a,b),_,_) if(isZero(a) && isUInt(b)) => one
        case DoPrim(Gt,  Seq(a,b),_,_) if(isZero(a) && isUInt(b)) => zero
        case DoPrim(Geq, Seq(a,b),_,_) if(isUInt(a) && isZero(b)) => one
        case e => e
      }
    }

    def foldIfOutsideRange(x: Expression): Expression = {
      //Note, only abides by a partial ordering
      case class Range(min: BigInt, max: BigInt) {
        def === (that: Range) =
          Seq(this.min, this.max, that.min, that.max)
            .sliding(2,1)
            .map(x => x(0) == x(1))
            .reduce(_ && _)
        def > (that: Range) = this.min > that.max
        def >= (that: Range) = this.min >= that.max
        def < (that: Range) = this.max < that.min
        def <= (that: Range) = this.max <= that.min
      }
      def range(e: Expression): Range = e match {
        case UIntLiteral(value, _) => Range(value, value)
        case SIntLiteral(value, _) => Range(value, value)
        case _ => tpe(e) match {
          case SIntType(IntWidth(width)) => Range(
            min = BigInt(0) - BigInt(2).pow(width.toInt - 1),
            max = BigInt(2).pow(width.toInt - 1) - BigInt(1)
          )
          case UIntType(IntWidth(width)) => Range(
            min = BigInt(0),
            max = BigInt(2).pow(width.toInt) - BigInt(1)
          )
        }
      }
      // Calculates an expression's range of values
      x match {
        case e: DoPrim => {
          def r0 = range(e.args(0))
          def r1 = range(e.args(1))
          e.op match {
            // Always true
            case Lt  if (r0 < r1) => one
            case Leq if (r0 <= r1) => one
            case Gt  if (r0 > r1) => one
            case Geq if (r0 >= r1) => one
            // Always false
            case Lt  if (r0 >= r1) => zero
            case Leq if (r0 > r1) => zero
            case Gt  if (r0 <= r1) => zero
            case Geq if (r0 < r1) => zero
            case _ => e
          }
        }
        case e => e
      }
    }
    foldIfZeroedArg(foldIfOutsideRange(e))
  }

  private def constPropPrim(e: DoPrim): Expression = e.op match {
    case Shl => foldShiftLeft(e)
    case Shr => foldShiftRight(e)
    case Cat => foldConcat(e)
    case And => FoldAND(e)
    case Or => FoldOR(e)
    case Xor => FoldXOR(e)
    case Eq => FoldEqual(e)
    case Neq => FoldNotEqual(e)
    case (Lt | Leq | Gt | Geq) => foldComparison(e)
    case Not => e.args(0) match {
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v ^ ((BigInt(1) << w.toInt) - 1), IntWidth(w))
      case _ => e
    }
    case Bits => e.args(0) match {
      case UIntLiteral(v, _) => {
        val hi = e.consts(0).toInt
        val lo = e.consts(1).toInt
        require(hi >= lo)
        UIntLiteral((v >> lo) & ((BigInt(1) << (hi - lo + 1)) - 1), widthBANG(tpe(e)))
      }
      case x if long_BANG(tpe(e)) == long_BANG(tpe(x)) => tpe(x) match {
        case t: UIntType => x
        case _ => DoPrim(AsUInt, Seq(x), Seq(), tpe(e))
      }
      case _ => e
    }
    case _ => e
  }

  private def constPropMuxCond(m: Mux) = {
    // Only propagate a value if its width matches the mux width
    def propagate(e: Expression, muxWidth: BigInt) = e match {
      case UIntLiteral(v, _) => UIntLiteral(v, IntWidth(muxWidth))
      case _ => tpe(e) match {
        case UIntType(IntWidth(w)) if muxWidth == w => e
        case _ => m
      }
    }
    (m.cond, m.tpe) match {
      case (UIntLiteral(c, _), UIntType(IntWidth(w))) => propagate(if (c == 1) m.tval else m.fval, w)
      case _ => m
    }
  }

  private def constPropMux(m: Mux): Expression = (m.tval, m.fval) match {
    case _ if m.tval == m.fval => m.tval
    case (t: UIntLiteral, f: UIntLiteral) =>
      if (t.value == 1 && f.value == 0 && long_BANG(m.tpe) == 1) m.cond
      else constPropMuxCond(m)
    case _ => constPropMuxCond(m)
  }

  private def constPropNodeRef(r: WRef, e: Expression) = e match {
    case _: UIntLiteral | _: SIntLiteral | _: WRef => e
    case _ => r
  }

  @tailrec
  private def constPropModule(m: Module): Module = {
    var nPropagated = 0L
    val nodeMap = collection.mutable.HashMap[String, Expression]()

    def constPropExpression(e: Expression): Expression = {
      val old = e map constPropExpression
      val propagated = old match {
        case p: DoPrim => constPropPrim(p)
        case m: Mux => constPropMux(m)
        case r: WRef if nodeMap contains r.name => constPropNodeRef(r, nodeMap(r.name))
        case x => x
      }
      if (old ne propagated)
        nPropagated += 1
      propagated
    }

    def constPropStmt(s: Statement): Statement = {
      s match {
        case x: DefNode => nodeMap(x.name) = x.value
        case _ =>
      }
      s map constPropStmt map constPropExpression
    }

    val res = Module(m.info, m.name, m.ports, constPropStmt(m.body))
    if (nPropagated > 0) constPropModule(res) else res
  }

  def run(c: Circuit): Circuit = {
    val modulesx = c.modules.map {
      case m: ExtModule => m
      case m: Module => constPropModule(m)
    }
    Circuit(c.info, modulesx, c.main)
  }
}
