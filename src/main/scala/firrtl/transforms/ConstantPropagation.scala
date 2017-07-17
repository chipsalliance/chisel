// See LICENSE for license details.

package firrtl
package transforms

import firrtl._
import firrtl.annotations._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.WrappedExpression.weq

import annotation.tailrec
import collection.mutable

class ConstantPropagation extends Transform {
  def inputForm = LowForm
  def outputForm = LowForm

  private def pad(e: Expression, t: Type) = (bitWidth(e.tpe), bitWidth(t)) match {
    case (we, wt) if we < wt => DoPrim(Pad, Seq(e), Seq(wt), t)
    case (we, wt) if we == wt => e
  }

  private def asUInt(e: Expression, t: Type) = DoPrim(AsUInt, Seq(e), Seq(), t)

  trait FoldLogicalOp {
    def fold(c1: Literal, c2: Literal): Expression
    def simplify(e: Expression, lhs: Literal, rhs: Expression): Expression

    def apply(e: DoPrim): Expression = (e.args.head, e.args(1)) match {
      case (lhs: Literal, rhs: Literal) => fold(lhs, rhs)
      case (lhs: Literal, rhs) => pad(simplify(e, lhs, rhs), e.tpe)
      case (lhs, rhs: Literal) => pad(simplify(e, rhs, lhs), e.tpe)
      case _ => e
    }
  }

  object FoldAND extends FoldLogicalOp {
    def fold(c1: Literal, c2: Literal) = UIntLiteral(c1.value & c2.value, c1.width max c2.width)
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, w) if v == BigInt(0) => UIntLiteral(0, w)
      case SIntLiteral(v, w) if v == BigInt(0) => UIntLiteral(0, w)
      case UIntLiteral(v, IntWidth(w)) if v == (BigInt(1) << bitWidth(rhs.tpe).toInt) - 1 => rhs
      case _ => e
    }
  }

  object FoldOR extends FoldLogicalOp {
    def fold(c1: Literal, c2: Literal) = UIntLiteral(c1.value | c2.value, c1.width max c2.width)
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, _) if v == BigInt(0) => rhs
      case SIntLiteral(v, _) if v == BigInt(0) => asUInt(rhs, e.tpe)
      case UIntLiteral(v, IntWidth(w)) if v == (BigInt(1) << bitWidth(rhs.tpe).toInt) - 1 => lhs
      case _ => e
    }
  }

  object FoldXOR extends FoldLogicalOp {
    def fold(c1: Literal, c2: Literal) = UIntLiteral(c1.value ^ c2.value, c1.width max c2.width)
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, _) if v == BigInt(0) => rhs
      case SIntLiteral(v, _) if v == BigInt(0) => asUInt(rhs, e.tpe)
      case _ => e
    }
  }

  object FoldEqual extends FoldLogicalOp {
    def fold(c1: Literal, c2: Literal) = UIntLiteral(if (c1.value == c2.value) 1 else 0, IntWidth(1))
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, IntWidth(w)) if v == BigInt(1) && w == BigInt(1) && bitWidth(rhs.tpe) == BigInt(1) => rhs
      case _ => e
    }
  }

  object FoldNotEqual extends FoldLogicalOp {
    def fold(c1: Literal, c2: Literal) = UIntLiteral(if (c1.value != c2.value) 1 else 0, IntWidth(1))
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, IntWidth(w)) if v == BigInt(0) && w == BigInt(1) && bitWidth(rhs.tpe) == BigInt(1) => rhs
      case _ => e
    }
  }

  private def foldConcat(e: DoPrim) = (e.args.head, e.args(1)) match {
    case (UIntLiteral(xv, IntWidth(xw)), UIntLiteral(yv, IntWidth(yw))) => UIntLiteral(xv << yw.toInt | yv, IntWidth(xw + yw))
    case _ => e
  }

  private def foldShiftLeft(e: DoPrim) = e.consts.head.toInt match {
    case 0 => e.args.head
    case x => e.args.head match {
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v << x, IntWidth(w + x))
      case SIntLiteral(v, IntWidth(w)) => SIntLiteral(v << x, IntWidth(w + x))
      case _ => e
    }
  }

  private def foldShiftRight(e: DoPrim) = e.consts.head.toInt match {
    case 0 => e.args.head
    case x => e.args.head match {
      // TODO when amount >= x.width, return a zero-width wire
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v >> x, IntWidth((w - x) max 1))
      // take sign bit if shift amount is larger than arg width
      case SIntLiteral(v, IntWidth(w)) => SIntLiteral(v >> x, IntWidth((w - x) max 1))
      case _ => e
    }
  }

  private def foldComparison(e: DoPrim) = {
    def foldIfZeroedArg(x: Expression): Expression = {
      def isUInt(e: Expression): Boolean = e.tpe match {
        case UIntType(_) => true
        case _ => false
      }
      def isZero(e: Expression) = e match {
          case UIntLiteral(value, _) => value == BigInt(0)
          case SIntLiteral(value, _) => value == BigInt(0)
          case _ => false
        }
      x match {
        case DoPrim(Lt,  Seq(a,b),_,_) if isUInt(a) && isZero(b) => zero
        case DoPrim(Leq, Seq(a,b),_,_) if isZero(a) && isUInt(b) => one
        case DoPrim(Gt,  Seq(a,b),_,_) if isZero(a) && isUInt(b) => zero
        case DoPrim(Geq, Seq(a,b),_,_) if isUInt(a) && isZero(b) => one
        case ex => ex
      }
    }

    def foldIfOutsideRange(x: Expression): Expression = {
      //Note, only abides by a partial ordering
      case class Range(min: BigInt, max: BigInt) {
        def === (that: Range) =
          Seq(this.min, this.max, that.min, that.max)
            .sliding(2,1)
            .map(x => x.head == x(1))
            .reduce(_ && _)
        def > (that: Range) = this.min > that.max
        def >= (that: Range) = this.min >= that.max
        def < (that: Range) = this.max < that.min
        def <= (that: Range) = this.max <= that.min
      }
      def range(e: Expression): Range = e match {
        case UIntLiteral(value, _) => Range(value, value)
        case SIntLiteral(value, _) => Range(value, value)
        case _ => e.tpe match {
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
        case ex: DoPrim =>
          def r0 = range(ex.args.head)
          def r1 = range(ex.args(1))
          ex.op match {
            // Always true
            case Lt  if r0 < r1 => one
            case Leq if r0 <= r1 => one
            case Gt  if r0 > r1 => one
            case Geq if r0 >= r1 => one
            // Always false
            case Lt  if r0 >= r1 => zero
            case Leq if r0 > r1 => zero
            case Gt  if r0 <= r1 => zero
            case Geq if r0 < r1 => zero
            case _ => ex
          }
        case ex => ex
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
    case Not => e.args.head match {
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v ^ ((BigInt(1) << w.toInt) - 1), IntWidth(w))
      case _ => e
    }
    case AsUInt => e.args.head match {
      case SIntLiteral(v, IntWidth(w)) => UIntLiteral(v + (if (v < 0) BigInt(1) << w.toInt else 0), IntWidth(w))
      case u: UIntLiteral => u
      case _ => e
    }
    case AsSInt => e.args.head match {
      case UIntLiteral(v, IntWidth(w)) => SIntLiteral(v - ((v >> (w.toInt-1)) << w.toInt), IntWidth(w))
      case s: SIntLiteral => s
      case _ => e
    }
    case Pad => e.args.head match {
      case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v, IntWidth(e.consts.head max w))
      case SIntLiteral(v, IntWidth(w)) => SIntLiteral(v, IntWidth(e.consts.head max w))
      case _ if bitWidth(e.args.head.tpe) == e.consts.head => e.args.head
      case _ => e
    }
    case Bits => e.args.head match {
      case lit: Literal =>
        val hi = e.consts.head.toInt
        val lo = e.consts(1).toInt
        require(hi >= lo)
        UIntLiteral((lit.value >> lo) & ((BigInt(1) << (hi - lo + 1)) - 1), getWidth(e.tpe))
      case x if bitWidth(e.tpe) == bitWidth(x.tpe) => x.tpe match {
        case t: UIntType => x
        case _ => asUInt(x, e.tpe)
      }
      case _ => e
    }
    case _ => e
  }

  private def constPropMuxCond(m: Mux) = m.cond match {
    case UIntLiteral(c, _) => pad(if (c == BigInt(1)) m.tval else m.fval, m.tpe)
    case _ => m
  }

  private def constPropMux(m: Mux): Expression = (m.tval, m.fval) match {
    case _ if m.tval == m.fval => m.tval
    case (t: UIntLiteral, f: UIntLiteral) =>
      if (t.value == BigInt(1) && f.value == BigInt(0) && bitWidth(m.tpe) == BigInt(1)) m.cond
      else constPropMuxCond(m)
    case _ => constPropMuxCond(m)
  }

  private def constPropNodeRef(r: WRef, e: Expression) = e match {
    case _: UIntLiteral | _: SIntLiteral | _: WRef => e
    case _ => r
  }

  // Is "a" a "better name" than "b"?
  private def betterName(a: String, b: String): Boolean = (a.head != '_') && (b.head == '_')

  // Two pass process
  // 1. Propagate constants in expressions and forward propagate references
  // 2. Propagate references again for backwards reference (Wires)
  // TODO Replacing all wires with nodes makes the second pass unnecessary
  //   However, preserving decent names DOES require a second pass
  //   Replacing all wires with nodes makes it unnecessary for preserving decent names to trigger an
  //   extra iteration though
  @tailrec
  private def constPropModule(m: Module, dontTouches: Set[String]): Module = {
    var nPropagated = 0L
    val nodeMap = mutable.HashMap.empty[String, Expression]
    // For cases where we are trying to constprop a bad name over a good one, we swap their names
    // during the second pass
    val swapMap = mutable.HashMap.empty[String, String]

    def backPropExpr(expr: Expression): Expression = {
      val old = expr map backPropExpr
      val propagated = old match {
        // When swapping, we swap both rhs and lhs
        case ref @ WRef(rname, _,_,_) if swapMap.contains(rname) =>
          ref.copy(name = swapMap(rname))
        // Only const prop on the rhs
        case ref @ WRef(rname, _,_, MALE) if nodeMap.contains(rname) =>
          constPropNodeRef(ref, nodeMap(rname))
        case x => x
      }
      if (old ne propagated) {
        nPropagated += 1
      }
      propagated
    }

    def backPropStmt(stmt: Statement): Statement = stmt map backPropExpr match {
      case decl: IsDeclaration if swapMap.contains(decl.name) =>
        val newName = swapMap(decl.name)
        nPropagated += 1
        decl match {
          case node: DefNode => node.copy(name = newName)
          case wire: DefWire => wire.copy(name = newName)
          case reg: DefRegister => reg.copy(name = newName)
          case other => throwInternalError
        }
      case other => other map backPropStmt
    }

    def constPropExpression(e: Expression): Expression = {
      val old = e map constPropExpression
      val propagated = old match {
        case p: DoPrim => constPropPrim(p)
        case m: Mux => constPropMux(m)
        case ref @ WRef(rname, _,_, MALE) if nodeMap.contains(rname) =>
          constPropNodeRef(ref, nodeMap(rname))
        case x => x
      }
      propagated
    }

    // When propagating a reference, check if we want to keep the name that would be deleted
    def propagateRef(lname: String, value: Expression): Unit = {
      value match {
        case WRef(rname,_,_,_) if betterName(lname, rname) && !swapMap.contains(rname) =>
          assert(!swapMap.contains(lname)) // <- Shouldn't be possible because lname is either a
          // node declaration or the single connection to a wire or register
          swapMap += (lname -> rname, rname -> lname)
        case _ =>
      }
      nodeMap(lname) = value
    }

    def constPropStmt(s: Statement): Statement = {
      val stmtx = s map constPropStmt map constPropExpression
      stmtx match {
        case x: DefNode if !dontTouches.contains(x.name) => propagateRef(x.name, x.value)
        case Connect(_, WRef(wname, wtpe, WireKind, _), expr: Literal) if !dontTouches.contains(wname) =>
          val exprx = constPropExpression(pad(expr, wtpe))
          propagateRef(wname, exprx)
        // Const prop registers that are fed only a constant or a mux between and constant and the
        // register itself
        // This requires that reset has been made explicit
        case Connect(_, rref @ WRef(rname, rtpe, RegKind, _), expr) => expr match {
          case lit: Literal =>
            nodeMap(rname) = constPropExpression(pad(lit, rtpe))
          case Mux(_, tval: WRef, fval: Literal, _) if weq(rref, tval) =>
            nodeMap(rname) = constPropExpression(pad(fval, rtpe))
          case Mux(_, tval: Literal, fval: WRef, _) if weq(rref, fval) =>
            nodeMap(rname) = constPropExpression(pad(tval, rtpe))
          case _ =>
        }
        case _ =>
      }
      stmtx
    }

    val res = m.copy(body = backPropStmt(constPropStmt(m.body)))
    if (nPropagated > 0) constPropModule(res, dontTouches) else res
  }

  private def run(c: Circuit, dontTouchMap: Map[String, Set[String]]): Circuit = {
    val modulesx = c.modules.map {
      case m: ExtModule => m
      case m: Module => constPropModule(m, dontTouchMap.getOrElse(m.name, Set.empty))
    }
    Circuit(c.info, modulesx, c.main)
  }

  def execute(state: CircuitState): CircuitState = {
    val dontTouches: Seq[(String, String)] = state.annotations match {
      case Some(aMap) => aMap.annotations.collect {
        case DontTouchAnnotation(ComponentName(c, ModuleName(m, _))) => m -> c
      }
      case None => Seq.empty
    }
    // Map from module name to component names
    val dontTouchMap: Map[String, Set[String]] =
      dontTouches.groupBy(_._1).mapValues(_.map(_._2).toSet)

    state.copy(circuit = run(state.circuit, dontTouchMap))
  }
}
