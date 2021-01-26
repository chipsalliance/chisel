// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl._
import firrtl.annotations._
import firrtl.annotations.TargetToken._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.PrimOps._
import firrtl.graph.DiGraph
import firrtl.analyses.InstanceKeyGraph
import firrtl.annotations.TargetToken.Ref
import firrtl.options.Dependency
import firrtl.stage.DisableFold

import annotation.tailrec
import collection.mutable

object ConstantPropagation {
  private def litOfType(value: BigInt, t: Type): Literal = t match {
    case UIntType(w) => UIntLiteral(value, w)
    case SIntType(w) => SIntLiteral(value, w)
  }

  private def asUInt(e: Expression, t: Type) = DoPrim(AsUInt, Seq(e), Seq(), t)

  /** Pads e to the width of t */
  def pad(e: Expression, t: Type) = (bitWidth(e.tpe), bitWidth(t)) match {
    case (we, wt) if we < wt  => DoPrim(Pad, Seq(e), Seq(wt), t)
    case (we, wt) if we == wt => e
  }

  def constPropBitExtract(e: DoPrim) = {
    val arg = e.args.head
    val (hi, lo) = e.op match {
      case Bits => (e.consts.head.toInt, e.consts(1).toInt)
      case Tail => ((bitWidth(arg.tpe) - 1 - e.consts.head).toInt, 0)
      case Head => ((bitWidth(arg.tpe) - 1).toInt, (bitWidth(arg.tpe) - e.consts.head).toInt)
    }

    arg match {
      case lit: Literal =>
        require(hi >= lo)
        UIntLiteral((lit.value >> lo) & ((BigInt(1) << (hi - lo + 1)) - 1), getWidth(e.tpe))
      case x if bitWidth(e.tpe) == bitWidth(x.tpe) =>
        x.tpe match {
          case t: UIntType => x
          case _ => asUInt(x, e.tpe)
        }
      case _ => e
    }
  }

  def foldShiftRight(e: DoPrim) = e.consts.head.toInt match {
    case 0 => e.args.head
    case x =>
      e.args.head match {
        // TODO when amount >= x.width, return a zero-width wire
        case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v >> x, IntWidth((w - x).max(1)))
        // take sign bit if shift amount is larger than arg width
        case SIntLiteral(v, IntWidth(w)) => SIntLiteral(v >> x, IntWidth((w - x).max(1)))
        case _                           => e
      }
  }

  /** ********************************************
    * REGISTER CONSTANT PROPAGATION HELPER TYPES *
    * ********************************************
    */

  // A utility class that is somewhat like an Option but with two variants containing Nothing.
  // for register constant propagation (register or literal).
  private abstract class ConstPropBinding[+T] {
    def resolve[V >: T](that: ConstPropBinding[V]): ConstPropBinding[V] = (this, that) match {
      case (x, y) if (x == y)   => x
      case (x, UnboundConstant) => x
      case (UnboundConstant, y) => y
      case _                    => NonConstant
    }
  }

  // BoundConstant means that it has exactly the one allowable source of that type.
  private case class BoundConstant[T](value: T) extends ConstPropBinding[T]

  // NonConstant indicates multiple distinct sources.
  private case object NonConstant extends ConstPropBinding[Nothing]

  // UnboundConstant means that a node does not yet have a source of the two allowable types
  private case object UnboundConstant extends ConstPropBinding[Nothing]

  // A RegCPEntry tracks whether a given signal could be part of a register constant propagation
  // loop. It contains const prop bindings for a register name and a literal, which represent the
  // fact that a constant propagation loop can include both self-assignments and consistent literals.
  private case class RegCPEntry(r: ConstPropBinding[String], l: ConstPropBinding[Literal]) {
    def resolve(that: RegCPEntry) = RegCPEntry(r.resolve(that.r), l.resolve(that.l))
    def nonConstant: Boolean = r == NonConstant || l == NonConstant
  }

}

class ConstantPropagation extends Transform with DependencyAPIMigration {
  import ConstantPropagation._

  override def prerequisites =
    ((new mutable.LinkedHashSet())
      ++ firrtl.stage.Forms.LowForm
      - Dependency(firrtl.passes.Legalize)
      + Dependency(firrtl.passes.RemoveValidIf)).toSeq

  override def optionalPrerequisites = Seq.empty

  override def optionalPrerequisiteOf =
    Seq(
      Dependency(firrtl.passes.memlib.VerilogMemDelays),
      Dependency(firrtl.passes.SplitExpressions),
      Dependency[SystemVerilogEmitter],
      Dependency[VerilogEmitter]
    )

  override def invalidates(a: Transform): Boolean = a match {
    case firrtl.passes.Legalize => true
    case _                      => false
  }

  sealed trait SimplifyBinaryOp {
    def matchingArgsValue(e: DoPrim, arg: Expression): Expression
    def apply(e: DoPrim): Expression = {
      if (e.args.head == e.args(1)) matchingArgsValue(e, e.args.head) else e
    }
  }

  sealed trait FoldCommutativeOp extends SimplifyBinaryOp {
    def fold(c1:    Literal, c2:     Literal): Expression
    def simplify(e: Expression, lhs: Literal, rhs: Expression): Expression

    override def apply(e: DoPrim): Expression = (e.args.head, e.args(1)) match {
      case (lhs: Literal, rhs: Literal) => fold(lhs, rhs)
      case (lhs: Literal, rhs) => pad(simplify(e, lhs, rhs), e.tpe)
      case (lhs, rhs: Literal) => pad(simplify(e, rhs, lhs), e.tpe)
      case (lhs, rhs) if (lhs == rhs) => matchingArgsValue(e, lhs)
      case _                          => e
    }
  }

  /** Interface for describing a simplification of a reduction primitive op */
  sealed trait SimplifyReductionOp {

    /** The initial value used in the reduction */
    def identityValue: Boolean

    /** The reduction function of the primitive op expressed */
    def reduce: (Boolean, Boolean) => Boolean

    /** Utility to simplify a reduction op of a literal, parameterized by identityValue and reduce methods. This will
      * return the identityValue in the event of reducing a zero-width literal.
      */
    private def simplifyLiteral(a: Literal): Literal = {

      val w: BigInt = getWidth(a) match {
        case IntWidth(b) => b
      }

      val maskedValue = Utils.maskBigInt(a.value, w.toInt)
      val v: Seq[Boolean] = s"%${w}s".format(maskedValue.toString(2)).map(_ == '1')

      (BigInt(0) until w).zip(v).foldLeft(identityValue) {
        case (acc, (_, x)) => reduce(acc, x)
      } match {
        case false => zero
        case true  => one
      }
    }

    /** Reduce a reduction primitive op to a simpler expression if possible
      * @param prim the primitive op to reduce
      * @return a simplified expression or the original primitive op
      */
    def apply(prim: DoPrim): Expression = prim.args.head match {
      case a: Literal => simplifyLiteral(a)
      case _ => prim
    }

  }

  object FoldADD extends FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal) = ((c1, c2): @unchecked) match {
      case (_: UIntLiteral, _: UIntLiteral) => UIntLiteral(c1.value + c2.value, (c1.width.max(c2.width)) + IntWidth(1))
      case (_: SIntLiteral, _: SIntLiteral) => SIntLiteral(c1.value + c2.value, (c1.width.max(c2.width)) + IntWidth(1))
    }
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, w) if v == BigInt(0) => rhs
      case SIntLiteral(v, w) if v == BigInt(0) => rhs
      case _                                   => e
    }
    def matchingArgsValue(e: DoPrim, arg: Expression) = e
  }

  object SimplifySUB extends SimplifyBinaryOp {
    def matchingArgsValue(e: DoPrim, arg: Expression) = litOfType(0, e.tpe)
  }

  object SimplifyDIV extends SimplifyBinaryOp {
    def matchingArgsValue(e: DoPrim, arg: Expression) = litOfType(1, e.tpe)
  }

  object SimplifyREM extends SimplifyBinaryOp {
    def matchingArgsValue(e: DoPrim, arg: Expression) = litOfType(0, e.tpe)
  }

  object FoldAND extends FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal) = {
      val width = (c1.width.max(c2.width)).asInstanceOf[IntWidth]
      UIntLiteral.masked(c1.value & c2.value, width)
    }
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, w) if v == BigInt(0)                                            => UIntLiteral(0, w)
      case SIntLiteral(v, w) if v == BigInt(0)                                            => UIntLiteral(0, w)
      case UIntLiteral(v, IntWidth(w)) if v == (BigInt(1) << bitWidth(rhs.tpe).toInt) - 1 => rhs
      case _                                                                              => e
    }
    def matchingArgsValue(e: DoPrim, arg: Expression) = asUInt(arg, e.tpe)
  }

  object FoldOR extends FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal) = {
      val width = (c1.width.max(c2.width)).asInstanceOf[IntWidth]
      UIntLiteral.masked((c1.value | c2.value), width)
    }
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, _) if v == BigInt(0)                                            => rhs
      case SIntLiteral(v, _) if v == BigInt(0)                                            => asUInt(rhs, e.tpe)
      case UIntLiteral(v, IntWidth(w)) if v == (BigInt(1) << bitWidth(rhs.tpe).toInt) - 1 => lhs
      case _                                                                              => e
    }
    def matchingArgsValue(e: DoPrim, arg: Expression) = asUInt(arg, e.tpe)
  }

  object FoldXOR extends FoldCommutativeOp {
    def fold(c1: Literal, c2: Literal) = {
      val width = (c1.width.max(c2.width)).asInstanceOf[IntWidth]
      UIntLiteral.masked((c1.value ^ c2.value), width)
    }
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, _) if v == BigInt(0) => rhs
      case SIntLiteral(v, _) if v == BigInt(0) => asUInt(rhs, e.tpe)
      case _                                   => e
    }
    def matchingArgsValue(e: DoPrim, arg: Expression) = UIntLiteral(0, getWidth(arg.tpe))
  }

  object FoldEqual extends FoldCommutativeOp {
    def fold(c1:    Literal, c2:     Literal) = UIntLiteral(if (c1.value == c2.value) 1 else 0, IntWidth(1))
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, IntWidth(w)) if v == BigInt(1) && w == BigInt(1) && bitWidth(rhs.tpe) == BigInt(1) => rhs
      case UIntLiteral(v, IntWidth(w)) if v == BigInt(0) && w == BigInt(1) && bitWidth(rhs.tpe) == BigInt(1) =>
        DoPrim(Not, Seq(rhs), Nil, e.tpe)
      case _ => e
    }
    def matchingArgsValue(e: DoPrim, arg: Expression) = UIntLiteral(1)
  }

  object FoldNotEqual extends FoldCommutativeOp {
    def fold(c1:    Literal, c2:     Literal) = UIntLiteral(if (c1.value != c2.value) 1 else 0, IntWidth(1))
    def simplify(e: Expression, lhs: Literal, rhs: Expression) = lhs match {
      case UIntLiteral(v, IntWidth(w)) if v == BigInt(0) && w == BigInt(1) && bitWidth(rhs.tpe) == BigInt(1) => rhs
      case UIntLiteral(v, IntWidth(w)) if v == BigInt(1) && w == BigInt(1) && bitWidth(rhs.tpe) == BigInt(1) =>
        DoPrim(Not, Seq(rhs), Nil, e.tpe)
      case _ => e
    }
    def matchingArgsValue(e: DoPrim, arg: Expression) = UIntLiteral(0)
  }

  private def foldConcat(e: DoPrim) = (e.args.head, e.args(1)) match {
    case (UIntLiteral(xv, IntWidth(xw)), UIntLiteral(yv, IntWidth(yw))) =>
      UIntLiteral(xv << yw.toInt | yv, IntWidth(xw + yw))
    case _ => e
  }

  private def foldShiftLeft(e: DoPrim) = e.consts.head.toInt match {
    case 0 => e.args.head
    case x =>
      e.args.head match {
        case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v << x, IntWidth(w + x))
        case SIntLiteral(v, IntWidth(w)) => SIntLiteral(v << x, IntWidth(w + x))
        case _                           => e
      }
  }

  private def foldDynamicShiftLeft(e: DoPrim) = e.args.last match {
    case UIntLiteral(v, IntWidth(w)) =>
      val shl = DoPrim(Shl, Seq(e.args.head), Seq(v), UnknownType)
      pad(PrimOps.set_primop_type(shl), e.tpe)
    case _ => e
  }

  private def foldDynamicShiftRight(e: DoPrim) = e.args.last match {
    case UIntLiteral(v, IntWidth(w)) =>
      val shr = DoPrim(Shr, Seq(e.args.head), Seq(v), UnknownType)
      pad(PrimOps.set_primop_type(shr), e.tpe)
    case _ => e
  }

  private def foldComparison(e: DoPrim) = {
    def foldIfZeroedArg(x: Expression): Expression = {
      def isUInt(e: Expression): Boolean = e.tpe match {
        case UIntType(_) => true
        case _           => false
      }
      def isZero(e: Expression) = e match {
        case UIntLiteral(value, _) => value == BigInt(0)
        case SIntLiteral(value, _) => value == BigInt(0)
        case _                     => false
      }
      x match {
        case DoPrim(Lt, Seq(a, b), _, _) if isUInt(a) && isZero(b)  => zero
        case DoPrim(Leq, Seq(a, b), _, _) if isZero(a) && isUInt(b) => one
        case DoPrim(Gt, Seq(a, b), _, _) if isZero(a) && isUInt(b)  => zero
        case DoPrim(Geq, Seq(a, b), _, _) if isUInt(a) && isZero(b) => one
        case ex                                                     => ex
      }
    }

    def foldIfOutsideRange(x: Expression): Expression = {
      //Note, only abides by a partial ordering
      case class Range(min: BigInt, max: BigInt) {
        def ===(that: Range) =
          Seq(this.min, this.max, that.min, that.max)
            .sliding(2, 1)
            .map(x => x.head == x(1))
            .reduce(_ && _)
        def >(that:  Range) = this.min > that.max
        def >=(that: Range) = this.min >= that.max
        def <(that:  Range) = this.max < that.min
        def <=(that: Range) = this.max <= that.min
      }
      def range(e: Expression): Range = e match {
        case UIntLiteral(value, _) => Range(value, value)
        case SIntLiteral(value, _) => Range(value, value)
        case _ =>
          e.tpe match {
            case SIntType(IntWidth(width)) =>
              Range(
                min = BigInt(0) - BigInt(2).pow(width.toInt - 1),
                max = BigInt(2).pow(width.toInt - 1) - BigInt(1)
              )
            case UIntType(IntWidth(width)) =>
              Range(
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
            case Lt if r0 < r1   => one
            case Leq if r0 <= r1 => one
            case Gt if r0 > r1   => one
            case Geq if r0 >= r1 => one
            // Always false
            case Lt if r0 >= r1 => zero
            case Leq if r0 > r1 => zero
            case Gt if r0 <= r1 => zero
            case Geq if r0 < r1 => zero
            case _              => ex
          }
        case ex => ex
      }
    }

    def foldIfMatchingArgs(x: Expression) = x match {
      case DoPrim(op, Seq(a, b), _, _) if (a == b) =>
        op match {
          case (Lt | Gt)   => zero
          case (Leq | Geq) => one
          case _           => x
        }
      case _ => x
    }
    foldIfZeroedArg(foldIfOutsideRange(foldIfMatchingArgs(e)))
  }

  final object FoldANDR extends SimplifyReductionOp {
    override def identityValue = true
    override def reduce = (a: Boolean, b: Boolean) => a & b
  }

  final object FoldORR extends SimplifyReductionOp {
    override def identityValue = false
    override def reduce = (a: Boolean, b: Boolean) => a | b
  }

  final object FoldXORR extends SimplifyReductionOp {
    override def identityValue = false
    override def reduce = (a: Boolean, b: Boolean) => a ^ b
  }

  private def constPropPrim(e: DoPrim, disabledOps: Set[PrimOp]): Expression = e.op match {
    case a if disabledOps(a)   => e
    case Shl                   => foldShiftLeft(e)
    case Dshl                  => foldDynamicShiftLeft(e)
    case Shr                   => foldShiftRight(e)
    case Dshr                  => foldDynamicShiftRight(e)
    case Cat                   => foldConcat(e)
    case Add                   => FoldADD(e)
    case Sub                   => SimplifySUB(e)
    case Div                   => SimplifyDIV(e)
    case Rem                   => SimplifyREM(e)
    case And                   => FoldAND(e)
    case Or                    => FoldOR(e)
    case Xor                   => FoldXOR(e)
    case Eq                    => FoldEqual(e)
    case Neq                   => FoldNotEqual(e)
    case Andr                  => FoldANDR(e)
    case Orr                   => FoldORR(e)
    case Xorr                  => FoldXORR(e)
    case (Lt | Leq | Gt | Geq) => foldComparison(e)
    case Not =>
      e.args.head match {
        case UIntLiteral(v, IntWidth(w)) => UIntLiteral(v ^ ((BigInt(1) << w.toInt) - 1), IntWidth(w))
        case _                           => e
      }
    case AsUInt =>
      e.args.head match {
        case SIntLiteral(v, IntWidth(w)) => UIntLiteral(v + (if (v < 0) BigInt(1) << w.toInt else 0), IntWidth(w))
        case arg =>
          arg.tpe match {
            case _: UIntType => arg
            case _ => e
          }
      }
    case AsSInt =>
      e.args.head match {
        case UIntLiteral(v, IntWidth(w)) => SIntLiteral(v - ((v >> (w.toInt - 1)) << w.toInt), IntWidth(w))
        case arg =>
          arg.tpe match {
            case _: SIntType => arg
            case _ => e
          }
      }
    case AsClock =>
      val arg = e.args.head
      arg.tpe match {
        case ClockType => arg
        case _         => e
      }
    case AsAsyncReset =>
      val arg = e.args.head
      arg.tpe match {
        case AsyncResetType => arg
        case _              => e
      }
    case Pad =>
      e.args.head match {
        case UIntLiteral(v, IntWidth(w))                     => UIntLiteral(v, IntWidth(e.consts.head.max(w)))
        case SIntLiteral(v, IntWidth(w))                     => SIntLiteral(v, IntWidth(e.consts.head.max(w)))
        case _ if bitWidth(e.args.head.tpe) >= e.consts.head => e.args.head
        case _                                               => e
      }
    case (Bits | Head | Tail) => constPropBitExtract(e)
    case _                    => e
  }

  private def constPropMuxCond(m: Mux) = m.cond match {
    case UIntLiteral(c, _) => pad(if (c == BigInt(1)) m.tval else m.fval, m.tpe)
    case _                 => m
  }

  private def constPropMux(m: Mux): Expression = (m.tval, m.fval) match {
    case _ if m.tval == m.fval => m.tval
    case (t: UIntLiteral, f: UIntLiteral)
        if t.value == BigInt(1) && f.value == BigInt(0) && bitWidth(m.tpe) == BigInt(1) =>
      m.cond
    case (t: UIntLiteral, _) if t.value == BigInt(1) && bitWidth(m.tpe) == BigInt(1) =>
      DoPrim(Or, Seq(m.cond, m.fval), Nil, m.tpe)
    case (_, f: UIntLiteral) if f.value == BigInt(0) && bitWidth(m.tpe) == BigInt(1) =>
      DoPrim(And, Seq(m.cond, m.tval), Nil, m.tpe)
    case _ => constPropMuxCond(m)
  }

  private def constPropNodeRef(r: WRef, e: Expression): Expression = {
    def doit(ex: Expression) = ex match {
      case _: UIntLiteral | _: SIntLiteral | _: WRef => ex
      case _ => r
    }
    InfoExpr.map(e)(doit)
  }

  // Is "a" a "better name" than "b"?
  private def betterName(a: String, b: String): Boolean = (a.head != '_') && (b.head == '_')

  def optimize(e: Expression): Expression =
    constPropExpression(
      new NodeMap(),
      Map.empty[Instance, OfModule],
      Map.empty[OfModule, Map[String, Literal]],
      Set.empty
    )(e)
  def optimize(e: Expression, nodeMap: NodeMap): Expression =
    constPropExpression(nodeMap, Map.empty[Instance, OfModule], Map.empty[OfModule, Map[String, Literal]], Set.empty)(e)

  private def constPropExpression(
    nodeMap:         NodeMap,
    instMap:         collection.Map[Instance, OfModule],
    constSubOutputs: Map[OfModule, Map[String, Literal]],
    disabledOps:     Set[PrimOp]
  )(e:               Expression
  ): Expression = {
    val old = e.map(constPropExpression(nodeMap, instMap, constSubOutputs, disabledOps))
    val propagated = old match {
      case p: DoPrim => constPropPrim(p, disabledOps)
      case m: Mux    => constPropMux(m)
      case ref @ WRef(rname, _, _, SourceFlow) if nodeMap.contains(rname) =>
        constPropNodeRef(ref, InfoExpr.unwrap(nodeMap(rname))._2)
      case ref @ WSubField(WRef(inst, _, InstanceKind, _), pname, _, SourceFlow) =>
        val module = instMap(inst.Instance)
        // Check constSubOutputs to see if the submodule is driving a constant
        constSubOutputs.get(module).flatMap(_.get(pname)).getOrElse(ref)
      case x => x
    }
    // We're done when the Expression no longer changes
    if (propagated eq old) propagated
    else constPropExpression(nodeMap, instMap, constSubOutputs, disabledOps)(propagated)
  }

  /** Hacky way of propagating source locators across nodes and connections that have just a
    * reference on the right-hand side
    *
    * @todo generalize source locator propagation across Expressions and delete this method
    * @todo is the `orElse` the way we want to do propagation here?
    */
  private def propagateDirectConnectionInfoOnly(nodeMap: NodeMap, dontTouch: Set[String])(stmt: Statement): Statement =
    stmt match {
      // We check rname because inlining it would cause the original declaration to go away
      case node @ DefNode(info0, name, WRef(rname, _, NodeKind, _)) if !dontTouch(rname) =>
        val (info1, _) = InfoExpr.unwrap(nodeMap(rname))
        node.copy(info = InfoExpr.orElse(info1, info0))
      case con @ Connect(info0, lhs, rref @ WRef(rname, _, NodeKind, _)) if !dontTouch(rname) =>
        val (info1, _) = InfoExpr.unwrap(nodeMap(rname))
        con.copy(info = InfoExpr.orElse(info1, info0))
      case other => other
    }

  /* Constant propagate a Module
   *
   * Two pass process
   * 1. Propagate constants in expressions and forward propagate references
   * 2. Propagate references again for backwards reference (Wires)
   * TODO Replacing all wires with nodes makes the second pass unnecessary
   *   However, preserving decent names DOES require a second pass
   *   Replacing all wires with nodes makes it unnecessary for preserving decent names to trigger an
   *   extra iteration though
   *
   * @param m the Module to run constant propagation on
   * @param dontTouches names of components local to m that should not be propagated across
   * @param instMap map of instance names to Module name
   * @param constInputs map of names of m's input ports to literal driving it (if applicable)
   * @param constSubOutputs Map of Module name to Map of output port name to literal driving it
   * @param disabledOps a Set of any PrimOps that should not be folded
   * @return (Constpropped Module, Map of output port names to literal value,
   *   Map of submodule modulenames to Map of input port names to literal values)
   */
  @tailrec
  private def constPropModule(
    m:               Module,
    dontTouches:     Set[String],
    instMap:         collection.Map[Instance, OfModule],
    constInputs:     Map[String, Literal],
    constSubOutputs: Map[OfModule, Map[String, Literal]],
    disabledOps:     Set[PrimOp]
  ): (Module, Map[String, Literal], Map[OfModule, Map[String, Seq[Literal]]]) = {

    var nPropagated = 0L
    val nodeMap = new NodeMap()
    // For cases where we are trying to constprop a bad name over a good one, we swap their names
    // during the second pass
    val swapMap = mutable.HashMap.empty[String, String]
    // Keep track of any outputs we drive with a constant
    val constOutputs = mutable.HashMap.empty[String, Literal]
    // Keep track of any submodule inputs we drive with a constant
    // (can have more than 1 of the same submodule)
    val constSubInputs = mutable.HashMap.empty[OfModule, mutable.HashMap[String, Seq[Literal]]]
    // AsyncReset registers don't have reset turned into a mux so we must be careful
    val asyncResetRegs = mutable.HashMap.empty[String, DefRegister]

    // Register constant propagation is intrinsically more complicated, as it is not feed-forward.
    // Therefore, we must store some memoized information about how nodes can be canditates for
    // forming part of a register const-prop "self-loop," where a register gets some combination of
    // self-assignments and assignments of the same literal value.
    val nodeRegCPEntries = new mutable.HashMap[String, RegCPEntry]

    // Copy constant mapping for constant inputs (except ones marked dontTouch!)
    nodeMap ++= constInputs.filterNot { case (pname, _) => dontTouches.contains(pname) }

    // Note that on back propagation we *only* worry about swapping names and propagating references
    // to constant wires, we don't need to worry about propagating primops or muxes since we'll do
    // that on the next iteration if necessary
    def backPropExpr(expr: Expression): Expression = {
      val old = expr.map(backPropExpr)
      val propagated = old match {
        // When swapping, we swap both rhs and lhs
        case ref @ WRef(rname, _, _, _) if swapMap.contains(rname) =>
          ref.copy(name = swapMap(rname))
        // Only const prop on the rhs
        case ref @ WRef(rname, _, _, SourceFlow) if nodeMap.contains(rname) =>
          constPropNodeRef(ref, InfoExpr.unwrap(nodeMap(rname))._2)
        case x => x
      }
      if (old ne propagated) {
        nPropagated += 1
      }
      propagated
    }

    def backPropStmt(stmt: Statement): Statement = stmt match {
      case reg: DefRegister if (WrappedExpression.weq(reg.init, WRef(reg))) =>
        // Self-init reset is an idiom for "no reset," and must be handled separately
        swapMap
          .get(reg.name)
          .map(newName => reg.copy(name = newName, init = WRef(reg).copy(name = newName)))
          .getOrElse(reg)
      case s =>
        s.map(backPropExpr) match {
          case decl: IsDeclaration if swapMap.contains(decl.name) =>
            val newName = swapMap(decl.name)
            nPropagated += 1
            decl match {
              case node: DefNode     => node.copy(name = newName)
              case wire: DefWire     => wire.copy(name = newName)
              case reg:  DefRegister => reg.copy(name = newName)
              case other => throwInternalError()
            }
          case other => other.map(backPropStmt)
        }
    }

    // When propagating a reference, check if we want to keep the name that would be deleted
    def propagateRef(lname: String, value: Expression, info: Info): Unit = {
      value match {
        case WRef(rname, _, kind, _) if betterName(lname, rname) && !swapMap.contains(rname) && kind != PortKind =>
          assert(!swapMap.contains(lname)) // <- Shouldn't be possible because lname is either a
          // node declaration or the single connection to a wire or register
          swapMap += lname -> rname
          swapMap += rname -> lname
        case _ =>
      }
      nodeMap(lname) = InfoExpr.wrap(info, value)
    }

    def constPropStmt(s: Statement): Statement = {
      val s0 = s.map(constPropStmt) // Statement recurse
      val s1 = propagateDirectConnectionInfoOnly(nodeMap, dontTouches)(s0) // hacky source locator propagation
      // propagate sub-Expressions
      val stmtx = s1.map(constPropExpression(nodeMap, instMap, constSubOutputs, disabledOps))
      // Record things that should be propagated
      stmtx match {
        case DefNode(info, name, value) if !dontTouches.contains(name) =>
          propagateRef(name, value, info)
        case reg: DefRegister if reg.reset.tpe == AsyncResetType =>
          asyncResetRegs(reg.name) = reg
        case Connect(info, WRef(wname, wtpe, WireKind, _), expr: Literal) if !dontTouches.contains(wname) =>
          val exprx = constPropExpression(nodeMap, instMap, constSubOutputs, disabledOps)(pad(expr, wtpe))
          propagateRef(wname, exprx, info)
        // Record constants driving outputs
        case Connect(_, WRef(pname, ptpe, PortKind, _), lit: Literal) if !dontTouches.contains(pname) =>
          val paddedLit =
            constPropExpression(nodeMap, instMap, constSubOutputs, disabledOps)(pad(lit, ptpe)).asInstanceOf[Literal]
          constOutputs(pname) = paddedLit
        // Const prop registers that are driven by a mux tree containing only instances of one constant or self-assigns
        // This requires that reset has been made explicit
        case Connect(_, lref @ WRef(lname, ltpe, RegKind, _), rhs) if !dontTouches(lname) =>
          /* Checks if an RHS expression e of a register assignment is convertible to a constant assignment.
           * Here, this means that e must be 1) a literal, 2) a self-connect, or 3) a mux tree of
           * cases (1) and (2).  In case (3), it also recursively checks that the two mux cases can
           * be resolved: each side is allowed one candidate register and one candidate literal to
           * appear in their source trees, referring to the potential constant propagation case that
           * they could allow. If the two are compatible (no different bound sources of either of
           * the two types), they can be resolved by combining sources. Otherwise, they propagate
           * NonConstant values.  When encountering a node reference, it expands the node by to its
           * RHS assignment and recurses.
           *
           * @note Some optimization of Mux trees turn 1-bit mux operators into boolean operators. This
           * can stifle register constant propagations, which looks at drivers through value-preserving
           * Muxes and Connects only. By speculatively expanding some 1-bit Or and And operations into
           * muxes, we can obtain the best possible insight on the value of the mux with a simple peephole
           * de-optimization that does not actually appear in the output code.
           *
           * @return a RegCPEntry describing the constant prop-compatible sources driving this expression
           */

          val unbound = RegCPEntry(UnboundConstant, UnboundConstant)
          val selfBound = RegCPEntry(BoundConstant(lname), UnboundConstant)

          def zero = passes.RemoveValidIf.getGroundZero(ltpe)
          def regConstant(e: Expression, baseCase: RegCPEntry): RegCPEntry = e match {
            case lit: Literal => baseCase.resolve(RegCPEntry(UnboundConstant, BoundConstant(lit)))
            case WRef(regName, _, RegKind, _) => baseCase.resolve(RegCPEntry(BoundConstant(regName), UnboundConstant))
            case WRef(nodeName, _, NodeKind, _) if nodeMap.contains(nodeName) =>
              val (_, expr) = InfoExpr.unwrap(nodeMap(nodeName))
              val cached = nodeRegCPEntries.getOrElseUpdate(nodeName, { regConstant(expr, unbound) })
              baseCase.resolve(cached)
            case Mux(_, tval, fval, _) =>
              regConstant(tval, baseCase).resolve(regConstant(fval, baseCase))
            case DoPrim(Or, Seq(a, b), _, BoolType) =>
              val aSel = regConstant(Mux(a, one, b, BoolType), baseCase)
              if (!aSel.nonConstant) aSel else regConstant(Mux(b, one, a, BoolType), baseCase)
            case DoPrim(And, Seq(a, b), _, BoolType) =>
              val aSel = regConstant(Mux(a, b, zero, BoolType), baseCase)
              if (!aSel.nonConstant) aSel else regConstant(Mux(b, a, zero, BoolType), baseCase)
            case _ =>
              RegCPEntry(NonConstant, NonConstant)
          }

          // Updates nodeMap after analyzing the returned value from regConstant
          def updateNodeMapIfConstant(e: Expression): Unit = regConstant(e, selfBound) match {
            case RegCPEntry(UnboundConstant, UnboundConstant)     => nodeMap(lname) = padCPExp(zero)
            case RegCPEntry(BoundConstant(_), UnboundConstant)    => nodeMap(lname) = padCPExp(zero)
            case RegCPEntry(UnboundConstant, BoundConstant(lit))  => nodeMap(lname) = padCPExp(lit)
            case RegCPEntry(BoundConstant(_), BoundConstant(lit)) => nodeMap(lname) = padCPExp(lit)
            case _                                                =>
          }

          def padCPExp(e: Expression) =
            constPropExpression(nodeMap, instMap, constSubOutputs, disabledOps)(pad(e, ltpe))

          asyncResetRegs.get(lname) match {
            // Normal Register
            case None => updateNodeMapIfConstant(rhs)
            // Async Register
            case Some(reg: DefRegister) => updateNodeMapIfConstant(Mux(reg.reset, reg.init, rhs))
          }

        // Mark instance inputs connected to a constant
        case Connect(_, lref @ WSubField(WRef(inst, _, InstanceKind, _), port, ptpe, _), lit: Literal) =>
          val paddedLit =
            constPropExpression(nodeMap, instMap, constSubOutputs, disabledOps)(pad(lit, ptpe)).asInstanceOf[Literal]
          val module = instMap(inst.Instance)
          val portsMap = constSubInputs.getOrElseUpdate(module, mutable.HashMap.empty)
          portsMap(port) = paddedLit +: portsMap.getOrElse(port, List.empty)
        case _ =>
      }
      // Actually transform some statements
      stmtx match {
        // Propagate connections to references
        case Connect(info0, lhs, rref @ WRef(rname, _, NodeKind, _)) if !dontTouches.contains(rname) =>
          val (info1, value) = InfoExpr.unwrap(nodeMap(rname))
          // Is this the right info combination/propagation function?
          // See propagateDirectConnectionInfoOnly
          Connect(InfoExpr.orElse(info1, info0), lhs, value)
        // If an Attach has at least 1 port, any wires are redundant and can be removed
        case Attach(info, exprs) if exprs.exists(kind(_) == PortKind) =>
          Attach(info, exprs.filterNot(kind(_) == WireKind))
        case other => other
      }
    }

    val modx = m.copy(body = backPropStmt(constPropStmt(m.body)))

    // When we call this function again, constOutputs and constSubInputs are reconstructed and
    // strictly a superset of the versions here
    if (nPropagated > 0) constPropModule(modx, dontTouches, instMap, constInputs, constSubOutputs, disabledOps)
    else (modx, constOutputs.toMap, constSubInputs.mapValues(_.toMap).toMap)
  }

  // Unify two maps using f to combine values of duplicate keys
  private def unify[K, V](a: Map[K, V], b: Map[K, V])(f: (V, V) => V): Map[K, V] =
    b.foldLeft(a) {
      case (acc, (k, v)) =>
        acc + (k -> acc.get(k).map(f(_, v)).getOrElse(v))
    }

  private def run(c: Circuit, dontTouchMap: Map[OfModule, Set[String]], disabledOps: Set[PrimOp]): Circuit = {
    val iGraph = InstanceKeyGraph(c)
    val moduleDeps = iGraph.getChildInstanceMap
    val instCount = iGraph.staticInstanceCount

    // DiGraph using Module names as nodes, destination of edge is a parent Module
    val parentGraph: DiGraph[OfModule] = iGraph.graph.reverse.transformNodes(_.OfModule)

    // This outer loop works by applying constant propagation to the modules in a topologically
    // sorted order from leaf to root
    // Modules will register any outputs they drive with a constant in constOutputs which is then
    // checked by later modules in the same iteration (since we iterate from leaf to root)
    // Since Modules can be instantiated multiple times, for inputs we must check that all instances
    // are driven with the same constant value. Then, if we find a Module input where each instance
    // is driven with the same constant (and not seen in a previous iteration), we iterate again
    @tailrec
    def iterate(
      toVisit:     Set[OfModule],
      modules:     Map[OfModule, Module],
      constInputs: Map[OfModule, Map[String, Literal]]
    ): Map[OfModule, DefModule] = {
      if (toVisit.isEmpty) modules
      else {
        // Order from leaf modules to root so that any module driving an output
        // with a constant will be visible to modules that instantiate it
        // TODO Generating order as we execute constant propagation on each module would be faster
        val order = parentGraph.subgraph(toVisit).linearize
        // Execute constant propagation on each module in order
        // Aggreagte Module outputs that are driven constant for use by instaniating Modules
        // Aggregate submodule inputs driven constant for checking later
        val (modulesx, _, constInputsx) =
          order.foldLeft((modules, Map[OfModule, Map[String, Literal]](), Map[OfModule, Map[String, Seq[Literal]]]())) {
            case ((mmap, constOutputs, constInputsAcc), mname) =>
              val dontTouches = dontTouchMap.getOrElse(mname, Set.empty)
              val (mx, mco, mci) = constPropModule(
                modules(mname),
                dontTouches,
                moduleDeps(mname),
                constInputs.getOrElse(mname, Map.empty),
                constOutputs,
                disabledOps
              )
              // Accumulate all Literals used to drive a particular Module port
              val constInputsx = unify(constInputsAcc, mci)((a, b) => unify(a, b)((c, d) => c ++ d))
              (mmap + (mname -> mx), constOutputs + (mname -> mco), constInputsx)
          }
        // Determine which module inputs have all of the same, new constants driving them
        val newProppedInputs = constInputsx.flatMap {
          case (mname, ports) =>
            val portsx = ports.flatMap {
              case (pname, lits) =>
                val newPort = !constInputs.get(mname).map(_.contains(pname)).getOrElse(false)
                val isModule = modules.contains(mname) // ExtModules are not contained in modules
                val allSameConst = lits.size == instCount(mname) && lits.toSet.size == 1
                if (isModule && newPort && allSameConst) Some(pname -> lits.head)
                else None
            }
            if (portsx.nonEmpty) Some(mname -> portsx) else None
        }
        val modsWithConstInputs = newProppedInputs.keySet
        val newToVisit = modsWithConstInputs ++
          modsWithConstInputs.flatMap(parentGraph.reachableFrom)
        // Combine const inputs (there can't be duplicate values in the inner maps)
        val nextConstInputs = unify(constInputs, newProppedInputs)((a, b) => a ++ b)
        iterate(newToVisit.toSet, modulesx, nextConstInputs)
      }
    }

    val modulesx = {
      val nameMap = c.modules.collect { case m: Module => m.OfModule -> m }.toMap
      // We only pass names of Modules, we can't apply const prop to ExtModules
      val mmap = iterate(nameMap.keySet, nameMap, Map.empty)
      c.modules.map(m => mmap.getOrElse(m.OfModule, m))
    }

    Circuit(c.info, modulesx, c.main)
  }

  def execute(state: CircuitState): CircuitState = {
    val dontTouches: Seq[(OfModule, String)] = state.annotations.flatMap {
      case anno: HasDontTouches =>
        anno.dontTouches
          // We treat all ReferenceTargets as if they were local because of limitations of
          // EliminateTargetPaths
          .map(rt => OfModule(rt.encapsulatingModule) -> rt.ref)
      case o => Nil
    }

    // Map from module name to component names
    val dontTouchMap: Map[OfModule, Set[String]] =
      dontTouches.groupBy(_._1).mapValues(_.map(_._2).toSet).toMap

    val disabledOps = state.annotations.collect { case DisableFold(op) => op }.toSet

    state.copy(circuit = run(state.circuit, dontTouchMap, disabledOps))
  }
}
