// See LICENSE for license details.

package firrtl.passes

import firrtl.PrimOps._
import firrtl.ir._
import firrtl._
import firrtl.Mappers._
import Implicits.{bigint2WInt}
import firrtl.constraint.IsKnown
import firrtl.options.{Dependency, PreservesAll}

import scala.math.BigDecimal.RoundingMode._

class WrapWithRemainder(info: Info, mname: String, wrap: DoPrim)
  extends PassException({
    val toWrap = wrap.args.head.serialize
    val toWrapTpe = wrap.args.head.tpe.serialize
    val wrapTo = wrap.args(1).serialize
    val wrapToTpe = wrap.args(1).tpe.serialize
    s"$info: [module $mname] Wraps with remainder currently unsupported: $toWrap:$toWrapTpe cannot be wrapped to $wrapTo's type $wrapToTpe"
  })


/** Replaces IntervalType with SIntType, three AST walks:
  * 1) Align binary points
  *    - adds shift operators to primop args and connections
  *    - does not affect declaration- or inferred-types
  * 2) Replace Interval [[DefNode]] with [[DefWire]] + [[Connect]]
  *    - You have to do this to capture the smaller bitwidths of nodes that intervals give you. Otherwise, any future
  *    InferTypes would reinfer the larger widths on these nodes from SInt width inference rules
  * 3) Replace declaration IntervalType's with SIntType's
  *    - for each declaration:
  *      a. remove non-zero binary points
  *      b. remove open bounds
  *      c. replace with SIntType
  * 3) Run InferTypes
  */
class RemoveIntervals extends Pass with PreservesAll[Transform] {

  override val prerequisites: Seq[Dependency[Transform]] =
    Seq( Dependency(PullMuxes),
         Dependency(ReplaceAccesses),
         Dependency(ExpandConnects),
         Dependency(RemoveAccesses),
         Dependency[ExpandWhensAndCheck] ) ++ firrtl.stage.Forms.Deduped

  def run(c: Circuit): Circuit = {
    val alignedCircuit = c
    val errors = new Errors()
    val wiredCircuit = alignedCircuit map makeWireModule
    val replacedCircuit = wiredCircuit map replaceModuleInterval(errors)
    errors.trigger()
    InferTypes.run(replacedCircuit)
  }

  /* Replace interval types */
  private def replaceModuleInterval(errors: Errors)(m: DefModule): DefModule =
    m map replaceStmtInterval(errors, m.name) map replacePortInterval

  private def replaceStmtInterval(errors: Errors, mname: String)(s: Statement): Statement = {
    val info = s match {
      case h: HasInfo => h.info
      case _ => NoInfo
    }
    s map replaceTypeInterval map replaceStmtInterval(errors, mname) map replaceExprInterval(errors, info, mname)

  }

  private def replaceExprInterval(errors: Errors, info: Info, mname: String)(e: Expression): Expression = e match {
    case _: WRef | _: WSubIndex | _: WSubField => e
    case o =>
      o map replaceExprInterval(errors, info, mname) match {
        case DoPrim(AsInterval, Seq(a1), _, tpe) => DoPrim(AsSInt, Seq(a1), Seq.empty, tpe)
        case DoPrim(IncP, args, consts, tpe) => DoPrim(Shl, args, consts, tpe)
        case DoPrim(DecP, args, consts, tpe) => DoPrim(Shr, args, consts, tpe)
        case DoPrim(Clip, Seq(a1, _), Nil, tpe: IntervalType) =>
          // Output interval (pre-calculated)
          val clipLo = tpe.minAdjusted.get
          val clipHi = tpe.maxAdjusted.get
          // Input interval
          val (inLow, inHigh) = a1.tpe match {
            case t2: IntervalType => (t2.minAdjusted.get, t2.maxAdjusted.get)
            case _ => sys.error("Shouldn't be here")
          }
          val gtOpt = clipHi >= inHigh
          val ltOpt = clipLo <= inLow
          (gtOpt, ltOpt) match {
            // input range within output range -> no optimization
            case (true, true) => a1
            case (true, false) => Mux(Lt(a1, clipLo.S), clipLo.S, a1)
            case (false, true) => Mux(Gt(a1, clipHi.S), clipHi.S, a1)
            case _ => Mux(Gt(a1, clipHi.S), clipHi.S, Mux(Lt(a1, clipLo.S), clipLo.S, a1))
          }

        case sqz@DoPrim(Squeeze, Seq(a1, a2), Nil, tpe: IntervalType) =>
          // Using (conditional) reassign interval w/o adding mux
          val a1tpe = a1.tpe.asInstanceOf[IntervalType]
          val a2tpe = a2.tpe.asInstanceOf[IntervalType]
          val min2 = a2tpe.min.get * BigDecimal(BigInt(1) << a1tpe.point.asInstanceOf[IntWidth].width.toInt)
          val max2 = a2tpe.max.get * BigDecimal(BigInt(1) << a1tpe.point.asInstanceOf[IntWidth].width.toInt)
          val w1 = Seq(a1tpe.minAdjusted.get.bitLength, a1tpe.maxAdjusted.get.bitLength).max + 1
          // Conservative
          val minOpt2 = min2.setScale(0, FLOOR).toBigInt
          val maxOpt2 = max2.setScale(0, CEILING).toBigInt
          val w2 = Seq(minOpt2.bitLength, maxOpt2.bitLength).max + 1
          if (w1 < w2) {
            a1
          } else {
            val bits = DoPrim(Bits, Seq(a1), Seq(w2 - 1, 0), UIntType(IntWidth(w2)))
            DoPrim(AsSInt, Seq(bits), Seq.empty, SIntType(IntWidth(w2)))
          }
        case w@DoPrim(Wrap, Seq(a1, a2), Nil, tpe: IntervalType) => a2.tpe match {
          // If a2 type is Interval wrap around range. If UInt, wrap around width
          case t: IntervalType =>
            // Need to match binary points before getting *adjusted!
            val (wrapLo, wrapHi) = t.copy(point = tpe.point) match {
              case t: IntervalType => (t.minAdjusted.get, t.maxAdjusted.get)
              case _ => Utils.throwInternalError(s"Illegal AST state: cannot have $e not have an IntervalType")
            }
            val (inLo, inHi) = a1.tpe match {
              case t2: IntervalType => (t2.minAdjusted.get, t2.maxAdjusted.get)
              case _ => sys.error("Shouldn't be here")
            }
            // If (max input) - (max wrap) + (min wrap) is less then (maxwrap), we can optimize when (max input > max wrap)
            val range = wrapHi - wrapLo
            val ltOpt = Add(a1, (range + 1).S)
            val gtOpt = Sub(a1, (range + 1).S)
            // [Angie]: This is dangerous. Would rather throw compilation error right now than allow "Rem" without the user explicitly including it.
            // If x < wl
            // output: wh - (wl - x) + 1 AKA x + r + 1
            // worst case: wh - (wl - xl) + 1 = wl
            // -> xl + wr + 1 = wl
            // If x > wh
            // output: wl + (x - wh) - 1 AKA x - r - 1
            // worst case: wl + (xh - wh) - 1 = wh
            // -> xh - wr - 1 = wh
            val default = Add(Rem(Sub(a1, wrapLo.S), Sub(wrapHi.S, wrapLo.S)), wrapLo.S)
            (wrapHi >= inHi, wrapLo <= inLo, (inHi - range - 1) <= wrapHi, (inLo + range + 1) >= wrapLo) match {
              case (true, true, _, _) => a1
              case (true, _, _, true) => Mux(Lt(a1, wrapLo.S), ltOpt, a1)
              case (_, true, true, _) => Mux(Gt(a1, wrapHi.S), gtOpt, a1)
              // Note: inHi - range - 1 = wrapHi can't be true when inLo + range + 1 = wrapLo (i.e. simultaneous extreme cases don't work)
              case (_, _, true, true) => Mux(Gt(a1, wrapHi.S), gtOpt, Mux(Lt(a1, wrapLo.S), ltOpt, a1))
              case _ =>
                errors.append(new WrapWithRemainder(info, mname, w))
                default
            }
          case _ => sys.error("Shouldn't be here")
        }
        case other => other
      }
  }

  private def replacePortInterval(p: Port): Port = p map replaceTypeInterval

  private def replaceTypeInterval(t: Type): Type = t match {
    case i@IntervalType(l: IsKnown, u: IsKnown, p: IntWidth) => SIntType(i.width)
    case i: IntervalType => sys.error(s"Shouldn't be here: $i")
    case v => v map replaceTypeInterval
  }

  /** Replace Interval Nodes with Interval Wires
    *
    * You have to do this to capture the smaller bitwidths of nodes that intervals give you. Otherwise,
    *   any future InferTypes would reinfer the larger widths on these nodes from SInt width inference rules
    * @param m module to replace nodes with wire + connection
    * @return
    */
  private def makeWireModule(m: DefModule): DefModule = m map makeWireStmt

  private def makeWireStmt(s: Statement): Statement = s match {
    case DefNode(info, name, value) => value.tpe match {
      case IntervalType(l, u, p) =>
        val newType = IntervalType(l, u, p)
        Block(Seq(DefWire(info, name, newType), Connect(info, WRef(name, newType, WireKind, FEMALE), value)))
      case other => s
    }
    case other => other map makeWireStmt
  }
}
