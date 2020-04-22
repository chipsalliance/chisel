// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Utils._
import firrtl.traversals.Foreachers._
import firrtl.WrappedType._
import firrtl.constraint.{Constraint, IsKnown}
import firrtl.options.{Dependency, PreservesAll}

object CheckTypes extends Pass with PreservesAll[Transform] {

  override def prerequisites = Dependency(InferTypes) +: firrtl.stage.Forms.WorkingIR

  override def optionalPrerequisiteOf =
    Seq( Dependency(passes.Uniquify),
         Dependency(passes.ResolveFlows),
         Dependency(passes.CheckFlows),
         Dependency[passes.InferWidths],
         Dependency(passes.CheckWidths) )

  // Custom Exceptions
  class SubfieldNotInBundle(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname ]  Subfield $name is not in bundle.")
  class SubfieldOnNonBundle(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname]  Subfield $name is accessed on a non-bundle.")
  class IndexTooLarge(info: Info, mname: String, value: Int) extends PassException(
    s"$info: [module $mname]  Index with value $value is too large.")
  class IndexOnNonVector(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Index illegal on non-vector type.")
  class AccessIndexNotUInt(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Access index must be a UInt type.")
  class IndexNotUInt(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Index is not of UIntType.")
  class EnableNotUInt(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Enable is not of UIntType.")
  class InvalidConnect(info: Info, mname: String, con: String, lhs: Expression, rhs: Expression)
      extends PassException({
    val ltpe = s"  ${lhs.serialize}: ${lhs.tpe.serialize}"
    val rtpe = s"  ${rhs.serialize}: ${rhs.tpe.serialize}"
    s"$info: [module $mname]  Type mismatch in '$con'.\n$ltpe\n$rtpe"
  })
  class InvalidRegInit(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Type of init must match type of DefRegister.")
  class PrintfArgNotGround(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Printf arguments must be either UIntType or SIntType.")
  class ReqClk(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Requires a clock typed signal.")
  class RegReqClk(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname]  Register $name requires a clock typed signal.")
  class EnNotUInt(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Enable must be a UIntType typed signal.")
  class PredNotUInt(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Predicate not a UIntType.")
  class OpNotGround(info: Info, mname: String, op: String) extends PassException(
    s"$info: [module $mname]  Primop $op cannot operate on non-ground types.")
  class OpNotUInt(info: Info, mname: String, op: String, e: String) extends PassException(
    s"$info: [module $mname]  Primop $op requires argument $e to be a UInt type.")
  class OpNotAllUInt(info: Info, mname: String, op: String) extends PassException(
    s"$info: [module $mname]  Primop $op requires all arguments to be UInt type.")
  class OpNotAllSameType(info: Info, mname: String, op: String) extends PassException(
    s"$info: [module $mname]  Primop $op requires all operands to have the same type.")
  class OpNoMixFix(info:Info, mname: String, op: String) extends PassException(s"${info}: [module ${mname}]  Primop ${op} cannot operate on args of some, but not all, fixed type.")
  class OpNotCorrectType(info:Info, mname: String, op: String, tpes: Seq[String]) extends PassException(s"${info}: [module ${mname}]  Primop ${op} does not have correct arg types: $tpes.")
  class OpNotAnalog(info: Info, mname: String, exp: String) extends PassException(
    s"$info: [module $mname]  Attach requires all arguments to be Analog type: $exp.")
  class NodePassiveType(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Node must be a passive type.")
  class MuxSameType(info: Info, mname: String, t1: String, t2: String) extends PassException(
    s"$info: [module $mname]  Must mux between equivalent types: $t1 != $t2.")
  class MuxPassiveTypes(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Must mux between passive types.")
  class MuxCondUInt(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  A mux condition must be of type UInt.")
  class MuxClock(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Firrtl does not support muxing clocks.")
  class ValidIfPassiveTypes(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Must validif a passive type.")
  class ValidIfCondUInt(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  A validif condition must be of type UInt.")
  class IllegalAnalogDeclaration(info: Info, mname: String, decName: String) extends PassException(
    s"$info: [module $mname]  Cannot declare a reg, node, or memory with an Analog type: $decName.")
  class IllegalAttachExp(info: Info, mname: String, expName: String) extends PassException(
    s"$info: [module $mname]  Attach expression must be an port, wire, or port of instance: $expName.")
  class IllegalResetType(info: Info, mname: String, exp: String) extends PassException(
    s"$info: [module $mname]  Register resets must have type Reset, AsyncReset, or UInt<1>: $exp.")
  class IllegalUnknownType(info: Info, mname: String, exp: String) extends PassException(
    s"$info: [module $mname]  Uninferred type: $exp."
  )

  def fits(bigger: Constraint, smaller: Constraint): Boolean = (bigger, smaller) match {
    case (IsKnown(v1), IsKnown(v2)) if v1 < v2 => false
    case _ => true
  }

  def legalResetType(tpe: Type): Boolean = tpe match {
    case UIntType(IntWidth(w)) if w == 1 => true
    case AsyncResetType => true
    case ResetType => true
    case UIntType(UnknownWidth) =>
      // cannot catch here, though width may ultimately be wrong
      true
    case _ => false
  }

  private def bulk_equals(t1: Type, t2: Type, flip1: Orientation, flip2: Orientation): Boolean = {
    (t1, t2) match {
      case (ClockType, ClockType) => flip1 == flip2
      case (_: UIntType, _: UIntType) => flip1 == flip2
      case (_: SIntType, _: SIntType) => flip1 == flip2
      case (_: FixedType, _: FixedType) => flip1 == flip2
      case (i1: IntervalType, i2: IntervalType) =>
        import Implicits.width2constraint
        fits(i2.lower, i1.lower) && fits(i1.upper, i2.upper) && fits(i1.point, i2.point)
      case (_: AnalogType, _: AnalogType) => true
      case (AsyncResetType, AsyncResetType) => flip1 == flip2
      case (ResetType, tpe) => legalResetType(tpe) && flip1 == flip2
      case (tpe, ResetType) => legalResetType(tpe) && flip1 == flip2
      case (t1: BundleType, t2: BundleType) =>
        val t1_fields = (t1.fields foldLeft Map[String, (Type, Orientation)]())(
          (map, f1) => map + (f1.name ->( (f1.tpe, f1.flip) )))
        t2.fields forall (f2 =>
          t1_fields get f2.name match {
            case None => true
            case Some((f1_tpe, f1_flip)) =>
              bulk_equals(f1_tpe, f2.tpe, times(flip1, f1_flip), times(flip2, f2.flip))
          }
        )
      case (t1: VectorType, t2: VectorType) =>
        bulk_equals(t1.tpe, t2.tpe, flip1, flip2)
      case (_, _) => false
    }
  }

  def validConnect(locTpe: Type, expTpe: Type): Boolean = {
    val itFits = (locTpe, expTpe) match {
      case (i1: IntervalType, i2: IntervalType) =>
        import Implicits.width2constraint
        fits(i2.lower, i1.lower) && fits(i1.upper, i2.upper) && fits(i1.point, i2.point)
      case _ => true
    }
    wt(locTpe).superTypeOf(wt(expTpe)) && itFits
  }

  def validConnect(con: Connect): Boolean = validConnect(con.loc.tpe, con.expr.tpe)

  def validPartialConnect(con: PartialConnect): Boolean =
    bulk_equals(con.loc.tpe, con.expr.tpe, Default, Default)

  //;---------------- Helper Functions --------------
  def ut: UIntType = UIntType(UnknownWidth)
  def st: SIntType = SIntType(UnknownWidth)

  def run (c:Circuit) : Circuit = {
    val errors = new Errors()

    def passive(t: Type): Boolean = t match {
      case _: UIntType |_: SIntType => true
      case tx: VectorType => passive(tx.tpe)
      case tx: BundleType => tx.fields forall (x => x.flip == Default && passive(x.tpe))
      case tx => true
    }

    def check_types_primop(info: Info, mname: String, e: DoPrim): Unit = {
      def checkAllTypes(exprs: Seq[Expression], okUInt: Boolean, okSInt: Boolean, okClock: Boolean, okFix: Boolean, okAsync: Boolean, okInterval: Boolean): Unit = {
        exprs.foldLeft((false, false, false, false, false, false)) {
          case ((isUInt, isSInt, isClock, isFix, isAsync, isInterval), expr) => expr.tpe match {
            case u: UIntType    => (true,   isSInt, isClock, isFix, isAsync, isInterval)
            case s: SIntType    => (isUInt, true,   isClock, isFix, isAsync, isInterval)
            case ClockType      => (isUInt, isSInt, true,    isFix, isAsync, isInterval)
            case f: FixedType   => (isUInt, isSInt, isClock, true,  isAsync, isInterval)
            case AsyncResetType => (isUInt, isSInt, isClock, isFix, true,    isInterval)
            case i:IntervalType => (isUInt, isSInt, isClock, isFix, isAsync, true)
            case UnknownType    =>
              errors.append(new IllegalUnknownType(info, mname, e.serialize))
              (isUInt, isSInt, isClock, isFix, isAsync, isInterval)
            case other => throwInternalError(s"Illegal Type: ${other.serialize}")
          }
        } match {
          //   (UInt,  SInt,  Clock, Fixed, Async, Interval)
          case (isAll, false, false, false, false, false) if isAll == okUInt  =>
          case (false, isAll, false, false, false, false) if isAll == okSInt  =>
          case (false, false, isAll, false, false, false) if isAll == okClock =>
          case (false, false, false, isAll, false, false) if isAll == okFix   =>
          case (false, false, false, false, isAll, false) if isAll == okAsync =>
          case (false, false, false, false, false, isAll) if isAll == okInterval =>
          case x => errors.append(new OpNotCorrectType(info, mname, e.op.serialize, exprs.map(_.tpe.serialize)))
        }
      }
      e.op match {
        case AsUInt | AsSInt | AsClock | AsFixedPoint | AsAsyncReset | AsInterval =>
          // All types are ok
        case Dshl | Dshr =>
          checkAllTypes(Seq(e.args.head), okUInt=true, okSInt=true,  okClock=false, okFix=true,  okAsync=false, okInterval=true)
          checkAllTypes(Seq(e.args(1)),   okUInt=true, okSInt=false, okClock=false, okFix=false, okAsync=false, okInterval=false)
        case Add | Sub | Mul | Lt | Leq | Gt | Geq | Eq | Neq =>
          checkAllTypes(e.args, okUInt=true, okSInt=true, okClock=false, okFix=true, okAsync=false, okInterval=true)
        case Pad | Bits | Head | Tail =>
          checkAllTypes(e.args, okUInt=true, okSInt=true, okClock=false, okFix=true, okAsync=false, okInterval=false)
        case Shl | Shr | Cat =>
          checkAllTypes(e.args, okUInt=true, okSInt=true, okClock=false, okFix=true, okAsync=false, okInterval=true)
        case IncP | DecP | SetP =>
          checkAllTypes(e.args, okUInt=false, okSInt=false, okClock=false, okFix=true, okAsync=false, okInterval=true)
        case Wrap | Clip | Squeeze =>
          checkAllTypes(e.args, okUInt = false, okSInt = false, okClock = false, okFix = false, okAsync=false, okInterval = true)
        case _ =>
          checkAllTypes(e.args, okUInt=true, okSInt=true, okClock=false, okFix=false, okAsync=false, okInterval=false)
      }
    }

    def check_types_e(info:Info, mname: String)(e: Expression): Unit = {
      e match {
        case (e: WSubField) => e.expr.tpe match {
          case (t: BundleType) => t.fields find (_.name == e.name) match {
            case Some(_) =>
            case None => errors.append(new SubfieldNotInBundle(info, mname, e.name))
          }
          case _ => errors.append(new SubfieldOnNonBundle(info, mname, e.name))
        }
        case (e: WSubIndex) => e.expr.tpe match {
          case (t: VectorType) if e.value < t.size =>
          case (t: VectorType) =>
            errors.append(new IndexTooLarge(info, mname, e.value))
          case _ =>
            errors.append(new IndexOnNonVector(info, mname))
        }
        case (e: WSubAccess) =>
          e.expr.tpe match {
            case _: VectorType =>
            case _ => errors.append(new IndexOnNonVector(info, mname))
          }
          e.index.tpe match {
            case _: UIntType =>
            case _ => errors.append(new AccessIndexNotUInt(info, mname))
          }
        case (e: DoPrim) => check_types_primop(info, mname, e)
        case (e: Mux) =>
          if (wt(e.tval.tpe) != wt(e.fval.tpe))
            errors.append(new MuxSameType(info, mname, e.tval.tpe.serialize, e.fval.tpe.serialize))
          if (!passive(e.tpe))
            errors.append(new MuxPassiveTypes(info, mname))
          e.cond.tpe match {
            case _: UIntType =>
            case _ => errors.append(new MuxCondUInt(info, mname))
          }
        case (e: ValidIf) =>
          if (!passive(e.tpe))
            errors.append(new ValidIfPassiveTypes(info, mname))
          e.cond.tpe match {
            case _: UIntType =>
            case _ => errors.append(new ValidIfCondUInt(info, mname))
          }
        case _ =>
      }
      e foreach check_types_e(info, mname)
    }

    def check_types_s(minfo: Info, mname: String)(s: Statement): Unit = {
      val info = get_info(s) match { case NoInfo => minfo case x => x }
      s match {
        case sx: Connect if !validConnect(sx) =>
          val conMsg = sx.copy(info = NoInfo).serialize
          errors.append(new InvalidConnect(info, mname, conMsg, sx.loc, sx.expr))
        case sx: PartialConnect if !validPartialConnect(sx) =>
          val conMsg = sx.copy(info = NoInfo).serialize
          errors.append(new InvalidConnect(info, mname, conMsg, sx.loc, sx.expr))
        case sx: DefRegister =>
          sx.tpe match {
            case AnalogType(_) => errors.append(new IllegalAnalogDeclaration(info, mname, sx.name))
            case t if wt(sx.tpe) != wt(sx.init.tpe) => errors.append(new InvalidRegInit(info, mname))
            case t if !validConnect(sx.tpe, sx.init.tpe) =>
              val conMsg = sx.copy(info = NoInfo).serialize
              errors.append(new CheckTypes.InvalidConnect(info, mname, conMsg, WRef(sx), sx.init))
            case t =>
          }
          if (!legalResetType(sx.reset.tpe)) {
            errors.append(new IllegalResetType(info, mname, sx.name))
          }
          if (sx.clock.tpe != ClockType) {
            errors.append(new RegReqClk(info, mname, sx.name))
          }
        case sx: Conditionally if wt(sx.pred.tpe) != wt(ut) =>
          errors.append(new PredNotUInt(info, mname))
        case sx: DefNode => sx.value.tpe match {
          case AnalogType(w) => errors.append(new IllegalAnalogDeclaration(info, mname, sx.name))
          case t if !passive(sx.value.tpe) => errors.append(new NodePassiveType(info, mname))
          case t =>
        }
        case sx: Attach =>
          for (e <- sx.exprs) {
            e.tpe match {
              case _: AnalogType =>
              case _ => errors.append(new OpNotAnalog(info, mname, e.serialize))
            }
            kind(e) match {
              case (InstanceKind | PortKind | WireKind) =>
              case _ =>  errors.append(new IllegalAttachExp(info, mname, e.serialize))
            }
          }
        case sx: Stop =>
          if (wt(sx.clk.tpe) != wt(ClockType)) errors.append(new ReqClk(info, mname))
          if (wt(sx.en.tpe) != wt(ut)) errors.append(new EnNotUInt(info, mname))
        case sx: Print =>
          if (sx.args exists (x => wt(x.tpe) != wt(ut) && wt(x.tpe) != wt(st)))
            errors.append(new PrintfArgNotGround(info, mname))
          if (wt(sx.clk.tpe) != wt(ClockType)) errors.append(new ReqClk(info, mname))
          if (wt(sx.en.tpe) != wt(ut)) errors.append(new EnNotUInt(info, mname))
        case sx: DefMemory => sx.dataType match {
          case AnalogType(w) => errors.append(new IllegalAnalogDeclaration(info, mname, sx.name))
          case t =>
        }
        case _ =>
      }
      s foreach check_types_e(info, mname)
      s foreach check_types_s(info, mname)
    }

    c.modules foreach (m => m foreach check_types_s(m.info, m.name))
    errors.trigger()
    c
  }
}
