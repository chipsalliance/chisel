// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Utils._
import firrtl.traversals.Foreachers._
import firrtl.WrappedType._
import firrtl.constraint.{Constraint, IsKnown}

trait CheckHighFormLike {
  type NameSet = collection.mutable.HashSet[String]

  // Custom Exceptions
  class NotUniqueException(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname] Reference $name does not have a unique name.")
  class InvalidLOCException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Invalid connect to an expression that is not a reference or a WritePort.")
  class NegUIntException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] UIntLiteral cannot be negative.")
  class UndeclaredReferenceException(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname] Reference $name is not declared.")
  class PoisonWithFlipException(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname] Poison $name cannot be a bundle type with flips.")
  class MemWithFlipException(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname] Memory $name cannot be a bundle type with flips.")
  class IllegalMemLatencyException(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname] Memory $name must have non-negative read latency and positive write latency.")
  class RegWithFlipException(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname] Register $name cannot be a bundle type with flips.")
  class InvalidAccessException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Invalid access to non-reference.")
  class ModuleNotDefinedException(info: Info, mname: String, name: String) extends PassException(
    s"$info: Module $name is not defined.")
  class IncorrectNumArgsException(info: Info, mname: String, op: String, n: Int) extends PassException(
    s"$info: [module $mname] Primop $op requires $n expression arguments.")
  class IncorrectNumConstsException(info: Info, mname: String, op: String, n: Int) extends PassException(
    s"$info: [module $mname] Primop $op requires $n integer arguments.")
  class NegWidthException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Width cannot be negative or zero.")
  class NegVecSizeException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Vector type size cannot be negative.")
  class NegMemSizeException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Memory size cannot be negative or zero.")
  class BadPrintfException(info: Info, mname: String, x: Char) extends PassException(
    s"$info: [module $mname] Bad printf format: " + "\"%" + x + "\"")
  class BadPrintfTrailingException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Bad printf format: trailing " + "\"%\"")
  class BadPrintfIncorrectNumException(info: Info, mname: String) extends PassException(
    s"$info: [module $mname] Bad printf format: incorrect number of arguments")
  class InstanceLoop(info: Info, mname: String, loop: String) extends PassException(
    s"$info: [module $mname] Has instance loop $loop")
  class NoTopModuleException(info: Info, name: String) extends PassException(
    s"$info: A single module must be named $name.")
  class NegArgException(info: Info, mname: String, op: String, value: Int) extends PassException(
    s"$info: [module $mname] Primop $op argument $value < 0.")
  class LsbLargerThanMsbException(info: Info, mname: String, op: String, lsb: Int, msb: Int) extends PassException(
    s"$info: [module $mname] Primop $op lsb $lsb > $msb.")
  class ResetInputException(info: Info, mname: String, expr: Expression) extends PassException(
    s"$info: [module $mname] Abstract Reset not allowed as top-level input: ${expr.serialize}")
  class ResetExtModuleOutputException(info: Info, mname: String, expr: Expression) extends PassException(
    s"$info: [module $mname] Abstract Reset not allowed as ExtModule output: ${expr.serialize}")


  // Is Chirrtl allowed for this check? If not, return an error
  def errorOnChirrtl(info: Info, mname: String, s: Statement): Option[PassException]

  def run(c: Circuit): Circuit = {
    val errors = new Errors()
    val moduleGraph = new ModuleGraph
    val moduleNames = (c.modules map (_.name)).toSet

    def checkHighFormPrimop(info: Info, mname: String, e: DoPrim): Unit = {
      def correctNum(ne: Option[Int], nc: Int): Unit = {
        ne match {
          case Some(i) if e.args.length != i =>
            errors.append(new IncorrectNumArgsException(info, mname, e.op.toString, i))
          case _ => // Do Nothing
        }
        if (e.consts.length != nc)
          errors.append(new IncorrectNumConstsException(info, mname, e.op.toString, nc))
      }

      e.op match {
        case Add | Sub | Mul | Div | Rem | Lt | Leq | Gt | Geq |
             Eq | Neq | Dshl | Dshr | And | Or | Xor | Cat | Dshlw | Clip | Wrap | Squeeze =>
          correctNum(Option(2), 0)
        case AsUInt | AsSInt | AsClock | AsAsyncReset | Cvt | Neq | Not =>
          correctNum(Option(1), 0)
        case AsFixedPoint | Pad | Head | Tail | IncP | DecP | SetP =>
          correctNum(Option(1), 1)
        case Shl | Shr =>
          correctNum(Option(1), 1)
          val amount = e.consts.map(_.toInt).filter(_ < 0).foreach {
            c => errors.append(new NegArgException(info, mname, e.op.toString, c))
          }
        case Bits =>
          correctNum(Option(1), 2)
          val (msb, lsb) = (e.consts(0).toInt, e.consts(1).toInt)
          if (lsb > msb) {
            errors.append(new LsbLargerThanMsbException(info, mname, e.op.toString, lsb, msb))
          }
        case AsInterval =>
          correctNum(Option(1), 3)
        case Andr | Orr | Xorr | Neg =>
          correctNum(None,0)
      }
    }

    def checkFstring(info: Info, mname: String, s: StringLit, i: Int): Unit = {
      val validFormats = "bdxc"
      val (percent, npercents) = s.string.foldLeft((false, 0)) {
        case ((percentx, n), b) if percentx && (validFormats contains b) =>
          (false, n + 1)
        case ((percentx, n), b) if percentx && b != '%' =>
          errors.append(new BadPrintfException(info, mname, b.toChar))
          (false, n)
        case ((percentx, n), b) =>
          (if (b == '%') !percentx else false /* %% -> percentx = false */, n)
      }
      if (percent) errors.append(new BadPrintfTrailingException(info, mname))
      if (npercents != i) errors.append(new BadPrintfIncorrectNumException(info, mname))
    }

    def checkValidLoc(info: Info, mname: String, e: Expression): Unit = e match {
      case _: UIntLiteral | _: SIntLiteral | _: DoPrim =>
        errors.append(new InvalidLOCException(info, mname))
      case _ => // Do Nothing
    }

    def checkHighFormW(info: Info, mname: String)(w: Width): Unit = {
      w match {
        case wx: IntWidth if wx.width < 0 => errors.append(new NegWidthException(info, mname))
        case wx => // Do nothing
      }
    }

    def checkHighFormT(info: Info, mname: String)(t: Type): Unit = {
      t foreach checkHighFormT(info, mname)
      t match {
        case tx: VectorType if tx.size < 0 =>
          errors.append(new NegVecSizeException(info, mname))
        case _: IntervalType =>
        case _ => t foreach checkHighFormW(info, mname)
      }
    }

    def validSubexp(info: Info, mname: String)(e: Expression): Unit = {
      e match {
        case _: Reference | _: SubField | _: SubIndex | _: SubAccess => // No error
        case _: WRef | _: WSubField | _: WSubIndex | _: WSubAccess | _: Mux | _: ValidIf => // No error
        case _ => errors.append(new InvalidAccessException(info, mname))
      }
    }

    def checkHighFormE(info: Info, mname: String, names: NameSet)(e: Expression): Unit = {
      e match {
        case ex: Reference if !names(ex.name) =>
          errors.append(new UndeclaredReferenceException(info, mname, ex.name))
        case ex: WRef if !names(ex.name) =>
          errors.append(new UndeclaredReferenceException(info, mname, ex.name))
        case ex: UIntLiteral if ex.value < 0 =>
          errors.append(new NegUIntException(info, mname))
        case ex: DoPrim => checkHighFormPrimop(info, mname, ex)
        case _: Reference | _: WRef | _: UIntLiteral | _: Mux | _: ValidIf =>
        case ex: SubAccess => validSubexp(info, mname)(ex.expr)
        case ex: WSubAccess => validSubexp(info, mname)(ex.expr)
        case ex => ex foreach validSubexp(info, mname)
      }
      e foreach checkHighFormW(info, mname + "/" + e.serialize)
      e foreach checkHighFormT(info, mname + "/" + e.serialize)
      e foreach checkHighFormE(info, mname, names)
    }

    def checkName(info: Info, mname: String, names: NameSet)(name: String): Unit = {
      if (names(name))
        errors.append(new NotUniqueException(info, mname, name))
      names += name
    }

    def checkInstance(info: Info, child: String, parent: String): Unit = {
      if (!moduleNames(child))
        errors.append(new ModuleNotDefinedException(info, parent, child))
      // Check to see if a recursive module instantiation has occured
      val childToParent = moduleGraph add (parent, child)
      if (childToParent.nonEmpty)
        errors.append(new InstanceLoop(info, parent, childToParent mkString "->"))
    }

    def checkHighFormS(minfo: Info, mname: String, names: NameSet)(s: Statement): Unit = {
      val info = get_info(s) match {case NoInfo => minfo case x => x}
      s foreach checkName(info, mname, names)
      s match {
        case DefRegister(info, name, tpe, _, reset, init) =>
          if (hasFlip(tpe))
            errors.append(new RegWithFlipException(info, mname, name))
        case sx: DefMemory =>
          if (sx.readLatency < 0 || sx.writeLatency <= 0)
            errors.append(new IllegalMemLatencyException(info, mname, sx.name))
          if (hasFlip(sx.dataType))
            errors.append(new MemWithFlipException(info, mname, sx.name))
          if (sx.depth <= 0)
            errors.append(new NegMemSizeException(info, mname))
        case sx: DefInstance => checkInstance(info, mname, sx.module)
        case sx: WDefInstance => checkInstance(info, mname, sx.module)
        case sx: Connect => checkValidLoc(info, mname, sx.loc)
        case sx: PartialConnect => checkValidLoc(info, mname, sx.loc)
        case sx: Print => checkFstring(info, mname, sx.string, sx.args.length)
        case _: CDefMemory | _: CDefMPort => errorOnChirrtl(info, mname, s).foreach { e => errors.append(e) }
        case sx => // Do Nothing
      }
      s foreach checkHighFormT(info, mname)
      s foreach checkHighFormE(info, mname, names)
      s foreach checkHighFormS(minfo, mname, names)
    }

    def checkHighFormP(mname: String, names: NameSet)(p: Port): Unit = {
      if (names(p.name))
        errors.append(new NotUniqueException(NoInfo, mname, p.name))
      names += p.name
      checkHighFormT(p.info, mname)(p.tpe)
    }

    // Search for ResetType Ports of direction
    def findBadResetTypePorts(m: DefModule, dir: Direction): Seq[(Port, Expression)] = {
      val bad = to_flow(dir)
      for {
        port <- m.ports
        ref = WRef(port).copy(flow = to_flow(port.direction))
        expr <- create_exps(ref)
        if ((expr.tpe == ResetType) && (flow(expr) == bad))
      } yield (port, expr)
    }

    def checkHighFormM(m: DefModule): Unit = {
      val names = new NameSet
      m foreach checkHighFormP(m.name, names)
      m foreach checkHighFormS(m.info, m.name, names)
      m match {
        case _: Module =>
        case ext: ExtModule =>
          for ((port, expr) <- findBadResetTypePorts(ext, Output)) {
            errors.append(new ResetExtModuleOutputException(port.info, ext.name, expr))
          }
      }
    }

    c.modules foreach checkHighFormM
    c.modules.filter(_.name == c.main) match {
      case Seq(topMod) =>
        for ((port, expr) <- findBadResetTypePorts(topMod, Input)) {
          errors.append(new ResetInputException(port.info, topMod.name, expr))
        }
      case _ => errors.append(new NoTopModuleException(c.info, c.main))
    }
    errors.trigger()
    c
  }
}

object CheckHighForm extends Pass with CheckHighFormLike {
  class IllegalChirrtlMemException(info: Info, mname: String, name: String) extends PassException(
    s"$info: [module $mname] Memory $name has not been properly lowered from Chirrtl IR.")

  def errorOnChirrtl(info: Info, mname: String, s: Statement): Option[PassException] = {
    val memName = s match {
      case cm: CDefMemory => cm.name
      case cp: CDefMPort => cp.mem
    }
    Some(new IllegalChirrtlMemException(info, mname, memName))
  }
}
object CheckTypes extends Pass {

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

object CheckFlows extends Pass {
  type FlowMap = collection.mutable.HashMap[String, Flow]

  implicit def toStr(g: Flow): String = g match {
    case SourceFlow => "source"
    case SinkFlow => "sink"
    case UnknownFlow => "unknown"
    case DuplexFlow => "duplex"
  }

  class WrongFlow(info:Info, mname: String, expr: String, wrong: Flow, right: Flow) extends PassException(
    s"$info: [module $mname]  Expression $expr is used as a $wrong but can only be used as a $right.")

  def run (c:Circuit): Circuit = {
    val errors = new Errors()

    def get_flow(e: Expression, flows: FlowMap): Flow = e match {
      case (e: WRef) => flows(e.name)
      case (e: WSubIndex) => get_flow(e.expr, flows)
      case (e: WSubAccess) => get_flow(e.expr, flows)
      case (e: WSubField) => e.expr.tpe match {case t: BundleType =>
        val f = (t.fields find (_.name == e.name)).get
        times(get_flow(e.expr, flows), f.flip)
      }
      case _ => SourceFlow
    }

    def flip_q(t: Type): Boolean = {
      def flip_rec(t: Type, f: Orientation): Boolean = t match {
        case tx:BundleType => tx.fields exists (
          field => flip_rec(field.tpe, times(f, field.flip))
        )
        case tx: VectorType => flip_rec(tx.tpe, f)
        case tx => f == Flip
      }
      flip_rec(t, Default)
    }

    def check_flow(info:Info, mname: String, flows: FlowMap, desired: Flow)(e:Expression): Unit = {
      val flow = get_flow(e,flows)
      (flow, desired) match {
        case (SourceFlow, SinkFlow) =>
          errors.append(new WrongFlow(info, mname, e.serialize, desired, flow))
        case (SinkFlow, SourceFlow) => kind(e) match {
          case PortKind | InstanceKind if !flip_q(e.tpe) => // OK!
          case _ =>
            errors.append(new WrongFlow(info, mname, e.serialize, desired, flow))
        }
        case _ =>
      }
   }

    def check_flows_e (info:Info, mname: String, flows: FlowMap)(e:Expression): Unit = {
      e match {
        case e: Mux => e foreach check_flow(info, mname, flows, SourceFlow)
        case e: DoPrim => e.args foreach check_flow(info, mname, flows, SourceFlow)
        case _ =>
      }
      e foreach check_flows_e(info, mname, flows)
    }

    def check_flows_s(minfo: Info, mname: String, flows: FlowMap)(s: Statement): Unit = {
      val info = get_info(s) match { case NoInfo => minfo case x => x }
      s match {
        case (s: DefWire) => flows(s.name) = DuplexFlow
        case (s: DefRegister) => flows(s.name) = DuplexFlow
        case (s: DefMemory) => flows(s.name) = SourceFlow
        case (s: WDefInstance) => flows(s.name) = SourceFlow
        case (s: DefNode) =>
          check_flow(info, mname, flows, SourceFlow)(s.value)
          flows(s.name) = SourceFlow
        case (s: Connect) =>
          check_flow(info, mname, flows, SinkFlow)(s.loc)
          check_flow(info, mname, flows, SourceFlow)(s.expr)
        case (s: Print) =>
          s.args foreach check_flow(info, mname, flows, SourceFlow)
          check_flow(info, mname, flows, SourceFlow)(s.en)
          check_flow(info, mname, flows, SourceFlow)(s.clk)
        case (s: PartialConnect) =>
          check_flow(info, mname, flows, SinkFlow)(s.loc)
          check_flow(info, mname, flows, SourceFlow)(s.expr)
        case (s: Conditionally) =>
          check_flow(info, mname, flows, SourceFlow)(s.pred)
        case (s: Stop) =>
          check_flow(info, mname, flows, SourceFlow)(s.en)
          check_flow(info, mname, flows, SourceFlow)(s.clk)
        case _ =>
      }
      s foreach check_flows_e(info, mname, flows)
      s foreach check_flows_s(minfo, mname, flows)
    }

    for (m <- c.modules) {
      val flows = new FlowMap
      flows ++= (m.ports map (p => p.name -> to_flow(p.direction)))
      m foreach check_flows_s(m.info, m.name, flows)
    }
    errors.trigger()
    c
  }
}

@deprecated("Use 'CheckFlows'. This object will be removed in 1.3", "1.2")
object CheckGenders {

  implicit def toStr(g: Gender): String = g match {
    case MALE => "source"
    case FEMALE => "sink"
    case UNKNOWNGENDER => "unknown"
    case BIGENDER => "sourceOrSink"
  }

  def run(c: Circuit): Circuit = CheckFlows.run(c)

  @deprecated("Use 'CheckFlows.WrongFlow'. This class will be removed in 1.3", "1.2")
  class WrongGender(info:Info, mname: String, expr: String, wrong: Flow, right: Flow) extends PassException(
    s"$info: [module $mname]  Expression $expr is used as a $wrong but can only be used as a $right.")
}
