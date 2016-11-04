// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Utils._
import firrtl.Mappers._
import firrtl.WrappedType._

object CheckHighForm extends Pass {
  def name = "High Form Check"
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

  // TODO FIXME
  // - Do we need to check for uniquness on port names?
  def run(c: Circuit): Circuit = {
    val errors = new Errors()
    val moduleGraph = new ModuleGraph
    val moduleNames = (c.modules map (_.name)).toSet

    def checkHighFormPrimop(info: Info, mname: String, e: DoPrim) {
      def correctNum(ne: Option[Int], nc: Int) {
        ne match {
          case Some(i) if e.args.length != i =>
            errors append new IncorrectNumArgsException(info, mname, e.op.toString, i)
          case _ => // Do Nothing
        }
        if (e.consts.length != nc)
          errors append new IncorrectNumConstsException(info, mname, e.op.toString, nc)
      }

      e.op match {
        case Add | Sub | Mul | Div | Rem | Lt | Leq | Gt | Geq |
             Eq | Neq | Dshl | Dshr | And | Or | Xor | Cat =>
          correctNum(Option(2), 0)
        case AsUInt | AsSInt | AsClock | Cvt | Neq | Not =>
          correctNum(Option(1), 0)
        case AsFixedPoint | Pad | Shl | Shr | Head | Tail | BPShl | BPShr | BPSet =>
          correctNum(Option(1), 1)
        case Bits =>
          correctNum(Option(1), 2)
        case Andr | Orr | Xorr =>
          correctNum(None,0)
      }
    }

    def checkFstring(info: Info, mname: String, s: StringLit, i: Int) {
      val validFormats = "bdxc"
      val (percent, npercents) = (s.array foldLeft (false, 0)){
        case ((percentx, n), b) if percentx && (validFormats contains b) =>
          (false, n + 1)
        case ((percentx, n), b) if percentx && b != '%' =>
          errors append new BadPrintfException(info, mname, b.toChar)
          (false, n)
        case ((percentx, n), b) =>
          (if (b == '%') !percentx else false /* %% -> percentx = false */, n)
      }
      if (percent) errors append new BadPrintfTrailingException(info, mname)
      if (npercents != i) errors append new BadPrintfIncorrectNumException(info, mname)
    }

    def checkValidLoc(info: Info, mname: String, e: Expression) = e match {
      case _: UIntLiteral | _: SIntLiteral | _: DoPrim =>
        errors append new InvalidLOCException(info, mname)
      case _ => // Do Nothing
    }

    def checkHighFormW(info: Info, mname: String)(w: Width): Width = {
      w match {
        case wx: IntWidth if wx.width < 0 =>
          errors append new NegWidthException(info, mname)
        case wx => // Do nothing
      }
      w
    }

    def checkHighFormT(info: Info, mname: String)(t: Type): Type =
      t map checkHighFormT(info, mname) match {
        case tx: VectorType if tx.size < 0 => 
          errors append new NegVecSizeException(info, mname)
          t
        case _ => t map checkHighFormW(info, mname)
      }

    def validSubexp(info: Info, mname: String)(e: Expression): Expression = {
      e match {
        case _: WRef | _: WSubField | _: WSubIndex | _: WSubAccess | _: Mux | _: ValidIf => // No error
        case _ => errors append new InvalidAccessException(info, mname)
      }
      e
    }

    def checkHighFormE(info: Info, mname: String, names: NameSet)(e: Expression): Expression = {
      e match {
        case ex: WRef if !names(ex.name) =>
          errors append new UndeclaredReferenceException(info, mname, ex.name)
        case ex: UIntLiteral if ex.value < 0 =>
          errors append new NegUIntException(info, mname)
        case ex: DoPrim => checkHighFormPrimop(info, mname, ex)
        case _: WRef | _: UIntLiteral | _: Mux | _: ValidIf =>
        case ex: WSubAccess => validSubexp(info, mname)(ex.exp)
        case ex => ex map validSubexp(info, mname)
      }
      (e map checkHighFormW(info, mname)
         map checkHighFormT(info, mname)
         map checkHighFormE(info, mname, names))
    }

    def checkName(info: Info, mname: String, names: NameSet)(name: String): String = {
      if (names(name))
        errors append new NotUniqueException(info, mname, name)
      names += name
      name
    }

    def checkHighFormS(minfo: Info, mname: String, names: NameSet)(s: Statement): Statement = {
      val info = get_info(s) match {case NoInfo => minfo case x => x}
      s map checkName(info, mname, names) match {
        case sx: DefMemory =>
          if (hasFlip(sx.dataType))
            errors append new MemWithFlipException(info, mname, sx.name)
          if (sx.depth <= 0)
            errors append new NegMemSizeException(info, mname)
        case sx: WDefInstance =>
          if (!moduleNames(sx.module))
            errors append new ModuleNotDefinedException(info, mname, sx.module)
          // Check to see if a recursive module instantiation has occured
          val childToParent = moduleGraph add (mname, sx.module)
          if (childToParent.nonEmpty)
            errors append new InstanceLoop(info, mname, childToParent mkString "->")
        case sx: Connect => checkValidLoc(info, mname, sx.loc)
        case sx: PartialConnect => checkValidLoc(info, mname, sx.loc)
        case sx: Print => checkFstring(info, mname, sx.string, sx.args.length)
        case sx => // Do Nothing
      }
      (s map checkHighFormT(info, mname)
         map checkHighFormE(info, mname, names)
         map checkHighFormS(minfo, mname, names))
    }

    def checkHighFormP(mname: String, names: NameSet)(p: Port): Port = {
      names += p.name
      (p.tpe map checkHighFormT(p.info, mname)
             map checkHighFormW(p.info, mname))
      p
    }

    def checkHighFormM(m: DefModule) {
      val names = new NameSet
      (m map checkHighFormP(m.name, names)
         map checkHighFormS(m.info, m.name, names))
    }
    
    c.modules foreach checkHighFormM
    c.modules count (_.name == c.main) match {
      case 1 =>
      case _ => errors append new NoTopModuleException(c.info, c.main)
    }
    errors.trigger()
    c
  }
}

object CheckTypes extends Pass {
  def name = "Check Types"

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
  class InvalidConnect(info: Info, mname: String, lhs: String, rhs: String) extends PassException(
    s"$info: [module $mname]  Type mismatch. Cannot connect $lhs to $rhs.")
  class InvalidRegInit(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Type of init must match type of DefRegister.")
  class PrintfArgNotGround(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Printf arguments must be either UIntType or SIntType.")
  class ReqClk(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Requires a clock typed signal.")
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
  class OpNotAnalog(info: Info, mname: String, exp: String) extends PassException(
    s"$info: [module $mname]  Attach requires all arguments to be Analog type: $exp.")
  class NodePassiveType(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Node must be a passive type.")
  class MuxSameType(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Must mux between equivalent types.")
  class MuxPassiveTypes(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Must mux between passive types.")
  class MuxCondUInt(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  A mux condition must be of type UInt.")
  class ValidIfPassiveTypes(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  Must validif a passive type.")
  class ValidIfCondUInt(info: Info, mname: String) extends PassException(
    s"$info: [module $mname]  A validif condition must be of type UInt.")
  class IllegalAnalogDeclaration(info: Info, mname: String, decName: String) extends PassException(
    s"$info: [module $mname]  Cannot declare a reg, node, or memory with an Analog type: $decName.")
  class IllegalAttachSource(info: Info, mname: String, sourceName: String) extends PassException(
    s"$info: [module $mname]  Attach source must be a wire or port with an analog type: $sourceName.")
  class IllegalAttachExp(info: Info, mname: String, expName: String) extends PassException(
    s"$info: [module $mname]  Attach expression must be an instance: $expName.")

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
    def check_types_primop(info: Info, mname: String, e: DoPrim) {
      def all_same_type (ls:Seq[Expression]) {
        if (ls exists (x => wt(ls.head.tpe) != wt(e.tpe)))
          errors append new OpNotAllSameType(info, mname, e.op.serialize)
      }
      def allUSC(ls: Seq[Expression]) {
        val error = ls.foldLeft(false)((error, x) => x.tpe match {
          case (_: UIntType| _: SIntType| ClockType) => error
          case _ => true
        })
        if (error) errors.append(new OpNotGround(info, mname, e.op.serialize))
      }
      def allUSF(ls: Seq[Expression]) {
        val error = ls.foldLeft(false)((error, x) => x.tpe match {
          case (_: UIntType| _: SIntType| _: FixedType) => error
          case _ => true
        })
        if (error) errors.append(new OpNotGround(info, mname, e.op.serialize))
      }
      def allUS(ls: Seq[Expression]) {
        if (ls exists (x => x.tpe match {
          case _: UIntType | _: SIntType => false
          case _ => true
        })) errors append new OpNotGround(info, mname, e.op.serialize)
      }
      def allF(ls: Seq[Expression]) {
        val error = ls.foldLeft(false)((error, x) => x.tpe match {
          case _:FixedType => error
          case _ => true
        })
        if (error) errors.append(new OpNotGround(info, mname, e.op.serialize))
      }
      def strictFix(ls: Seq[Expression]) = 
        ls.filter(!_.tpe.isInstanceOf[FixedType]).size match {
          case 0 => 
          case x if(x == ls.size) =>
          case x => errors.append(new OpNoMixFix(info, mname, e.op.serialize))
        }
      def all_uint (ls: Seq[Expression]) {
        if (ls exists (x => x.tpe match {
          case _: UIntType => false
          case _ => true
        })) errors append new OpNotAllUInt(info, mname, e.op.serialize)
      }
      def is_uint (x:Expression) {
        if (x.tpe match {
          case _: UIntType => false
          case _ => true
        }) errors append new OpNotUInt(info, mname, e.op.serialize, x.serialize)
      }
      e.op match {
        case AsUInt | AsSInt | AsFixedPoint =>
        case AsClock => allUSC(e.args)
        case Dshl => is_uint(e.args(1)); allUSF(e.args)
        case Dshr => is_uint(e.args(1)); allUSF(e.args)
        case Add | Sub | Mul | Lt | Leq | Gt | Geq | Eq | Neq => allUSF(e.args); strictFix(e.args)
        case Pad | Shl | Shr | Cat | Bits | Head | Tail => allUSF(e.args)
        case BPShl | BPShr | BPSet => allF(e.args)
        case _ => allUS(e.args)
      }
    }

    def check_types_e(info:Info, mname: String)(e: Expression): Expression = {
      e match {
        case (e: WSubField) => e.exp.tpe match {
          case (t: BundleType) => t.fields find (_.name == e.name) match {
            case Some(_) =>
            case None => errors append new SubfieldNotInBundle(info, mname, e.name)
          }
          case _ => errors append new SubfieldOnNonBundle(info, mname, e.name)
        }
        case (e: WSubIndex) => e.exp.tpe match {
          case (t: VectorType) if e.value < t.size =>
          case (t: VectorType) =>
            errors append new IndexTooLarge(info, mname, e.value)
          case _ =>
            errors append new IndexOnNonVector(info, mname)
        }
        case (e: WSubAccess) =>
          e.exp.tpe match {
            case _: VectorType =>
            case _ => errors append new IndexOnNonVector(info, mname)
          }
          e.index.tpe match {
            case _: UIntType =>
            case _ => errors append new AccessIndexNotUInt(info, mname)
          }
        case (e: DoPrim) => check_types_primop(info, mname, e)
        case (e: Mux) =>
          if (wt(e.tval.tpe) != wt(e.fval.tpe))
            errors append new MuxSameType(info, mname)
          if (!passive(e.tpe))
            errors append new MuxPassiveTypes(info, mname)
          e.cond.tpe match {
            case _: UIntType =>
            case _ => errors append new MuxCondUInt(info, mname)
          }
        case (e: ValidIf) =>
          if (!passive(e.tpe))
            errors append new ValidIfPassiveTypes(info, mname)
          e.cond.tpe match {
            case _: UIntType =>
            case _ => errors append new ValidIfCondUInt(info, mname)
          }
        case _ =>
      }
      e map check_types_e(info, mname)
    }

    def bulk_equals(t1: Type, t2: Type, flip1: Orientation, flip2: Orientation): Boolean = {
      //;println_all(["Inside with t1:" t1 ",t2:" t2 ",f1:" flip1 ",f2:" flip2])
      (t1, t2) match {
        case (ClockType, ClockType) => flip1 == flip2
        case (_: UIntType, _: UIntType) => flip1 == flip2
        case (_: SIntType, _: SIntType) => flip1 == flip2
        case (_: FixedType, _: FixedType) => flip1 == flip2
        case (_: AnalogType, _: AnalogType) => false
        case (t1: BundleType, t2: BundleType) =>
          val t1_fields = (t1.fields foldLeft Map[String, (Type, Orientation)]())(
            (map, f1) => map + (f1.name -> (f1.tpe, f1.flip)))
          t2.fields forall (f2 =>
            t1_fields get f2.name match {
              case None => true
              case Some((f1_tpe, f1_flip)) =>
                bulk_equals(f1_tpe, f2.tpe, times(flip1, f1_flip), times(flip2, f2.flip))
            }
          )
        case (t1: VectorType, t2: VectorType) =>
          bulk_equals(t1.tpe, t2.tpe, flip1, flip2)
        case (t1, t2) => false
      }
    }

    def check_types_s(minfo: Info, mname: String)(s: Statement): Statement = {
      val info = get_info(s) match { case NoInfo => minfo case x => x }
      s match {
        case sx: Connect if wt(sx.loc.tpe) != wt(sx.expr.tpe) =>
          errors append new InvalidConnect(info, mname, sx.loc.serialize, sx.expr.serialize)
        case sx: PartialConnect if !bulk_equals(sx.loc.tpe, sx.expr.tpe, Default, Default) =>
          errors append new InvalidConnect(info, mname, sx.loc.serialize, sx.expr.serialize)
        case sx: DefRegister => sx.tpe match {
          case AnalogType(w) => errors append new IllegalAnalogDeclaration(info, mname, sx.name)
          case t if wt(sx.tpe) != wt(sx.init.tpe) => errors append new InvalidRegInit(info, mname)
          case t =>
        }
        case sx: Conditionally if wt(sx.pred.tpe) != wt(ut) =>
          errors append new PredNotUInt(info, mname)
        case sx: DefNode => sx.value.tpe match {
          case AnalogType(w) => errors append new IllegalAnalogDeclaration(info, mname, sx.name)
          case t if !passive(sx.value.tpe) => errors append new NodePassiveType(info, mname)
          case t =>
        }
        case sx: Attach =>
          (sx.source.tpe, kind(sx.source)) match {
            case (AnalogType(w), PortKind | WireKind)  =>
            case _ => errors append new IllegalAttachSource(info, mname, sx.source.serialize)
          }
          sx.exprs foreach { e =>
            e.tpe match {
              case _: AnalogType =>
              case _ => errors append new OpNotAnalog(info, mname, e.serialize)
            }
            kind(e) match {
              case InstanceKind =>
              case _ =>  errors append new IllegalAttachExp(info, mname, e.serialize)
            }
          }
        case sx: Stop =>
          if (wt(sx.clk.tpe) != wt(ClockType)) errors append new ReqClk(info, mname)
          if (wt(sx.en.tpe) != wt(ut)) errors append new EnNotUInt(info, mname)
        case sx: Print =>
          if (sx.args exists (x => wt(x.tpe) != wt(ut) && wt(x.tpe) != wt(st)))
            errors append new PrintfArgNotGround(info, mname)
          if (wt(sx.clk.tpe) != wt(ClockType)) errors append new ReqClk(info, mname)
          if (wt(sx.en.tpe) != wt(ut)) errors append new EnNotUInt(info, mname)
        case sx: DefMemory => sx.dataType match {
          case AnalogType(w) => errors append new IllegalAnalogDeclaration(info, mname, sx.name)
          case t =>
        }
        case _ =>
      }
      s map check_types_e(info, mname) map check_types_s(info, mname)
    }

    c.modules foreach (m => m map check_types_s(m.info, m.name))
    errors.trigger()
    c
  }
}

object CheckGenders extends Pass {
  def name = "Check Genders"
  type GenderMap = collection.mutable.HashMap[String, Gender]

  implicit def toStr(g: Gender): String = g match {
    case MALE => "source"
    case FEMALE => "sink"
    case UNKNOWNGENDER => "unknown"
    case BIGENDER => "sourceOrSink"
  }
   
  class WrongGender(info:Info, mname: String, expr: String, wrong: Gender, right: Gender) extends PassException(
    s"$info: [module $mname]  Expression $expr is used as a $wrong but can only be used as a $right.")
   
  def run (c:Circuit): Circuit = {
    val errors = new Errors()

    def get_gender(e: Expression, genders: GenderMap): Gender = e match {
      case (e: WRef) => genders(e.name)
      case (e: WSubIndex) => get_gender(e.exp, genders)
      case (e: WSubAccess) => get_gender(e.exp, genders)
      case (e: WSubField) => e.exp.tpe match {case t: BundleType =>
        val f = (t.fields find (_.name == e.name)).get
        times(get_gender(e.exp, genders), f.flip)
      }
      case _ => MALE
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

    def check_gender(info:Info, mname: String, genders: GenderMap, desired: Gender)(e:Expression): Expression = {
      val gender = get_gender(e,genders)
      (gender, desired) match {
        case (MALE, FEMALE) =>
          errors append new WrongGender(info, mname, e.serialize, desired, gender)
        case (FEMALE, MALE) => kind(e) match {
          case PortKind | InstanceKind if !flip_q(e.tpe) => // OK!
          case _ =>
            errors append new WrongGender(info, mname, e.serialize, desired, gender)
        }
        case _ =>
      }
      e
   }
   
    def check_genders_e (info:Info, mname: String, genders: GenderMap)(e:Expression): Expression = {
      e match {
        case e: Mux => e map check_gender(info, mname, genders, MALE)
        case e: DoPrim => e.args map check_gender(info, mname, genders, MALE)
        case _ =>
      }
      e map check_genders_e(info, mname, genders)
    }
        
    def check_genders_s(minfo: Info, mname: String, genders: GenderMap)(s: Statement): Statement = {
      val info = get_info(s) match { case NoInfo => minfo case x => x }
      s match {
        case (s: DefWire) => genders(s.name) = BIGENDER
        case (s: DefRegister) => genders(s.name) = BIGENDER
        case (s: DefMemory) => genders(s.name) = MALE
        case (s: WDefInstance) => genders(s.name) = MALE
        case (s: DefNode) =>
          check_gender(info, mname, genders, MALE)(s.value)
          genders(s.name) = MALE
        case (s: Connect) =>
          check_gender(info, mname, genders, FEMALE)(s.loc)
          check_gender(info, mname, genders, MALE)(s.expr)
        case (s: Print) =>
          s.args map check_gender(info, mname, genders, MALE)
          check_gender(info, mname, genders, MALE)(s.en)
          check_gender(info, mname, genders, MALE)(s.clk)
        case (s: PartialConnect) =>
          check_gender(info, mname, genders, FEMALE)(s.loc)
          check_gender(info, mname, genders, MALE)(s.expr)
        case (s: Conditionally) =>
          check_gender(info, mname, genders, MALE)(s.pred)
        case (s: Stop) =>
          check_gender(info, mname, genders, MALE)(s.en)
          check_gender(info, mname, genders, MALE)(s.clk)
        case _ =>
      }
      s map check_genders_e(info, mname, genders) map check_genders_s(minfo, mname, genders)
    }
   
    for (m <- c.modules) {
      val genders = new GenderMap
      genders ++= (m.ports map (p => p.name -> to_gender(p.direction)))
      m map check_genders_s(m.info, m.name, genders)
    }
    errors.trigger()
    c
  }
}

object CheckWidths extends Pass {
  def name = "Width Check"
  class UninferredWidth (info: Info, mname: String) extends PassException(
    s"$info : [module $mname]  Uninferred width.")
  class WidthTooSmall(info: Info, mname: String, b: BigInt) extends PassException(
    s"$info : [module $mname]  Width too small for constant ${serialize(b)}.")
  class WidthTooBig(info: Info, mname: String) extends PassException(
    s"$info : [module $mname]  Width of dshl shift amount cannot be larger than 31 bits.")
  class NegWidthException(info:Info, mname: String) extends PassException(
    s"$info: [module $mname] Width cannot be negative or zero.")
  class BitsWidthException(info: Info, mname: String, hi: BigInt, width: BigInt) extends PassException(
    s"$info: [module $mname] High bit $hi in bits operator is larger than input width $width.")
  class HeadWidthException(info: Info, mname: String, n: BigInt, width: BigInt) extends PassException(
    s"$info: [module $mname] Parameter $n in head operator is larger than input width $width.")
  class TailWidthException(info: Info, mname: String, n: BigInt, width: BigInt) extends PassException(
    s"$info: [module $mname] Parameter $n in tail operator is larger than input width $width.")
  class AttachWidthsNotEqual(info: Info, mname: String, eName: String, source: String) extends PassException(
    s"$info: [module $mname] Attach source $source and expression $eName must have identical widths.")

  def run(c: Circuit): Circuit = {
    val errors = new Errors()

    def check_width_w(info: Info, mname: String)(w: Width): Width = {
      w match {
        case w: IntWidth if w.width >= 0 =>
        case _: IntWidth =>
          errors append new NegWidthException(info, mname)
        case _ =>
          errors append new UninferredWidth(info, mname)
      }
      w
    }

    def check_width_t(info: Info, mname: String)(t: Type): Type =
      t map check_width_t(info, mname) map check_width_w(info, mname)

    def check_width_e(info: Info, mname: String)(e: Expression): Expression = {
      e match {
        case e: UIntLiteral => e.width match {
          case w: IntWidth if math.max(1, e.value.bitLength) > w.width =>
            errors append new WidthTooSmall(info, mname, e.value)
          case _ =>
        }
        case e: SIntLiteral => e.width match {
          case w: IntWidth if e.value.bitLength + 1 > w.width =>
            errors append new WidthTooSmall(info, mname, e.value)
          case _ =>
        }
        case DoPrim(Bits, Seq(a), Seq(hi, lo), _) if bitWidth(a.tpe) <= hi =>
          errors append new BitsWidthException(info, mname, hi, bitWidth(a.tpe))
        case DoPrim(Head, Seq(a), Seq(n), _) if bitWidth(a.tpe) < n =>
          errors append new HeadWidthException(info, mname, n, bitWidth(a.tpe))
        case DoPrim(Tail, Seq(a), Seq(n), _) if bitWidth(a.tpe) <= n =>
          errors append new TailWidthException(info, mname, n, bitWidth(a.tpe))
        case DoPrim(Dshl, Seq(a, b), _, _) if bitWidth(b.tpe) >= BigInt(32) =>
          errors append new WidthTooBig(info, mname)
        case _ =>
      }
      //e map check_width_t(info, mname) map check_width_e(info, mname)
      e map check_width_e(info, mname)
    }


    def check_width_s(minfo: Info, mname: String)(s: Statement): Statement = {
      val info = get_info(s) match { case NoInfo => minfo case x => x }
      s map check_width_e(info, mname) map check_width_s(info, mname) map check_width_t(info, mname) match {
        case Attach(infox, source, exprs) => 
          exprs foreach ( e =>
            if (bitWidth(e.tpe) != bitWidth(source.tpe))
              errors append new AttachWidthsNotEqual(infox, mname, e.serialize, source.serialize)
          )
          s
        case _ => s
      } 
    }

    def check_width_p(minfo: Info, mname: String)(p: Port): Port = p.copy(tpe =  check_width_t(p.info, mname)(p.tpe))

    def check_width_m(m: DefModule) {
      m map check_width_p(m.info, m.name) map check_width_s(m.info, m.name)
    }

    c.modules foreach check_width_m
    errors.trigger()
    c
  }
}
