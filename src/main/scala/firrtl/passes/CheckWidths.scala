// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Mappers._
import firrtl.Utils._

object CheckWidths extends Pass {
  /** The maximum allowed width for any circuit element */
  val MaxWidth = 1000000
  val DshlMaxWidth = ceilLog2(MaxWidth + 1)
  class UninferredWidth (info: Info, mname: String) extends PassException(
    s"$info : [module $mname]  Uninferred width.")
  class WidthTooSmall(info: Info, mname: String, b: BigInt) extends PassException(
    s"$info : [module $mname]  Width too small for constant $b.")
  class WidthTooBig(info: Info, mname: String, b: BigInt) extends PassException(
    s"$info : [module $mname]  Width $b greater than max allowed width of $MaxWidth bits")
  class DshlTooBig(info: Info, mname: String) extends PassException(
    s"$info : [module $mname]  Width of dshl shift amount cannot be larger than $DshlMaxWidth bits.")
  class NegWidthException(info:Info, mname: String) extends PassException(
    s"$info: [module $mname] Width cannot be negative or zero.")
  class BitsWidthException(info: Info, mname: String, hi: BigInt, width: BigInt, exp: String) extends PassException(
    s"$info: [module $mname] High bit $hi in bits operator is larger than input width $width in $exp.")
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
        case IntWidth(width) if width >= MaxWidth =>
          errors.append(new WidthTooBig(info, mname, width))
        case w: IntWidth if w.width >= 0 =>
        case _: IntWidth =>
          errors append new NegWidthException(info, mname)
        case _ =>
          errors append new UninferredWidth(info, mname)
      }
      w
    }

    def hasWidth(tpe: Type): Boolean = tpe match {
      case GroundType(IntWidth(w)) => true
      case GroundType(_) => false
      case _ => throwInternalError(Some(s"hasWidth - $tpe"))
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
        case DoPrim(Bits, Seq(a), Seq(hi, lo), _) if (hasWidth(a.tpe) && bitWidth(a.tpe) <= hi) =>
          errors append new BitsWidthException(info, mname, hi, bitWidth(a.tpe), e.serialize)
        case DoPrim(Head, Seq(a), Seq(n), _) if (hasWidth(a.tpe) && bitWidth(a.tpe) < n) =>
          errors append new HeadWidthException(info, mname, n, bitWidth(a.tpe))
        case DoPrim(Tail, Seq(a), Seq(n), _) if (hasWidth(a.tpe) && bitWidth(a.tpe) <= n) =>
          errors append new TailWidthException(info, mname, n, bitWidth(a.tpe))
        case DoPrim(Dshl, Seq(a, b), _, _) if (hasWidth(a.tpe) && bitWidth(b.tpe) >= DshlMaxWidth) =>
          errors append new DshlTooBig(info, mname)
        case _ =>
      }
      //e map check_width_t(info, mname) map check_width_e(info, mname)
      e map check_width_e(info, mname)
    }


    def check_width_s(minfo: Info, mname: String)(s: Statement): Statement = {
      val info = get_info(s) match { case NoInfo => minfo case x => x }
      s map check_width_e(info, mname) map check_width_s(info, mname) map check_width_t(info, mname) match {
        case Attach(infox, exprs) =>
          exprs.tail.foreach ( e =>
            if (bitWidth(e.tpe) != bitWidth(exprs.head.tpe))
              errors.append(new AttachWidthsNotEqual(infox, mname, e.serialize, exprs.head.serialize))
          )
          s
        case sx: DefRegister =>
          sx.reset.tpe match {
            case UIntType(IntWidth(w)) if w == 1 =>
            case _ => errors.append(new CheckTypes.IllegalResetType(info, mname, sx.name))
          }
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
