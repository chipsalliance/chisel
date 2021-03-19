// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.PrimOps._
import firrtl.Utils.{error, getGroundZero, zero, BoolType}
import firrtl.ir._
import firrtl.options.Dependency
import firrtl.transforms.ConstantPropagation
import firrtl.{bitWidth, getWidth, Transform}
import firrtl.Mappers._

// Replace shr by amount >= arg width with 0 for UInts and MSB for SInts
// TODO replace UInt with zero-width wire instead
object Legalize extends Pass {

  override def prerequisites = firrtl.stage.Forms.MidForm :+ Dependency(LowerTypes)

  override def optionalPrerequisites = Seq.empty

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform) = false

  private def legalizeShiftRight(e: DoPrim): Expression = {
    require(e.op == Shr)
    e.args.head match {
      case _: UIntLiteral | _: SIntLiteral => ConstantPropagation.foldShiftRight(e)
      case _ =>
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
  }
  private def legalizeBitExtract(expr: DoPrim): Expression = {
    expr.args.head match {
      case _: UIntLiteral | _: SIntLiteral => ConstantPropagation.constPropBitExtract(expr)
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
  // Convert `-x` to `0 - x`
  private def legalizeNeg(expr: DoPrim): Expression = {
    val arg = expr.args.head
    arg.tpe match {
      case tpe: SIntType =>
        val zero = getGroundZero(tpe)
        DoPrim(Sub, Seq(zero, arg), Nil, expr.tpe)
      case tpe: UIntType =>
        val zero = getGroundZero(tpe)
        val sub = DoPrim(Sub, Seq(zero, arg), Nil, UIntType(tpe.width + IntWidth(1)))
        DoPrim(AsSInt, Seq(sub), Nil, expr.tpe)
    }
  }
  private def legalizeConnect(c: Connect): Statement = {
    val t = c.loc.tpe
    val w = bitWidth(t)
    if (w >= bitWidth(c.expr.tpe)) {
      c
    } else {
      val bits = DoPrim(Bits, Seq(c.expr), Seq(w - 1, 0), UIntType(IntWidth(w)))
      val expr = t match {
        case UIntType(_)               => bits
        case SIntType(_)               => DoPrim(AsSInt, Seq(bits), Seq(), SIntType(IntWidth(w)))
        case FixedType(_, IntWidth(p)) => DoPrim(AsFixedPoint, Seq(bits), Seq(p), t)
      }
      Connect(c.info, c.loc, expr)
    }
  }
  def run(c: Circuit): Circuit = {
    def legalizeE(expr: Expression): Expression = expr.map(legalizeE) match {
      case prim: DoPrim =>
        prim.op match {
          case Shr                => legalizeShiftRight(prim)
          case Pad                => legalizePad(prim)
          case Bits | Head | Tail => legalizeBitExtract(prim)
          case Neg                => legalizeNeg(prim)
          case _                  => prim
        }
      case e => e // respect pre-order traversal
    }
    def legalizeS(s: Statement): Statement = {
      val legalizedStmt = s match {
        case c: Connect => legalizeConnect(c)
        case _ => s
      }
      legalizedStmt.map(legalizeS).map(legalizeE)
    }
    c.copy(modules = c.modules.map(_.map(legalizeS)))
  }
}
