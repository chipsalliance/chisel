package firrtl.passes

import firrtl.PrimOps._
import firrtl.Utils.{BoolType, error, zero}
import firrtl.ir._
import firrtl.options.{PreservesAll, Dependency}
import firrtl.transforms.ConstantPropagation
import firrtl.{Transform, bitWidth}
import firrtl.Mappers._

// Replace shr by amount >= arg width with 0 for UInts and MSB for SInts
// TODO replace UInt with zero-width wire instead
object Legalize extends Pass with PreservesAll[Transform] {

  override def prerequisites = firrtl.stage.Forms.MidForm :+ Dependency(LowerTypes)

  override def optionalPrerequisites = Seq.empty

  override def optionalPrerequisiteOf = Seq.empty

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
        case FixedType(_, IntWidth(p)) => DoPrim(AsFixedPoint, Seq(bits), Seq(p), t)
      }
      Connect(c.info, c.loc, expr)
    }
  }
  def run (c: Circuit): Circuit = {
    def legalizeE(expr: Expression): Expression = expr map legalizeE match {
      case prim: DoPrim => prim.op match {
        case Shr => legalizeShiftRight(prim)
        case Pad => legalizePad(prim)
        case Bits | Head | Tail => legalizeBitExtract(prim)
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
