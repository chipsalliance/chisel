package firrtl
package passes

import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Mappers._

// Makes all implicit width extensions and truncations explicit
object PadWidths extends Pass {
  def name = "Pad Widths"
  private def width(t: Type): Int = bitWidth(t).toInt
  private def width(e: Expression): Int = width(e.tpe)
  // Returns an expression with the correct integer width
  private def fixup(i: Int)(e: Expression) = {
    def tx = e.tpe match {
      case t: UIntType => UIntType(IntWidth(i))
      case t: SIntType => SIntType(IntWidth(i))
      // default case should never be reached
    }
    width(e) match {
      case j if i > j => DoPrim(Pad, Seq(e), Seq(i), tx)
      case j if i < j =>
        val e2 = DoPrim(Bits, Seq(e), Seq(i - 1, 0), UIntType(IntWidth(i)))
        // Bit Select always returns UInt, cast if selecting from SInt
        e.tpe match {
          case UIntType(_) => e2
          case SIntType(_) => DoPrim(AsSInt, Seq(e2), Seq.empty, SIntType(IntWidth(i)))
        }
      case _ => e
    }
  }

  // Recursive, updates expression so children exp's have correct widths
  private def onExp(e: Expression): Expression = e map onExp match {
    case Mux(cond, tval, fval, tpe) =>
      Mux(cond, fixup(width(tpe))(tval), fixup(width(tpe))(fval), tpe)
    case e: ValidIf => e copy (value = fixup(width(e.tpe))(e.value))
    case e: DoPrim => e.op match {
      case Lt | Leq | Gt | Geq | Eq | Neq | Not | And | Or | Xor |
           Add | Sub | Mul | Div | Rem | Shr =>
        // sensitive ops
        e map fixup((e.args map width foldLeft 0)(math.max))
      case Dshl =>
        // special case as args aren't all same width
        e copy (op = Dshlw, args = Seq(fixup(width(e.tpe))(e.args.head), e.args(1)))
      case Shl =>
        // special case as arg should be same width as result
        e copy (op = Shlw, args = Seq(fixup(width(e.tpe))(e.args.head)))
      case _ => e
    }
    case e => e
  }

  // Recursive. Fixes assignments and register initialization widths
  private def onStmt(s: Statement): Statement = s map onExp match {
    case s: Connect =>
      s copy (expr = fixup(width(s.loc))(s.expr))
    case s: DefRegister =>
      s copy (init = fixup(width(s.tpe))(s.init))
    case s => s map onStmt
  }

  def run(c: Circuit): Circuit = c copy (modules = c.modules map (_ map onStmt))
}
