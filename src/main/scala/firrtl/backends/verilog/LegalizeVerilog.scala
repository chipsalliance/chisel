// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.verilog

import firrtl.PrimOps._
import firrtl.Utils.{error, getGroundZero, zero, BoolType}
import firrtl.ir._
import firrtl.transforms.ConstantPropagation
import firrtl.{bitWidth, Dshlw, Transform}
import firrtl.Mappers._
import firrtl.passes.{Pass, SplitExpressions}

/** Rewrites some expressions for valid/better Verilog emission.
  * - solves shift right overflows by replacing the shift with 0 for UInts and MSB for SInts
  * - ensures that bit extracts on literals get resolved
  * - ensures that all negations are replaced with subtract from zero
  * - adds padding for rem and dshl which breaks firrtl width invariance, but is needed to match Verilog semantics
  */
object LegalizeVerilog extends Pass {

  override def prerequisites = firrtl.stage.Forms.LowForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Transform): Boolean = a match {
    case SplitExpressions => true // we generate pad and bits operations inline which need to be split up
    case _                => false
  }

  private def legalizeBitExtract(expr: DoPrim): Expression = {
    expr.args.head match {
      case _: UIntLiteral | _: SIntLiteral => ConstantPropagation.constPropBitExtract(expr)
      case _ => expr
    }
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

  import firrtl.passes.PadWidths.forceWidth
  private def getWidth(e: Expression): Int = bitWidth(e.tpe).toInt

  private def onExpr(expr: Expression): Expression = expr.map(onExpr) match {
    case prim: DoPrim =>
      prim.op match {
        case Shr                => ConstantPropagation.foldShiftRight(prim)
        case Bits | Head | Tail => legalizeBitExtract(prim)
        case Neg                => legalizeNeg(prim)
        case Rem                => prim.map(forceWidth(prim.args.map(getWidth).max))
        case Dshl               =>
          // special case as args aren't all same width
          prim.copy(op = Dshlw, args = Seq(forceWidth(getWidth(prim))(prim.args.head), prim.args(1)))
        case _ => prim
      }
    case e => e // respect pre-order traversal
  }

  def run(c: Circuit): Circuit = {
    def legalizeS(s: Statement): Statement = s.mapStmt(legalizeS).mapExpr(onExpr)
    c.copy(modules = c.modules.map(_.map(legalizeS)))
  }
}
