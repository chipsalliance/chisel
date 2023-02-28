// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.verilog

import firrtl.PrimOps._
import firrtl.Utils.{error, getGroundZero, zero, BoolType}
import firrtl.ir._
import firrtl.transforms.ConstantPropagation
import firrtl.{bitWidth, Dshlw, Transform}
import firrtl.Mappers._
import firrtl.passes.{Pass, SplitExpressions}
import firrtl.passes.PadWidths.forceWidth

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

  private def getWidth(e: Expression): Int = bitWidth(e.tpe).toInt

  /* Verilog has the width of (a % b) = Max(W(a), W(b))
   * FIRRTL has the width of (a % b) = Min(W(a), W(b)), which makes more sense,
   * but nevertheless is a problem when emitting verilog
   *
   * This function pads the arguments to be the same [max] width (to avoid lint issues)
   * and then performs a bit extraction back down to the correct [min] width
   */
  private def legalizeRem(e: Expression): Expression = e match {
    case rem @ DoPrim(Rem, Seq(a, b), _, tpe) =>
      val awidth = getWidth(a)
      val bwidth = getWidth(b)
      // Do nothing if the widths are the same
      if (awidth == bwidth) {
        rem
      } else {
        // First pad the arguments to fix lint warnings because Verilog width is max of arguments
        val maxWidth = awidth.max(bwidth)
        val newType = tpe.mapWidth(_ => IntWidth(maxWidth))
        val paddedRem =
          rem
            .map(forceWidth(maxWidth)(_)) // Pad the input arguments
            .mapType(_ => newType) // Also make the width for this op correct
        // Second, bit extract back down to the min width of original arguments to match FIRRTL semantics
        val minWidth = awidth.min(bwidth)
        forceWidth(minWidth)(paddedRem)
      }
    case _ => e
  }

  private def onExpr(expr: Expression): Expression = expr.map(onExpr) match {
    case prim: DoPrim =>
      prim.op match {
        case Shr                => ConstantPropagation.foldShiftRight(prim)
        case Bits | Head | Tail => legalizeBitExtract(prim)
        case Neg                => legalizeNeg(prim)
        case Rem                => legalizeRem(prim)
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
