// SPDX-License-Identifier: Apache-2.0

package firrtl
package passes

import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Mappers._
import firrtl.options.Dependency
import firrtl.transforms.ConstantPropagation

// Makes all implicit width extensions and truncations explicit
object PadWidths extends Pass {

  override def prerequisites = firrtl.stage.Forms.LowForm

  override def optionalPrerequisiteOf =
    Seq(Dependency(firrtl.passes.memlib.VerilogMemDelays), Dependency[SystemVerilogEmitter], Dependency[VerilogEmitter])

  override def invalidates(a: Transform): Boolean = a match {
    case SplitExpressions => true // we generate pad and bits operations inline which need to be split up
    case _                => false
  }

  /** Adds padding or a bit extract to ensure that the expression is of the with specified.
    * @note only works on UInt and SInt type expressions, other expressions will yield a match error
    */
  private[firrtl] def forceWidth(width: Int)(e: Expression): Expression = {
    val old = getWidth(e)
    if (width == old) { e }
    else if (width > old) {
      // padding retains the signedness
      val newType = e.tpe match {
        case _: UIntType => UIntType(IntWidth(width))
        case _: SIntType => SIntType(IntWidth(width))
        case other => throw new RuntimeException(s"forceWidth does not support expressions of type $other")
      }
      ConstantPropagation.constPropPad(DoPrim(Pad, Seq(e), Seq(width), newType))
    } else {
      val extract = DoPrim(Bits, Seq(e), Seq(width - 1, 0), UIntType(IntWidth(width)))
      val e2 = ConstantPropagation.constPropBitExtract(extract)
      // Bit Select always returns UInt, cast if selecting from SInt
      e.tpe match {
        case UIntType(_) => e2
        case SIntType(_) => DoPrim(AsSInt, Seq(e2), Seq.empty, SIntType(IntWidth(width)))
      }
    }
  }

  private def getWidth(t: Type):       Int = bitWidth(t).toInt
  private def getWidth(e: Expression): Int = getWidth(e.tpe)

  // Recursive, updates expression so children exp's have correct widths
  private def onExp(e: Expression): Expression = e.map(onExp) match {
    case Mux(cond, tval, fval, tpe) =>
      Mux(cond, forceWidth(getWidth(tpe))(tval), forceWidth(getWidth(tpe))(fval), tpe)
    case ex: ValidIf => ex.copy(value = forceWidth(getWidth(ex.tpe))(ex.value))
    case ex: DoPrim =>
      ex.op match {
        // pad arguments to ops where the result width is determined as max(w_1, w_2) (+ const)?
        case Lt | Leq | Gt | Geq | Eq | Neq | And | Or | Xor | Add | Sub =>
          ex.map(forceWidth(ex.args.map(getWidth).max))
        case _ => ex
      }
    case ex => ex
  }

  // Recursive. Fixes assignments and register initialization widths
  private def onStmt(s: Statement): Statement = s.map(onExp) match {
    case sx: Connect =>
      assert(
        getWidth(sx.loc) == getWidth(sx.expr),
        "Connection widths should have been taken care of by LegalizeConnects!"
      )
      sx
    case sx: DefRegister =>
      assert(
        getWidth(sx.tpe) == getWidth(sx.init),
        "Register init widths should have been taken care of by LegalizeConnects!"
      )
      sx
    case sx => sx.map(onStmt)
  }

  def run(c: Circuit): Circuit = c.copy(modules = c.modules.map(_.map(onStmt)))
}
