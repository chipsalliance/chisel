package firrtl.passes

import firrtl.ir._
import firrtl.{WRef, WSubAccess, WSubIndex, WSubField}
import firrtl.Mappers._
import firrtl.Utils._
import firrtl.WrappedExpression._
import firrtl.Namespace
import scala.collection.mutable


/** Replaces constant [[firrtl.WSubAccess]] with [[firrtl.WSubIndex]]
  * TODO Fold in to High Firrtl Const Prop
  */
object ReplaceAccesses extends Pass {
  def name = "Replace Accesses"

  def run(c: Circuit): Circuit = {
    def onStmt(s: Statement): Statement = s map onStmt map onExp
    def onExp(e: Expression): Expression = e match {
      case WSubAccess(e, UIntLiteral(value, width), t, g) => WSubIndex(e, value.toInt, t, g)
      case e => e map onExp
    }
  
    val newModules = c.modules map {
      case m: ExtModule => m
      case Module(i, n, ps, b) => Module(i, n, ps, onStmt(b))
    }

    Circuit(c.info, newModules, c.main)
  }
}
