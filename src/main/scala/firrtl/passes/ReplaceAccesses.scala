// See LICENSE for license details.

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
  def run(c: Circuit): Circuit = {
    def onStmt(s: Statement): Statement = s map onStmt map onExp
    def onExp(e: Expression): Expression = e match {
      case WSubAccess(ex, UIntLiteral(value, width), t, g) => WSubIndex(onExp(ex), value.toInt, t, g)
      case _ => e map onExp
    }
  
    c copy (modules = c.modules map (_ map onStmt))
  }
}
