// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.ir._
import firrtl.options.Dependency
import firrtl.{bitWidth, Transform}

/** Ensures that all connects + register inits have the same bit-width on the rhs and the lhs.
  * The rhs is padded or bit-extacted to fit the width of the lhs.
  * @note technically, width(rhs) > width(lhs) is not legal firrtl, however, we do not error for historic reasons.
  */
object LegalizeConnects extends Pass {

  override def prerequisites = firrtl.stage.Forms.MidForm :+ Dependency(LowerTypes)
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Transform) = false

  def onStmt(s: Statement): Statement = s match {
    case c: Connect =>
      c.copy(expr = PadWidths.forceWidth(bitWidth(c.loc.tpe).toInt)(c.expr))
    case r: DefRegister =>
      r.copy(init = PadWidths.forceWidth(bitWidth(r.tpe).toInt)(r.init))
    case other => other.mapStmt(onStmt)
  }

  def run(c: Circuit): Circuit = {
    c.copy(modules = c.modules.map(_.mapStmt(onStmt)))
  }
}
