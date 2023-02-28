// SPDX-License-Identifier: Apache-2.0

package firrtl.backends.experimental.smt.random

import firrtl.Utils
import firrtl.ir._

/** Named source of random values. If there is no clock expression, than it will be clocked by the global clock. */
case class DefRandom(
  info:  Info,
  name:  String,
  tpe:   Type,
  clock: Option[Expression],
  en:    Expression = Utils.True())
    extends Statement
    with HasInfo
    with IsDeclaration
    with CanBeReferenced
    with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement =
    DefRandom(info, name, tpe, clock.map(f), f(en))
  def mapType(f:     Type => Type):      Statement = this.copy(tpe = f(tpe))
  def mapString(f:   String => String):  Statement = this.copy(name = f(name))
  def mapInfo(f:     Info => Info):      Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = { clock.foreach(f); f(en) }
  def foreachType(f:   Type => Unit):   Unit = f(tpe)
  def foreachString(f: String => Unit): Unit = f(name)
  def foreachInfo(f:   Info => Unit):   Unit = f(info)
}
