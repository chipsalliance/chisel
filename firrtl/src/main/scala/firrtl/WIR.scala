// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.ir._

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class MPortDir extends FirrtlNode
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object MInfer extends MPortDir {
  def serialize: String = "infer"
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object MRead extends MPortDir {
  def serialize: String = "read"
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object MWrite extends MPortDir {
  def serialize: String = "write"
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object MReadWrite extends MPortDir {
  def serialize: String = "rdwr"
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class CDefMemory(
  info:           Info,
  name:           String,
  tpe:            Type,
  size:           BigInt,
  seq:            Boolean,
  readUnderWrite: ReadUnderWrite.Value = ReadUnderWrite.Undefined
) extends Statement
    with HasInfo
    with UseSerializer {
  def mapExpr(f:       Expression => Expression): Statement = this
  def mapStmt(f:       Statement => Statement):   Statement = this
  def mapType(f:       Type => Type):             Statement = this.copy(tpe = f(tpe))
  def mapString(f:     String => String):         Statement = this.copy(name = f(name))
  def mapInfo(f:       Info => Info):             Statement = this.copy(f(info))
  def foreachStmt(f:   Statement => Unit):        Unit = ()
  def foreachExpr(f:   Expression => Unit):       Unit = ()
  def foreachType(f:   Type => Unit):             Unit = f(tpe)
  def foreachString(f: String => Unit):           Unit = f(name)
  def foreachInfo(f:   Info => Unit):             Unit = f(info)
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class CDefMPort(info: Info, name: String, tpe: Type, mem: String, exps: Seq[Expression], direction: MPortDir)
    extends Statement
    with HasInfo
    with UseSerializer {
  def mapExpr(f:       Expression => Expression): Statement = this.copy(exps = exps.map(f))
  def mapStmt(f:       Statement => Statement):   Statement = this
  def mapType(f:       Type => Type):             Statement = this.copy(tpe = f(tpe))
  def mapString(f:     String => String):         Statement = this.copy(name = f(name))
  def mapInfo(f:       Info => Info):             Statement = this.copy(f(info))
  def foreachStmt(f:   Statement => Unit):        Unit = ()
  def foreachExpr(f:   Expression => Unit):       Unit = exps.foreach(f)
  def foreachType(f:   Type => Unit):             Unit = f(tpe)
  def foreachString(f: String => Unit):           Unit = f(name)
  def foreachInfo(f:   Info => Unit):             Unit = f(info)
}
