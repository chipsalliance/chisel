// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.ir._

trait Kind
case object WireKind extends Kind
case object PoisonKind extends Kind
case object RegKind extends Kind
case object InstanceKind extends Kind
case object PortKind extends Kind
case object NodeKind extends Kind
case object MemKind extends Kind
case object ExpKind extends Kind
case object UnknownKind extends Kind

trait Flow
case object SourceFlow extends Flow
case object SinkFlow extends Flow
case object DuplexFlow extends Flow
case object UnknownFlow extends Flow

abstract class MPortDir extends FirrtlNode
case object MInfer extends MPortDir {
  def serialize: String = "infer"
}
case object MRead extends MPortDir {
  def serialize: String = "read"
}
case object MWrite extends MPortDir {
  def serialize: String = "write"
}
case object MReadWrite extends MPortDir {
  def serialize: String = "rdwr"
}

case class CDefMemory(
  info:           Info,
  name:           String,
  tpe:            Type,
  size:           BigInt,
  seq:            Boolean,
  readUnderWrite: ReadUnderWrite.Value = ReadUnderWrite.Undefined)
    extends Statement
    with HasInfo
    with CanBeReferenced
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
case class CDefMPort(info: Info, name: String, tpe: Type, mem: String, exps: Seq[Expression], direction: MPortDir)
    extends Statement
    with HasInfo
    with CanBeReferenced
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
