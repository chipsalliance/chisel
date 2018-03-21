// See LICENSE for license details.

package firrtl.passes
package memlib

import firrtl._
import firrtl.ir._
import Utils.indent

case class DefAnnotatedMemory(
    info: Info,
    name: String,
    dataType: Type,
    depth: Int,
    writeLatency: Int,
    readLatency: Int,
    readers: Seq[String],
    writers: Seq[String],
    readwriters: Seq[String],
    readUnderWrite: Option[String],
    maskGran: Option[BigInt],
    memRef: Option[(String, String)] /* (Module, Mem) */
    //pins: Seq[Pin],
    ) extends Statement with IsDeclaration {
  def serialize: String = this.toMem.serialize
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = this
  def mapType(f: Type => Type): Statement = this.copy(dataType = f(dataType))
  def mapString(f: String => String): Statement = this.copy(name = f(name))
  def toMem = DefMemory(info, name, dataType, depth,
    writeLatency, readLatency, readers, writers,
    readwriters, readUnderWrite)
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
}
