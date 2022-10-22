// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.sourceinfo.SourceInfo

import scala.collection.immutable.SeqMap


/** Data type for representing any type that is ok to either leave dangling or unassigned from :<>= (TODO: :<=? :>=? :#=?)
  *
  * Uses the underlying opaque type support.
  *
  * @note This API is experimental and subject to change
  */
class Waivable[T <: Data] private[chisel3] (
  tpe: => T,
  val okToDangle: Boolean,
  val okToUnassign: Boolean,
)(
  implicit sourceInfo: SourceInfo,
  compileOptions:      CompileOptions
) extends Record {
  requireIsChiselType(tpe, s"Chisel hardware type $tpe must be a pure type, not bound to hardware.")

  /** The underlying hardware component, is either the Chisel data type (if `this` is unbound) or hardware component (if `this` is bound to hardware) */
  lazy val underlying: T = tpe

  val elements = SeqMap("" -> underlying)
  override def opaqueType = elements.size == 1
  override def cloneType: this.type = {
    val freshType = if (tpe.isSynthesizable) chiselTypeOf(tpe) else tpe.cloneType
    (new Waivable[T](freshType, okToDangle, okToUnassign)).asInstanceOf[this.type]
  }
}

/** Object that provides factory methods for [[Waivable]] objects
  *
  * @note This API is experimental and subject to change
  */
object Waivable {

  /** Build a Waivable[T <: Data]
    *
    * @param tpe the Chisel data type
    * @param default the Chisel default value, must be bound to a hardware value
    */
  def apply[T <: Data](
    tpe:     T,
    okToDangle: Boolean,
    okToUnassign: Boolean,
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Waivable[T] = new Waivable(tpe, okToDangle, okToUnassign)
}
