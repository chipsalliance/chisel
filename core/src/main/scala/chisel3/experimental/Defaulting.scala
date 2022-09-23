// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.sourceinfo.SourceInfo

import scala.collection.immutable.SeqMap

/** Data type for representing any type who has a default value
  *
  * Uses the underlying opaque type support.
  *
  * @note This API is experimental and subject to change
  */
final class Defaulting[T <: Data] private (private[chisel3] val tpe: T, val defaultValue: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) extends Record {
  requireIsChiselType(tpe, s"Chisel hardware type $tpe must be a pure type, not bound to hardware.")
  requireIsHardware(defaultValue, s"Default value $defaultValue must be bound to a hardware component")

  /** The underlying hardware component, is either the Chisel data type or hardware component */
  val underlying: T = tpe

  /** The default value for this Defaulting */
  val default: T = defaultValue

  val elements = SeqMap("" -> underlying)
  override def opaqueType = elements.size == 1
  override def cloneType: this.type = {
    val freshType = if(tpe.isSynthesizable) chiselTypeOf(tpe) else tpe.cloneType
    (new Defaulting[T](freshType, defaultValue)).asInstanceOf[this.type]
  }
}


/** Object that provides factory methods for [[Analog]] objects
  *
  * @note This API is experimental and subject to change
  */
object Defaulting {
  /** Build a Defaulting[T <: Data]
    *
    * @param tpe the Chisel data type
    * @param default the Chisel default value, must be bound to a hardware value
    */
  def apply[T <: Data](tpe: T, default: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Defaulting[T] = new Defaulting(tpe, default)

  /** Build a Defaulting[T <: Data]
    *
    * @param default the Chisel default value, must be bound to a hardware value. The underlying type is pulled from the default value.
    */
  def apply[T <: Data](default: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Defaulting[T] = new Defaulting(chiselTypeOf(default), default)
}
