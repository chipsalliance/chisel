// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.dataview.reify

import scala.language.experimental.macros
import chisel3.experimental.{requireIsChiselType, requireIsHardware, Analog, BaseModule}
import chisel3.experimental.{prefix, SourceInfo, UnlocatableSourceInfo}
import chisel3.experimental.dataview.{reifyIdentityView, reifySingleTarget, DataViewable}
import chisel3.internal.Builder.pushCommand
import chisel3.internal._
import chisel3.internal.binding._
import chisel3.internal.sourceinfo._
import chisel3.internal.firrtl.ir._
import chisel3.properties.Property
import chisel3.reflect.DataMirror
import chisel3.util.simpleClassName

import scala.reflect.ClassTag
import scala.util.Try

/** This forms the root of the type system for wire data types. The data value
  * must be representable as some number (need not be known at Chisel compile
  * time) of bits, and must have methods to pack / unpack structured data to /
  * from bits.
  *
  * @groupdesc Connect Utilities for connecting hardware components
  * @define coll data
  */
abstract class Data extends DataImpl with SourceInfoDoc {

  /** Does a reinterpret cast of the bits in this node into the format that provides.
    * Returns a new Wire of that type. Does not modify existing nodes.
    *
    * x.asTypeOf(that) performs the inverse operation of x := that.toBits.
    *
    * @note bit widths are NOT checked, may pad or drop bits from input
    * @note that should have known widths
    */
  def asTypeOf[T <: Data](that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_asTypeOf[T <: Data](that: T)(implicit sourceInfo: SourceInfo): T = _asTypeOfImpl(that)

  /** Reinterpret cast to UInt.
    *
    * @note value not guaranteed to be preserved: for example, a SInt of width
    * 3 and value -1 (0b111) would become an UInt with value 7
    * @note Aggregates are recursively packed with the first element appearing
    * in the least-significant bits of the result.
    */
  final def asUInt: UInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asUInt(implicit sourceInfo: SourceInfo): UInt = _asUIntImpl
}

object Data extends ObjectDataImpl
