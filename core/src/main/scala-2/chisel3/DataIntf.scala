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

private[chisel3] trait DataIntf extends SourceInfoDoc { self: Data =>

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

  /** The "strong connect" operator.
    * @group connection
    */
  final def :=(that: => Data)(implicit sourceInfo: SourceInfo): Unit = _colonEqImpl(that)

  /** The "bulk connect operator".
    * @group connection
    */
  final def <>(that: => Data)(implicit sourceInfo: SourceInfo): Unit = _bulkConnectImpl(that)
}

private[chisel3] trait WireFactory$Intf { self: WireFactory =>

  def apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T = _applyImpl(source)
}

private[chisel3] trait WireDefaultImpl$Intf { self: WireDefaultImpl =>

  def apply[T <: Data](t: T, init: DontCare.type)(implicit sourceInfo: SourceInfo): T =
    _applyDontCareImpl(t, init)

  def apply[T <: Data](t: T, init: T)(implicit sourceInfo: SourceInfo): T =
    _applyTwoArgImpl(t, init)

  def apply[T <: Data](init: T)(implicit sourceInfo: SourceInfo): T =
    _applyOneArgImpl(init)
}

private[chisel3] trait Data$ObjIntf { self: Data.type =>

  /**
    * Provides generic, recursive equality for [[Bundle]] and [[Vec]] hardware.
    *
    * @param lhs The [[Data]] hardware on the left-hand side of the equality
    */
  implicit class DataEquality[T <: Data](lhs: T)(implicit sourceInfo: SourceInfo) {

    /** Dynamic recursive equality operator for generic [[Data]]
      *
      * @param rhs a hardware [[Data]] to compare `lhs` to
      * @return a hardware [[Bool]] asserted if `lhs` is equal to `rhs`
      * @throws ChiselException when `lhs` and `rhs` are different types during elaboration time
      */
    def ===(rhs: T): Bool = _dataEqualityImpl(lhs, rhs)
  }

  implicit class AsReadOnly[T <: Data](self: T) {

    /** Returns a read-only view of this Data
      *
      * It is illegal to connect to the return value of this method.
      * This Data this method is called on must be a hardware type.
      */
    def readOnly(implicit sourceInfo: SourceInfo): T = _readOnlyImpl(self)
  }
}
