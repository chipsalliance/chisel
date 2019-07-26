// See LICENSE for license details.
package chisel3

import scala.language.experimental.macros
import chisel3.experimental.FixedPoint
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl.{BinaryPoint, DefPrim, ILit, KnownBinaryPoint}
import chisel3.internal.firrtl.PrimOp._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform}
import chisel3.internal.throwException

// scalastyle:off method.name number.of.methods
private[chisel3] trait NumBits[T <: Bits] extends Num[T] {
  this: T =>

  /** @group SourceInfoTransformMacro */
  def do_=/= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, NotEqualOp, that)
  /** @group SourceInfoTransformMacro */
  def do_=== (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, EqualOp, that)

  /** Reinterpret this $coll as a [[FixedPoint]].
   *
   * @note The value is not guaranteed to be preserved. For example, a [[UInt]] of width 3 and value 7 (0b111) would
   * become a [[FixedPoint]] with value -1. The interpretation of the number is also affected by the specified binary
   * point. '''Caution is advised!'''
   */
  final def asFixedPoint(that: BinaryPoint): FixedPoint = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_asFixedPoint(binaryPoint: BinaryPoint)
                     (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
    binaryPoint match {
      case KnownBinaryPoint(value) =>
        val iLit = ILit(value)
        pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref, iLit))
      case _ =>
        throwException(
          s"cannot call $this.asFixedPoint(binaryPoint=$binaryPoint), you must specify a known binaryPoint"
        )
    }
  }

  /** @group SourceInfoTransformMacro */
  def do_< (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, LessOp, that)
  /** @group SourceInfoTransformMacro */
  def do_> (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, GreaterOp, that)
  /** @group SourceInfoTransformMacro */
  def do_<= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, LessEqOp, that)
  /** @group SourceInfoTransformMacro */
  def do_>= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, GreaterEqOp, that)

}
