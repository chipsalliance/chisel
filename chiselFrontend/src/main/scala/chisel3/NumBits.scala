// See LICENSE for license details.
package chisel3

import scala.language.experimental.macros
import chisel3.experimental.FixedPoint
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl.{BinaryPoint, DefPrim, ILit, KnownBinaryPoint}
import chisel3.internal.firrtl.PrimOp._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform}
import chisel3.internal.throwException

// scalastyle:off method.name number.of.methods
private[chisel3] trait NumBits[T <: Bits] {
  this: T with Num[T] =>

  /** Static left shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many zeros concatenated to its least significant end
   * $sumWidthInt
   * @group Bitwise
   */
  override def << (that: BigInt): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Static left shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many zeros concatenated to its least significant end
   * $sumWidthInt
   * @group Bitwise
   */
  override def << (that: Int): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Dynamic left shift operator
   *
   * @param that a hardware component
   * @return this $coll dynamically shifted left by `that` many places, shifting in zeros from the right
   * @note The width of the returned $coll is `width of this + pow(2, width of that) - 1`.
   * @group Bitwise
   */
  override def << (that: UInt): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Static right shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many least significant bits truncated
   * $unchangedWidth
   * @group Bitwise
   */
  // REVIEW TODO: redundant
  override def >> (that: BigInt): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Static right shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many least significant bits truncated
   * $unchangedWidth
   * @group Bitwise
   */
  override def >> (that: Int): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Dynamic right shift operator
   *
   * @param that a hardware component
   * @return this $coll dynamically shifted right by the value of `that` component, inserting zeros into the most
   * significant bits.
   * $unchangedWidth
   * @group Bitwise
   */
  override def >> (that: UInt): T = macro SourceInfoWhiteboxTransform.thatArg


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

  /** @group SourceInfoTransformMacro */
  def do_unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T
  /** @group SourceInfoTransformMacro */
  def do_unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** @group SourceInfoTransformMacro */
  def do_+% (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T
  /** @group SourceInfoTransformMacro */
  def do_-% (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** add (default - no growth) operator */
  override def do_+ (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    this +% that
  /** subtract (default - no growth) operator */
  override def do_- (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    this -% that
}
