// See LICENSE for license details.

package chisel3

import scala.language.experimental.macros
import chisel3.internal.firrtl.PrimOp.{GreaterEqOp, GreaterOp, LessEqOp, LessOp, PadOp}
import chisel3.internal.firrtl.{KnownWidth, Width}
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform}

// scalastyle:off method.name

trait Bitwise[T <: Bits] extends Num[T] {
  this: T =>

  /** Pad operator
   *
   * @param that the width to pad to
   * @return this @coll zero padded up to width `that`. If `that` is less than the width of the original component,
   * this method returns the original component.
   * @note For [[SInt]]s only, this will do sign extension.
   * @group Bitwise
   */
  final def pad(that: Int): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_pad(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = this.width match {
    case KnownWidth(w) if w >= that => this
    case _ => binop(sourceInfo, cloneTypeWidth(this.width max Width(that)), PadOp, that).asInstanceOf[T]
  }

  /** Bitwise inversion operator
   *
   * @return this $coll with each bit inverted
   * @group Bitwise
   */
  def unary_~ (): T = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Static left shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many zeros concatenated to its least significant end
   * $sumWidthInt
   * @group Bitwise
   */
  def << (that: BigInt): T

  /** @group SourceInfoTransformMacro */
  def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Static left shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many zeros concatenated to its least significant end
   * $sumWidthInt
   * @group Bitwise
   */
  def << (that: Int): T

  /** @group SourceInfoTransformMacro */
  def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Dynamic left shift operator
   *
   * @param that a hardware component
   * @return this $coll dynamically shifted left by `that` many places, shifting in zeros from the right
   * @note The width of the returned $coll is `width of this + pow(2, width of that) - 1`.
   * @group Bitwise
   */
  def << (that: UInt): T

  /** @group SourceInfoTransformMacro */
  def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Static right shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many least significant bits truncated
   * $unchangedWidth
   * @group Bitwise
   */
  // REVIEW TODO: redundant
  def >> (that: BigInt): T

  /** @group SourceInfoTransformMacro */
  def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Static right shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many least significant bits truncated
   * $unchangedWidth
   * @group Bitwise
   */
  def >> (that: Int): T

  /** @group SourceInfoTransformMacro */
  def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Dynamic right shift operator
   *
   * @param that a hardware component
   * @return this $coll dynamically shifted right by the value of `that` component, inserting zeros into the most
   * significant bits.
   * $unchangedWidth
   * @group Bitwise
   */
  def >> (that: UInt): T

  /** @group SourceInfoTransformMacro */
  def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  // TODO: refactor to share documentation with Num or add independent scaladoc
  /** Unary negation (expanding width)
   *
   * @return a $coll equal to zero minus this $coll
   * $constantWidth
   * @group Arithmetic
   */
  final def unary_- (): T = macro SourceInfoTransform.noArg

  /** Unary negation (constant width)
   *
   * @return a $coll equal to zero minus this $coll shifted right by one.
   * $constantWidth
   * @group Arithmetic
   */
  final def unary_-% (): T = macro SourceInfoTransform.noArg

  override def do_< (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, LessOp, that)
  override def do_> (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, GreaterEqOp, that)
}
