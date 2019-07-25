// See LICENSE for license details.
package chisel3

import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform}
// scalastyle:off method.name number.of.methods

// REVIEW TODO: Further discussion needed on what Num actually is.

/** Abstract trait defining operations available on numeric-like hardware data types.
  *
  * @tparam T the underlying type of the number
  * @groupdesc Arithmetic Arithmetic hardware operators
  * @groupdesc Comparison Comparison hardware operators
  * @groupdesc Logical Logical hardware operators
  * @define coll numeric-like type
  * @define numType hardware type
  * @define canHaveHighCost can result in significant cycle time and area costs
  * @define canGenerateA This method generates a
  * @define singleCycleMul  @note $canGenerateA fully combinational multiplier which $canHaveHighCost.
  * @define singleCycleDiv  @note $canGenerateA fully combinational divider which $canHaveHighCost.
  * @define maxWidth        @note The width of the returned $numType is `max(width of this, width of that)`.
  * @define maxWidthPlusOne @note The width of the returned $numType is `max(width of this, width of that) + 1`.
  * @define sumWidth        @note The width of the returned $numType is `width of this` + `width of that`.
  * @define unchangedWidth  @note The width of the returned $numType is unchanged, i.e., the `width of this`.
  */
trait Num[T <: Data] {


  /** Unary negation (expanding width)
   *
   * @return a $coll equal to zero minus this $coll
   * $constantWidth
   * @group Arithmetic
   */
  def unary_- (): T = macro SourceInfoTransform.noArg

  /** Unary negation (constant width)
   *
   * @return a $coll equal to zero minus this $coll shifted right by one.
   * $constantWidth
   * @group Arithmetic
   */
  def unary_-% (): T = macro SourceInfoTransform.noArg

  /** Addition operator
    *
    * @param that a $numType
    * @return the sum of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */

  final def + (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_+ (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Multiplication operator
    *
    * @param that a $numType
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  final def * (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_* (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Division operator
    *
    * @param that a $numType
    * @return the quotient of this $coll divided by `that`
    * $singleCycleDiv
    * @todo full rules
    * @group Arithmetic
    */
  final def / (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_/ (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Modulo operator
    *
    * @param that a $numType
    * @return the remainder of this $coll divided by `that`
    * $singleCycleDiv
    * @group Arithmetic
    */
  final def % (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_% (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Subtraction operator
    *
    * @param that a $numType
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def - (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_- (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Less than operator
    *
    * @param that a $numType
    * @return a hardware [[Bool]] asserted if this $coll is less than `that`
    * @group Comparison
    */
  final def < (that: T): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_< (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Less than or equal to operator
    *
    * @param that a $numType
    * @return a hardware [[Bool]] asserted if this $coll is less than or equal to `that`
    * @group Comparison
    */
  final def <= (that: T): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Greater than operator
    *
    * @param that a hardware component
    * @return a hardware [[Bool]] asserted if this $coll is greater than `that`
    * @group Comparison
    */
  final def > (that: T): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_> (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Greater than or equal to operator
    *
    * @param that a hardware component
    * @return a hardware [[Bool]] asserted if this $coll is greather than or equal to `that`
    * @group Comparison
    */
  final def >= (that: T): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** Absolute value operator
    *
    * @return a $numType with a value equal to the absolute value of this $coll
    * $unchangedWidth
    * @group Arithmetic
    */
  final def abs(): T = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** Minimum operator
    *
    * @param that a hardware $coll
    * @return a $numType with a value equal to the mimimum value of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  final def min(that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_min(that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    Mux(this < that, this.asInstanceOf[T], that)

  /** Maximum operator
    *
    * @param that a $numType
    * @return a $numType with a value equal to the mimimum value of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  final def max(that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_max(that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
    Mux(this < that, that, this.asInstanceOf[T])

//  def do_< (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool
//  def do_> (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool
//  def do_<= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool
//  def do_>= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool


  /** Dynamic not equals operator
   *
   * @param that a hardware $coll
   * @return a hardware [[Bool]] asserted if this $coll is not equal to `that`
   * @group Comparison
   */
  def =/= (that: T): Bool = macro SourceInfoTransform.thatArg

  /** Dynamic equals operator
   *
   * @param that a hardware $coll
   * @return a hardware [[Bool]] asserted if this $coll is equal to `that`
   * @group Comparison
   */
  def === (that: T): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_=/= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool
  /** @group SourceInfoTransformMacro */
  def do_=== (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool

  /** @group SourceInfoTransformMacro */
  def do_& (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T
  /** @group SourceInfoTransformMacro */
  def do_| (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T
  /** @group SourceInfoTransformMacro */
  def do_^ (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T


  /** Bitwise inversion operator
   *
   * @return this $coll with each bit inverted
   * @group Bitwise
   */
  def unary_~ (): T = macro SourceInfoWhiteboxTransform.noArg


  /** Static left shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many zeros concatenated to its least significant end
   * $sumWidthInt
   * @group Bitwise
   */
  // REVIEW TODO: redundant
  // REVIEW TODO: should these return this.type or T?
  def << (that: BigInt): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Static left shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many zeros concatenated to its least significant end
   * $sumWidthInt
   * @group Bitwise
   */
  def << (that: Int): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Dynamic left shift operator
   *
   * @param that a hardware component
   * @return this $coll dynamically shifted left by `that` many places, shifting in zeros from the right
   * @note The width of the returned $coll is `width of this + pow(2, width of that) - 1`.
   * @group Bitwise
   */
  def << (that: UInt): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Static right shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many least significant bits truncated
   * $unchangedWidth
   * @group Bitwise
   */
  // REVIEW TODO: redundant
  def >> (that: BigInt): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Static right shift operator
   *
   * @param that an amount to shift by
   * @return this $coll with `that` many least significant bits truncated
   * $unchangedWidth
   * @group Bitwise
   */
  def >> (that: Int): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Dynamic right shift operator
   *
   * @param that a hardware component
   * @return this $coll dynamically shifted right by the value of `that` component, inserting zeros into the most
   * significant bits.
   * $unchangedWidth
   * @group Bitwise
   */
  def >> (that: UInt): T = macro SourceInfoWhiteboxTransform.thatArg

  /** Addition operator (expanding width)
   *
   * @param that a hardware $coll
   * @return the sum of this $coll and `that`
   * $maxWidthPlusOne
   * @group Arithmetic
   */
  final def +& (that: T): T = macro SourceInfoTransform.thatArg

  /** Addition operator (constant width)
   *
   * @param that a hardware $coll
   * @return the sum of this $coll and `that`
   * $maxWidth
   * @group Arithmetic
   */
  final def +% (that: T): T = macro SourceInfoTransform.thatArg

  /** Subtraction operator (increasing width)
   *
   * @param that a hardware $coll
   * @return the difference of this $coll less `that`
   * $maxWidthPlusOne
   * @group Arithmetic
   */
  final def -& (that: T): T = macro SourceInfoTransform.thatArg

  /** Subtraction operator (constant width)
   *
   * @param that a hardware $coll
   * @return the difference of this $coll less `that`
   * $maxWidth
   * @group Arithmetic
   */
  final def -% (that: T): T = macro SourceInfoTransform.thatArg

  /** Bitwise and operator
   *
   * @param that a hardware $coll
   * @return the bitwise and of  this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def & (that: T): T = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
   *
   * @param that a hardware $coll
   * @return the bitwise or of this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def | (that: T): T = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
   *
   * @param that a hardware $coll
   * @return the bitwise xor of this $coll and `that`
   * $maxWidth
   * @group Bitwise
   */
  def ^ (that: T): T = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** @group SourceInfoTransformMacro */
  def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** @group SourceInfoTransformMacro */
  def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** @group SourceInfoTransformMacro */
  def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** @group SourceInfoTransformMacro */
  def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** @group SourceInfoTransformMacro */
  def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

  /** @group SourceInfoTransformMacro */
  def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T

}
