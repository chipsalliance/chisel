// See LICENSE for license details.

package chisel3

import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform}

// scalastyle:off method.name

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
  self: Num[T] =>
  // def << (b: T): T
  // def >> (b: T): T
  //def unary_-(): T

  // REVIEW TODO: double check ops conventions against FIRRTL

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
}
