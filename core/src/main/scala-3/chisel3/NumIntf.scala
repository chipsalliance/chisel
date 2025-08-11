// SPDX-License-Identifier: Apache-2.0

package chisel3

// REVIEW TODO: Further discussion needed on what Num actually is.

private[chisel3] trait NumIntf[T <: Data] { self: Num[T] =>
  // def << (b: T): T
  // def >> (b: T): T
  // def unary_-(): T

  // REVIEW TODO: double check ops conventions against FIRRTL

  /** Addition operator
    *
    * @param that a $numType
    * @return the sum of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  def +(that: T): T

  /** Multiplication operator
    *
    * @param that a $numType
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  def *(that: T): T

  /** Division operator
    *
    * @param that a $numType
    * @return the quotient of this $coll divided by `that`
    * $singleCycleDiv
    * @todo full rules
    * @group Arithmetic
    */
  def /(that: T): T

  /** Modulo operator
    *
    * @param that a $numType
    * @return the remainder of this $coll divided by `that`
    * $singleCycleDiv
    * @group Arithmetic
    */
  def %(that: T): T

  /** Subtraction operator
    *
    * @param that a $numType
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  def -(that: T): T

  /** Less than operator
    *
    * @param that a $numType
    * @return a hardware [[Bool]] asserted if this $coll is less than `that`
    * @group Comparison
    */
  def <(that: T): Bool

  /** Less than or equal to operator
    *
    * @param that a $numType
    * @return a hardware [[Bool]] asserted if this $coll is less than or equal to `that`
    * @group Comparison
    */
  def <=(that: T): Bool

  /** Greater than operator
    *
    * @param that a hardware component
    * @return a hardware [[Bool]] asserted if this $coll is greater than `that`
    * @group Comparison
    */
  def >(that: T): Bool

  /** Greater than or equal to operator
    *
    * @param that a hardware component
    * @return a hardware [[Bool]] asserted if this $coll is greather than or equal to `that`
    * @group Comparison
    */
  def >=(that: T): Bool

  /** Absolute value operator
    *
    * @return a $numType with a value equal to the absolute value of this $coll
    * $unchangedWidth
    * @group Arithmetic
    */
  @deprecated(
    "Calling this function with an empty argument list is invalid in Scala 3. Use the form without parentheses instead",
    "Chisel 3.5"
  )
  def abs: T

  /** Minimum operator
    *
    * @param that a hardware $coll
    * @return a $numType with a value equal to the minimum value of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  def min(that: T): T =
    Mux(this < that, this.asInstanceOf[T], that)

  /** Maximum operator
    *
    * @param that a $numType
    * @return a $numType with a value equal to the maximum value of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  def max(that: T): T =
    Mux(this < that, that, this.asInstanceOf[T])
}
