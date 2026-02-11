// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl.ir._
import chisel3.internal.firrtl.ir.PrimOp._
import chisel3.experimental.{requireIsHardware, SourceInfo}

/** Exists to unify common interfaces of [[Bits]] and [[Reset]].
  *
  * @note This is a workaround because macros cannot override abstract methods.
  */
private[chisel3] sealed trait ToBoolable extends Element {
  def asBool: Bool
}

private[chisel3] trait BitsIntf extends ToBoolable { self: Bits =>

  /** Tail operator
    *
    * @param n the number of bits to remove
    * @return This $coll with the `n` most significant bits removed.
    * @group Bitwise
    */
  def tail(n: Int)(using SourceInfo): UInt = _tailImpl(n)

  /** Head operator
    *
    * @param n the number of bits to take
    * @return The `n` most significant bits of this $coll
    * @group Bitwise
    */
  def head(n: Int)(using SourceInfo): UInt = _headImpl(n)

  /** Returns the specified bit on this $coll as a [[Bool]], statically addressed.
    *
    * @param x an index
    * @return the specified bit
    */

  final def extract(x: BigInt)(using SourceInfo): Bool = _extractImpl(x)

  /** Returns the specified bit on this $coll as a [[Bool]], statically addressed.
    *
    * @param x an index
    * @return the specified bit
    */
  final def apply(x: Int)(using SourceInfo): Bool = _applyImpl(x)

  /** Grab the bottom n bits.  Return 0.U(0.W) if n==0. */
  final def take(n: Int)(using SourceInfo): UInt = _takeImpl(n)

  /** Returns the specified bit on this wire as a [[Bool]], dynamically addressed.
    *
    * @param x a hardware component whose value will be used for dynamic addressing
    * @return the specified bit
    */
  final def extract(x: UInt)(using SourceInfo): Bool = _extractImpl(x)

  /** Returns the specified bit on this wire as a [[Bool]], dynamically addressed.
    *
    * @param x a hardware component whose value will be used for dynamic addressing
    * @return the specified bit
    */
  final def apply(x: UInt)(using SourceInfo): Bool = _applyImpl(x)

  /** Returns a subset of bits on this $coll from `hi` to `lo` (inclusive), statically addressed.
    *
    * @example
    * {{{
    * myBits = 0x5 = 0b101
    * myBits(1,0) => 0b01  // extracts the two least significant bits
    * }}}
    * @param x the high bit
    * @param y the low bit
    * @return a hardware component contain the requested bits
    */
  final def apply(x: Int, y: Int)(using SourceInfo): UInt = _applyImpl(x, y)

  // REVIEW TODO: again, is this necessary? Or just have this and use implicits?
  /** Returns a subset of bits on this $coll from `hi` to `lo` (inclusive), statically addressed.
    *
    * @example
    * {{{
    * myBits = 0x5 = 0b101
    * myBits(1,0) => 0b01  // extracts the two least significant bits
    * }}}
    * @param x the high bit
    * @param y the low bit
    * @return a hardware component contain the requested bits
    */
  final def apply(x: BigInt, y: BigInt)(using SourceInfo): UInt = _applyImpl(x, y)

  /** Pad operator
    *
    * @param that the width to pad to
    * @return this @coll zero padded up to width `that`. If `that` is less than the width of the original component,
    * this method returns the original component.
    * @note For [[SInt]]s only, this will do sign extension.
    * @group Bitwise
    */
  def pad(that: Int)(using SourceInfo): Bits = _padImpl(that)

  /** Bitwise inversion operator
    *
    * @return this $coll with each bit inverted
    * @group Bitwise
    */
  def unary_~(using SourceInfo): Bits = _impl_unary_~

  /** Static left shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many zeros concatenated to its least significant end
    * $sumWidthInt
    * @group Bitwise
    */
  def <<(that: BigInt)(using SourceInfo): Bits = _impl_<<(that)

  /** Static left shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many zeros concatenated to its least significant end
    * $sumWidthInt
    * @group Bitwise
    */
  def <<(that: Int)(using SourceInfo): Bits = _impl_<<(that)

  /** Dynamic left shift operator
    *
    * @param that a hardware component
    * @return this $coll dynamically shifted left by `that` many places, shifting in zeros from the right
    * @note The width of the returned $coll is `width of this + pow(2, width of that) - 1`.
    * @group Bitwise
    */
  def <<(that: UInt)(using SourceInfo): Bits = _impl_<<(that)

  /** Static right shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many least significant bits truncated
    * $unchangedWidth
    * @group Bitwise
    */
  def >>(that: BigInt)(using SourceInfo): Bits = _impl_>>(that)

  /** Static right shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many least significant bits truncated
    * $unchangedWidth
    * @group Bitwise
    */
  def >>(that: Int)(using SourceInfo): Bits = _impl_>>(that)

  /** Dynamic right shift operator
    *
    * @param that a hardware component
    * @return this $coll dynamically shifted right by the value of `that` component, inserting zeros into the most
    * significant bits.
    * $unchangedWidth
    * @group Bitwise
    */
  def >>(that: UInt)(using SourceInfo): Bits = _impl_>>(that)

  /** Returns the contents of this wire as a [[scala.collection.Seq]] of [[Bool]]. */
  def asBools(using SourceInfo): Seq[Bool] = _asBoolsImpl

  /** Reinterpret this $coll as an [[SInt]]
    *
    * @note The arithmetic value is not preserved if the most-significant bit is set. For example, a [[UInt]] of
    * width 3 and value 7 (0b111) would become an [[SInt]] of width 3 and value -1.
    */
  def asSInt(using SourceInfo): SInt = _asSIntImpl

  def asBool: Bool = _asBoolImpl

  /** Concatenation operator
    *
    * @param that a hardware component
    * @return this $coll concatenated to the most significant end of `that`
    * $sumWidth
    * @group Bitwise
    */
  def ##(that: Bits)(using SourceInfo): UInt = _impl_##(that)
}

private[chisel3] trait UIntIntf { self: UInt =>

  // TODO: refactor to share documentation with Num or add independent scaladoc
  /** Unary negation (constant width)
    *
    * @return a $coll equal to zero minus this $coll
    * $constantWidth
    * @group Arithmetic
    */
  def unary_-(using SourceInfo): UInt = _impl_unary_-

  /** Unary negation (constant width)
    *
    * @return a $coll equal to zero minus this $coll shifted right by one.
    * $constantWidth
    * @group Arithmetic
    */
  @deprecated("Use unary_- which has the same behavior", "Chisel 6.8.0")
  def unary_-%(using SourceInfo): UInt = _impl_unary_-%

  override def +(that: UInt): UInt = _impl_+(that)
  override def -(that: UInt): UInt = _impl_-(that)
  override def /(that: UInt): UInt = _impl_/(that)
  override def %(that: UInt): UInt = _impl_%(that)
  override def *(that: UInt): UInt = _impl_*(that)

  /** Multiplication operator
    *
    * @param that a hardware [[SInt]]
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  def *(that: SInt): SInt = _impl_*(that)

  /** Addition operator (expanding width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  def +&(that: UInt): UInt = _impl_+&(that)

  /** Addition operator (constant width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  def +%(that: UInt): UInt = _impl_+%(that)

  /** Subtraction operator (increasing width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  def -&(that: UInt): UInt = _impl_-&(that)

  /** Subtraction operator (constant width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidth
    * @group Arithmetic
    */
  def -%(that: UInt): UInt = _impl_-%(that)

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  def &(that: UInt): UInt = _impl_&(that)

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  def |(that: UInt): UInt = _impl_|(that)

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  def ^(that: UInt): UInt = _impl_^(that)

  def abs: UInt = _absImpl

  override def unary_~(using SourceInfo): UInt = _impl_unary_~

  // REVIEW TODO: Can these be defined on Bits?
  /** Or reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll or'd together
    * @group Bitwise
    */
  def orR: Bool = _orRImpl

  /** And reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll and'd together
    * @group Bitwise
    */
  def andR: Bool = _andRImpl

  /** Exclusive or (xor) reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll xor'd together
    * @group Bitwise
    */
  final def xorR(using SourceInfo): Bool = _xorRImpl

  override def <(that:  UInt): Bool = _impl_<(that)
  override def >(that:  UInt): Bool = _impl_>(that)
  override def <=(that: UInt): Bool = _impl_<=(that)
  override def >=(that: UInt): Bool = _impl_>=(that)

  /** Dynamic not equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is not equal to `that`
    * @group Comparison
    */
  def =/=(that: UInt)(using SourceInfo): Bool = _impl_=/=(that)

  /** Dynamic equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is equal to `that`
    * @group Comparison
    */
  def ===(that: UInt)(using SourceInfo): Bool = _impl_===(that)

  /** Unary not
    *
    * @return a hardware [[Bool]] asserted if this $coll equals zero
    * @group Bitwise
    */
  def unary_!(using SourceInfo): Bool = _impl_unary_!

  override def <<(that: Int)(using SourceInfo):    UInt = _impl_<<(that)
  override def <<(that: BigInt)(using SourceInfo): UInt = _impl_<<(that)
  override def <<(that: UInt)(using SourceInfo):   UInt = _impl_<<(that)

  override def >>(that: Int)(using SourceInfo):    UInt = _impl_>>(that)
  override def >>(that: BigInt)(using SourceInfo): UInt = _impl_>>(that)
  override def >>(that: UInt)(using SourceInfo):   UInt = _impl_>>(that)

  /**
    * Circular shift to the left
    * @param that number of bits to rotate
    * @return UInt of same width rotated left n bits
    */
  def rotateLeft(n: Int)(using SourceInfo): UInt = _rotateLeftImpl(n)

  def rotateLeft(n: UInt)(using SourceInfo): UInt = _rotateLeftImpl(n)

  /**
    * Circular shift to the right
    * @param that number of bits to rotate
    * @return UInt of same width rotated right n bits
    */
  def rotateRight(n: Int)(using SourceInfo): UInt = _rotateRightImpl(n)

  def rotateRight(n: UInt)(using SourceInfo): UInt = _rotateRightImpl(n)

  /** Conditionally set or clear a bit
    *
    * @param off a dynamic offset
    * @param dat set if true, clear if false
    * @return a hrdware $coll with bit `off` set or cleared based on the value of `dat`
    * $unchangedWidth
    */
  def bitSet(off: UInt, dat: Bool)(using SourceInfo): UInt = _bitSetImpl(off, dat)

  // TODO: this eventually will be renamed as toSInt, once the existing toSInt
  // completes its deprecation phase.
  /** Zero extend as [[SInt]]
    *
    * @return an [[SInt]] equal to this $coll with an additional zero in its most significant bit
    * @note The width of the returned [[SInt]] is `width of this` + `1`.
    */
  def zext(using SourceInfo): SInt = _zextImpl

  override def asSInt(using SourceInfo): SInt = _asSIntImpl
}

private[chisel3] trait SIntIntf { self: SInt =>

  /** Unary negation (constant width)
    *
    * @return a hardware $coll equal to zero minus this $coll
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_-(using SourceInfo): SInt = _impl_unary_-

  /** Unary negation (constant width)
    *
    * @return a hardware $coll equal to zero minus `this` $coll
    * $constantWidth
    * @group Arithmetic
    */
  @deprecated("Use unary_- which has the same behavior", "Chisel 6.8.0")
  def unary_-%(using SourceInfo): SInt = _impl_unary_-%

  /** add (default - no growth) operator */
  override def +(that: SInt): SInt = _impl_+(that)

  /** subtract (default - no growth) operator */
  override def -(that: SInt): SInt = _impl_-(that)
  override def *(that: SInt): SInt = _impl_*(that)
  override def /(that: SInt): SInt = _impl_/(that)
  override def %(that: SInt): SInt = _impl_%(that)

  /** Multiplication operator
    *
    * @param that a hardware $coll
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  def *(that: UInt)(using SourceInfo): SInt = _impl_*(that)

  /** Addition operator (expanding width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  def +&(that: SInt)(using SourceInfo): SInt = _impl_+&(that)

  /** Addition operator (constant width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that` shifted right by one
    * $maxWidth
    * @group Arithmetic
    */
  def +%(that: SInt)(using SourceInfo): SInt = _impl_+%(that)

  /** Subtraction operator (increasing width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  def -&(that: SInt)(using SourceInfo): SInt = _impl_-&(that)

  /** Subtraction operator (constant width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that` shifted right by one
    * $maxWidth
    * @group Arithmetic
    */
  def -%(that: SInt)(using SourceInfo): SInt = _impl_-%(that)

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  def &(that: SInt)(using SourceInfo): SInt = _impl_&(that)

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  def |(that: SInt): SInt = _impl_|(that)

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  def ^(that: SInt)(using SourceInfo): SInt = _impl_^(that)

  override def unary_~(using SourceInfo): SInt = _impl_unary_~

  override def <(that:  SInt): Bool = _impl_<(that)
  override def >(that:  SInt): Bool = _impl_>(that)
  override def <=(that: SInt): Bool = _impl_<=(that)
  override def >=(that: SInt): Bool = _impl_>=(that)

  /** Dynamic not equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is not equal to `that`
    * @group Comparison
    */
  def =/=(that: SInt): Bool = _impl_=/=(that)

  /** Dynamic equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is equal to `that`
    * @group Comparison
    */
  def ===(that: SInt): Bool = _impl_===(that)

  def abs: SInt = _absImpl

  override def <<(that: Int)(using SourceInfo):    SInt = _impl_<<(that)
  override def <<(that: BigInt)(using SourceInfo): SInt = _impl_<<(that)
  override def <<(that: UInt)(using SourceInfo):   SInt = _impl_<<(that)

  override def >>(that: Int)(using SourceInfo):    SInt = _impl_>>(that)
  override def >>(that: BigInt)(using SourceInfo): SInt = _impl_>>(that)
  override def >>(that: UInt)(using SourceInfo):   SInt = _impl_>>(that)

  override def asSInt(using SourceInfo): SInt = _asSIntImpl
}

private[chisel3] trait ResetIntf extends ToBoolable { self: Reset =>
  def asAsyncReset(using SourceInfo): AsyncReset
  def asDisable(using SourceInfo):    Disable = _asDisableImpl
}

private[chisel3] trait ResetTypeIntf extends ToBoolable { self: ResetType =>
  def asAsyncReset(using SourceInfo): AsyncReset = _asAsyncResetImpl
  def asBool:                         Bool = _asBoolImpl
  def toBool:                         Bool = asBool
}

private[chisel3] trait AsyncResetIntf { self: AsyncReset =>
  override def toString:              String = stringAccessor("AsyncReset")
  def asAsyncReset(using SourceInfo): AsyncReset = _asAsyncResetImpl
  def asBool:                         Bool = _asBoolImpl
  def toBool:                         Bool = _asBoolImpl
}

private[chisel3] trait BoolIntf extends ToBoolable { self: Bool =>

  // REVIEW TODO: Why does this need to exist and have different conventions
  // than Bits?

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * @group Bitwise
    */
  def &(that: Bool)(using SourceInfo): Bool = _impl_&(that)

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * @group Bitwise
    */
  def |(that: Bool): Bool = _impl_|(that)

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * @group Bitwise
    */
  def ^(that: Bool)(using SourceInfo): Bool = _impl_^(that)

  override def unary_~(using SourceInfo): Bool = _impl_unary_~

  /** Logical or operator
    *
    * @param that a hardware $coll
    * @return the logical or of this $coll and `that`
    * @note this is equivalent to [[Bool!.|(that:chisel3\.Bool)* Bool.|)]]
    * @group Logical
    */
  def ||(that: Bool)(using SourceInfo): Bool = _impl_||(that)

  /** Logical and operator
    *
    * @param that a hardware $coll
    * @return the logical and of this $coll and `that`
    * @note this is equivalent to [[Bool!.&(that:chisel3\.Bool)* Bool.&]]
    * @group Logical
    */
  def &&(that: Bool)(using SourceInfo): Bool = _impl_&&(that)

  override def asBool: Bool = _asBoolImpl

  /** Reinterprets this $coll as a clock */
  def asClock(using SourceInfo): Clock = _asClockImpl

  def asAsyncReset(using SourceInfo): AsyncReset = _asAsyncResetImpl

  /** Reinterprets this $coll as a reset */
  def asReset(using SourceInfo): Reset = _asResetImpl

  /** Logical implication
    *
    * @param that a boolean signal
    * @return [[!this || that]]
    */
  def implies(that: Bool)(using SourceInfo): Bool = _impl_implies(that)
}
