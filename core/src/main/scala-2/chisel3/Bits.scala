// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.{
  IntLiteralApplyTransform,
  SourceInfoTransform,
  SourceInfoWhiteboxTransform,
  UIntTransform
}

/** Exists to unify common interfaces of [[Bits]] and [[Reset]].
  *
  * @note This is a workaround because macros cannot override abstract methods.
  */
private[chisel3] sealed trait ToBoolable extends Element {

  /** Casts this $coll to a [[Bool]]
    *
    * @note The width must be known and equal to 1
    */
  final def asBool: Bool = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asBool(implicit sourceInfo: SourceInfo): Bool
}

/** A data type for values represented by a single bitvector. This provides basic bitwise operations.
  *
  * @groupdesc Bitwise Bitwise hardware operators
  * @define coll [[Bits]]
  * @define sumWidthInt    @note The width of the returned $coll is `width of this` + `that`.
  * @define sumWidth       @note The width of the returned $coll is `width of this` + `width of that`.
  * @define unchangedWidth @note The width of the returned $coll is unchanged, i.e., the `width of this`.
  */
sealed abstract class Bits(private[chisel3] val width: Width) extends BitsImpl with ToBoolable {

  /** Tail operator
    *
    * @param n the number of bits to remove
    * @return This $coll with the `n` most significant bits removed.
    * @group Bitwise
    */
  final def tail(n: Int): UInt = macro SourceInfoTransform.nArg

  /** Head operator
    *
    * @param n the number of bits to take
    * @return The `n` most significant bits of this $coll
    * @group Bitwise
    */
  final def head(n: Int): UInt = macro SourceInfoTransform.nArg

  /** @group SourceInfoTransformMacro */
  def do_tail(n: Int)(implicit sourceInfo: SourceInfo): UInt = _tailImpl(n)

  /** @group SourceInfoTransformMacro */
  def do_head(n: Int)(implicit sourceInfo: SourceInfo): UInt = _headImpl(n)

  /** Returns the specified bit on this $coll as a [[Bool]], statically addressed.
    *
    * @param x an index
    * @return the specified bit
    */
  final def extract(x: BigInt): Bool = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  final def do_extract(x: BigInt)(implicit sourceInfo: SourceInfo): Bool = _extractImpl(x)

  /** Returns the specified bit on this $coll as a [[Bool]], statically addressed.
    *
    * @param x an index
    * @return the specified bit
    */
  final def apply(x: BigInt): Bool = macro IntLiteralApplyTransform.safeApply

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: BigInt)(implicit sourceInfo: SourceInfo): Bool = _applyImpl(x)

  /** Returns the specified bit on this $coll as a [[Bool]], statically addressed.
    *
    * @param x an index
    * @return the specified bit
    */
  final def apply(x: Int): Bool = macro IntLiteralApplyTransform.safeApply

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: Int)(implicit sourceInfo: SourceInfo): Bool = _applyImpl(x)

  /** Grab the bottom n bits.  Return 0.U(0.W) if n==0. */
  final def take(n: Int): UInt = macro SourceInfoTransform.nArg

  final def do_take(n: Int)(implicit sourceInfo: SourceInfo): UInt = _takeImpl(n)

  /** Returns the specified bit on this wire as a [[Bool]], dynamically addressed.
    *
    * @param x a hardware component whose value will be used for dynamic addressing
    * @return the specified bit
    */
  final def extract(x: UInt): Bool = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  final def do_extract(x: UInt)(implicit sourceInfo: SourceInfo): Bool = _extractImpl(x)

  /** Returns the specified bit on this wire as a [[Bool]], dynamically addressed.
    *
    * @param x a hardware component whose value will be used for dynamic addressing
    * @return the specified bit
    */
  final def apply(x: UInt): Bool = macro SourceInfoTransform.xArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: UInt)(implicit sourceInfo: SourceInfo): Bool = _applyImpl(x)

  /** Returns a subset of bits on this $coll from `hi` to `lo` (inclusive), statically addressed.
    *
    * @example
    * {{{
    *   val myBits = "0b101".U
    *   myBits(1, 0) // "0b01".U  // extracts the two least significant bits
    *
    *   // Note that zero-width ranges are also legal
    *   myBits(-1, 0) // 0.U(0.W) // zero-width UInt
    * }}}
    * @param x the high bit
    * @param y the low bit
    * @return a hardware component containing the requested bits
    */
  final def apply(x: Int, y: Int): UInt = macro SourceInfoTransform.xyArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: Int, y: Int)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(x, y)

  /** Returns a subset of bits on this $coll from `hi` to `lo` (inclusive), statically addressed.
    *
    * @example
    * {{{
    *   val myBits = "0b101".U
    *   myBits(1, 0) // "0b01".U  // extracts the two least significant bits
    *
    *   // Note that zero-width ranges are also legal
    *   myBits(-1, 0) // 0.U(0.W) // zero-width UInt
    * }}}
    * @param x the high bit
    * @param y the low bit
    * @return a hardware component containing the requested bits
    */
  final def apply(x: BigInt, y: BigInt): UInt = macro SourceInfoTransform.xyArg

  /** @group SourceInfoTransformMacro */
  final def do_apply(x: BigInt, y: BigInt)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(x, y)

  /** Pad operator
    *
    * @param that the width to pad to
    * @return this @coll zero padded up to width `that`. If `that` is less than the width of the original component,
    * this method returns the original component.
    * @note For [[SInt]]s only, this will do sign extension.
    * @group Bitwise
    */
  final def pad(that: Int): this.type = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_pad(that: Int)(implicit sourceInfo: SourceInfo): this.type = _padImpl(that)

  /** Bitwise inversion operator
    *
    * @return this $coll with each bit inverted
    * @group Bitwise
    */
  final def unary_~ : Bits = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_~(implicit sourceInfo: SourceInfo): Bits = _impl_unary_~

  /** Static left shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many zeros concatenated to its least significant end
    * $sumWidthInt
    * @group Bitwise
    */
  // REVIEW TODO: redundant
  // REVIEW TODO: should these return this.type or Bits?
  final def <<(that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<<(that: BigInt)(implicit sourceInfo: SourceInfo): Bits = _impl_<<(that)

  /** Static left shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many zeros concatenated to its least significant end
    * $sumWidthInt
    * @group Bitwise
    */
  final def <<(that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<<(that: Int)(implicit sourceInfo: SourceInfo): Bits = _impl_<<(that)

  /** Dynamic left shift operator
    *
    * @param that a hardware component
    * @return this $coll dynamically shifted left by `that` many places, shifting in zeros from the right
    * @note The width of the returned $coll is `width of this + pow(2, width of that) - 1`.
    * @group Bitwise
    */
  final def <<(that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_<<(that: UInt)(implicit sourceInfo: SourceInfo): Bits = _impl_<<(that)

  /** Static right shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many least significant bits truncated
    * $unchangedWidth
    * @group Bitwise
    */
  // REVIEW TODO: redundant
  final def >>(that: BigInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>>(that: BigInt)(implicit sourceInfo: SourceInfo): Bits = _impl_>>(that)

  /** Static right shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many least significant bits truncated
    * $unchangedWidth
    * @group Bitwise
    */
  final def >>(that: Int): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>>(that: Int)(implicit sourceInfo: SourceInfo): Bits = _impl_>>(that)

  /** Dynamic right shift operator
    *
    * @param that a hardware component
    * @return this $coll dynamically shifted right by the value of `that` component, inserting zeros into the most
    * significant bits.
    * $unchangedWidth
    * @group Bitwise
    */
  final def >>(that: UInt): Bits = macro SourceInfoWhiteboxTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_>>(that: UInt)(implicit sourceInfo: SourceInfo): Bits = _impl_>>(that)

  /** Returns the contents of this wire as a [[scala.collection.Seq]] of [[Bool]]. */
  final def asBools: Seq[Bool] = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asBools(implicit sourceInfo: SourceInfo): Seq[Bool] = _asBoolsImpl

  /** Reinterpret this $coll as an [[SInt]]
    *
    * @note The arithmetic value is not preserved if the most-significant bit is set. For example, a [[UInt]] of
    * width 3 and value 7 (0b111) would become an [[SInt]] of width 3 and value -1.
    */
  final def asSInt: SInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asSInt(implicit sourceInfo: SourceInfo): SInt = _asSIntImpl

  def do_asBool(implicit sourceInfo: SourceInfo): Bool = _asBoolImpl

  /** Concatenation operator
    *
    * @param that a hardware component
    * @return this $coll concatenated to the most significant end of `that`
    * $sumWidth
    * @group Bitwise
    */
  final def ##(that: Bits): UInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_##(that: Bits)(implicit sourceInfo: SourceInfo): UInt = _impl_##(that)
}

/** A data type for unsigned integers, represented as a binary bitvector. Defines arithmetic operations between other
  * integer types.
  *
  * @define coll [[UInt]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class UInt private[chisel3] (width: Width) extends Bits(width) with UIntImpl {

  // TODO: refactor to share documentation with Num or add independent scaladoc
  /** Unary negation (expanding width)
    *
    * @return a $coll equal to zero minus this $coll
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_- : UInt = macro SourceInfoTransform.noArg

  /** Unary negation (constant width)
    *
    * @return a $coll equal to zero minus this $coll shifted right by one.
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_-% : UInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_-(implicit sourceInfo: SourceInfo): UInt = _impl_unary_-

  /** @group SourceInfoTransformMacro */
  def do_unary_-%(implicit sourceInfo: SourceInfo): UInt = _impl_unary_-%

  override def do_+(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_+(that)
  override def do_-(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_-(that)
  override def do_/(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_/(that)
  override def do_%(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_%(that)
  override def do_*(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_*(that)

  /** Multiplication operator
    *
    * @param that a hardware [[SInt]]
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  final def *(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_*(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_*(that)

  /** Addition operator (expanding width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def +&(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Addition operator (constant width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidth
    * @group Arithmetic
    */
  final def +%(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (increasing width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def -&(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (constant width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidth
    * @group Arithmetic
    */
  final def -%(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_+&(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_+&(that)

  /** @group SourceInfoTransformMacro */
  def do_+%(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_+%(that)

  /** @group SourceInfoTransformMacro */
  def do_-&(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_-&(that)

  /** @group SourceInfoTransformMacro */
  def do_-%(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_-%(that)

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def &(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def |(that: UInt): UInt = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def ^(that: UInt): UInt = macro SourceInfoTransform.thatArg

  //  override def abs: UInt = macro SourceInfoTransform.noArgDummy
  def do_abs(implicit sourceInfo: SourceInfo): UInt = _absImpl

  /** @group SourceInfoTransformMacro */
  def do_&(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_&(that)

  /** @group SourceInfoTransformMacro */
  def do_|(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_|(that)

  /** @group SourceInfoTransformMacro */
  def do_^(that: UInt)(implicit sourceInfo: SourceInfo): UInt = _impl_^(that)

  /** @group SourceInfoTransformMacro */
  override def do_unary_~(implicit sourceInfo: SourceInfo): UInt = _impl_unary_~

  // REVIEW TODO: Can these be defined on Bits?
  /** Or reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll or'd together
    * @group Bitwise
    */
  final def orR: Bool = macro SourceInfoTransform.noArg

  /** And reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll and'd together
    * @group Bitwise
    */
  final def andR: Bool = macro SourceInfoTransform.noArg

  /** Exclusive or (xor) reduction operator
    *
    * @return a hardware [[Bool]] resulting from every bit of this $coll xor'd together
    * @group Bitwise
    */
  final def xorR: Bool = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_orR(implicit sourceInfo: SourceInfo): Bool = _orRImpl

  /** @group SourceInfoTransformMacro */
  def do_andR(implicit sourceInfo: SourceInfo): Bool = _andRImpl

  /** @group SourceInfoTransformMacro */
  def do_xorR(implicit sourceInfo: SourceInfo): Bool = _xorRImpl

  override def do_<(that:  UInt)(implicit sourceInfo: SourceInfo): Bool = _impl_<(that)
  override def do_>(that:  UInt)(implicit sourceInfo: SourceInfo): Bool = _impl_>(that)
  override def do_<=(that: UInt)(implicit sourceInfo: SourceInfo): Bool = _impl_<=(that)
  override def do_>=(that: UInt)(implicit sourceInfo: SourceInfo): Bool = _impl_>=(that)

  /** Dynamic not equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is not equal to `that`
    * @group Comparison
    */
  final def =/=(that: UInt): Bool = macro SourceInfoTransform.thatArg

  /** Dynamic equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is equal to `that`
    * @group Comparison
    */
  final def ===(that: UInt): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_=/=(that: UInt)(implicit sourceInfo: SourceInfo): Bool = _impl_=/=(that)

  /** @group SourceInfoTransformMacro */
  def do_===(that: UInt)(implicit sourceInfo: SourceInfo): Bool = _impl_===(that)

  /** Unary not
    *
    * @return a hardware [[Bool]] asserted if this $coll equals zero
    * @group Bitwise
    */
  final def unary_! : Bool = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_!(implicit sourceInfo: SourceInfo): Bool = _impl_unary_!

  override def do_<<(that: Int)(implicit sourceInfo:    SourceInfo): UInt = _impl_<<(that)
  override def do_<<(that: BigInt)(implicit sourceInfo: SourceInfo): UInt = _impl_<<(that)
  override def do_<<(that: UInt)(implicit sourceInfo:   SourceInfo): UInt = _impl_<<(that)

  override def do_>>(that: Int)(implicit sourceInfo:    SourceInfo): UInt = _impl_>>(that)
  override def do_>>(that: BigInt)(implicit sourceInfo: SourceInfo): UInt = _impl_>>(that)
  override def do_>>(that: UInt)(implicit sourceInfo:   SourceInfo): UInt = _impl_>>(that)

  /**
    * Circular shift to the left
    * @param that number of bits to rotate
    * @return UInt of same width rotated left n bits
    */
  final def rotateLeft(that: Int): UInt = macro SourceInfoWhiteboxTransform.thatArg

  def do_rotateLeft(n: Int)(implicit sourceInfo: SourceInfo): UInt = _rotateLeftImpl(n)

  /**
    * Circular shift to the right
    * @param that number of bits to rotate
    * @return UInt of same width rotated right n bits
    */
  final def rotateRight(that: Int): UInt = macro SourceInfoWhiteboxTransform.thatArg

  def do_rotateRight(n: Int)(implicit sourceInfo: SourceInfo): UInt = _rotateRightImpl(n)

  final def rotateRight(that: UInt): UInt = macro SourceInfoWhiteboxTransform.thatArg

  def do_rotateRight(n: UInt)(implicit sourceInfo: SourceInfo): UInt = _rotateRightImpl(n)

  final def rotateLeft(that: UInt): UInt = macro SourceInfoWhiteboxTransform.thatArg

  def do_rotateLeft(n: UInt)(implicit sourceInfo: SourceInfo): UInt = _rotateLeftImpl(n)

  /** Conditionally set or clear a bit
    *
    * @param off a dynamic offset
    * @param dat set if true, clear if false
    * @return a hrdware $coll with bit `off` set or cleared based on the value of `dat`
    * $unchangedWidth
    */
  final def bitSet(off: UInt, dat: Bool): UInt = macro UIntTransform.bitset

  /** @group SourceInfoTransformMacro */
  def do_bitSet(off: UInt, dat: Bool)(implicit sourceInfo: SourceInfo): UInt = _bitSetImpl(off, dat)

  // TODO: this eventually will be renamed as toSInt, once the existing toSInt
  // completes its deprecation phase.
  /** Zero extend as [[SInt]]
    *
    * @return an [[SInt]] equal to this $coll with an additional zero in its most significant bit
    * @note The width of the returned [[SInt]] is `width of this` + `1`.
    */
  final def zext: SInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_zext(implicit sourceInfo: SourceInfo): SInt = _zextImpl

  override def do_asSInt(implicit sourceInfo: SourceInfo): SInt = _asSIntImpl
}

/** A data type for signed integers, represented as a binary bitvector. Defines arithmetic operations between other
  * integer types.
  *
  * @define coll [[SInt]]
  * @define numType $coll
  * @define expandingWidth @note The width of the returned $coll is `width of this` + `1`.
  * @define constantWidth  @note The width of the returned $coll is unchanged, i.e., `width of this`.
  */
sealed class SInt private[chisel3] (width: Width) extends Bits(width) with SIntImpl {

  /** Unary negation (constant width)
    *
    * @return a hardware $coll equal to zero minus this $coll
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_- : SInt = macro SourceInfoTransform.noArg

  /** Unary negation (constant width)
    *
    * @return a hardware $coll equal to zero minus `this` shifted right by one
    * $constantWidth
    * @group Arithmetic
    */
  final def unary_-% : SInt = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_unary_-(implicit sourceInfo: SourceInfo): SInt = _impl_unary_-

  /** @group SourceInfoTransformMacro */
  def do_unary_-%(implicit sourceInfo: SourceInfo): SInt = _impl_unary_-%

  /** add (default - no growth) operator */
  override def do_+(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_+(that)

  /** subtract (default - no growth) operator */
  override def do_-(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_-(that)
  override def do_*(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_*(that)
  override def do_/(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_/(that)
  override def do_%(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_%(that)

  /** Multiplication operator
    *
    * @param that a hardware $coll
    * @return the product of this $coll and `that`
    * $sumWidth
    * $singleCycleMul
    * @group Arithmetic
    */
  final def *(that: UInt): SInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_*(that: UInt)(implicit sourceInfo: SourceInfo): SInt = _impl_*(that)

  /** Addition operator (expanding width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def +&(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Addition operator (constant width)
    *
    * @param that a hardware $coll
    * @return the sum of this $coll and `that` shifted right by one
    * $maxWidth
    * @group Arithmetic
    */
  final def +%(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (increasing width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that`
    * $maxWidthPlusOne
    * @group Arithmetic
    */
  final def -&(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Subtraction operator (constant width)
    *
    * @param that a hardware $coll
    * @return the difference of this $coll less `that` shifted right by one
    * $maxWidth
    * @group Arithmetic
    */
  final def -%(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_+&(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_+&(that)

  /** @group SourceInfoTransformMacro */
  def do_+%(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_+%(that)

  /** @group SourceInfoTransformMacro */
  def do_-&(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_-&(that)

  /** @group SourceInfoTransformMacro */
  def do_-%(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_-%(that)

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def &(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def |(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * $maxWidth
    * @group Bitwise
    */
  final def ^(that: SInt): SInt = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_&(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_&(that)

  /** @group SourceInfoTransformMacro */
  def do_|(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_|(that)

  /** @group SourceInfoTransformMacro */
  def do_^(that: SInt)(implicit sourceInfo: SourceInfo): SInt = _impl_^(that)

  /** @group SourceInfoTransformMacro */
  override def do_unary_~(implicit sourceInfo: SourceInfo): SInt = _impl_unary_~

  override def do_<(that:  SInt)(implicit sourceInfo: SourceInfo): Bool = _impl_<(that)
  override def do_>(that:  SInt)(implicit sourceInfo: SourceInfo): Bool = _impl_>(that)
  override def do_<=(that: SInt)(implicit sourceInfo: SourceInfo): Bool = _impl_<=(that)
  override def do_>=(that: SInt)(implicit sourceInfo: SourceInfo): Bool = _impl_>=(that)

  /** Dynamic not equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is not equal to `that`
    * @group Comparison
    */
  final def =/=(that: SInt): Bool = macro SourceInfoTransform.thatArg

  /** Dynamic equals operator
    *
    * @param that a hardware $coll
    * @return a hardware [[Bool]] asserted if this $coll is equal to `that`
    * @group Comparison
    */
  final def ===(that: SInt): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_=/=(that: SInt)(implicit sourceInfo: SourceInfo): Bool = _impl_=/=(that)

  /** @group SourceInfoTransformMacro */
  def do_===(that: SInt)(implicit sourceInfo: SourceInfo): Bool = _impl_===(that)

  def do_abs(implicit sourceInfo: SourceInfo): SInt = _absImpl

  override def do_<<(that: Int)(implicit sourceInfo:    SourceInfo): SInt = _impl_<<(that)
  override def do_<<(that: BigInt)(implicit sourceInfo: SourceInfo): SInt = _impl_<<(that)
  override def do_<<(that: UInt)(implicit sourceInfo:   SourceInfo): SInt = _impl_<<(that)

  override def do_>>(that: Int)(implicit sourceInfo:    SourceInfo): SInt = _impl_>>(that)
  override def do_>>(that: BigInt)(implicit sourceInfo: SourceInfo): SInt = _impl_>>(that)
  override def do_>>(that: UInt)(implicit sourceInfo:   SourceInfo): SInt = _impl_>>(that)

  override def do_asSInt(implicit sourceInfo: SourceInfo): SInt = _asSIntImpl
}

sealed trait Reset extends ResetImpl with ToBoolable {

  /** Casts this $coll to an [[AsyncReset]] */
  final def asAsyncReset: AsyncReset = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo): AsyncReset

  /** Casts this $coll to a [[Disable]] */
  final def asDisable: Disable = macro SourceInfoWhiteboxTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asDisable(implicit sourceInfo: SourceInfo): Disable = _asDisableImpl
}

object Reset {
  def apply(): Reset = new ResetType
}

/** "Abstract" Reset Type inferred in FIRRTL to either [[AsyncReset]] or [[Bool]]
  *
  * @note This shares a common interface with [[AsyncReset]] and [[Bool]] but is not their actual
  * super type due to Bool inheriting from abstract class UInt
  */
final class ResetType(private[chisel3] val width: Width = Width(1)) extends Reset with ResetTypeImpl with ToBoolable {

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo): AsyncReset = _asAsyncResetImpl

  /** @group SourceInfoTransformMacro */
  def do_asBool(implicit sourceInfo: SourceInfo): Bool = _asBoolImpl

  /** @group SourceInfoTransformMacro */
  def do_toBool(implicit sourceInfo: SourceInfo): Bool = do_asBool
}

object AsyncReset {
  def apply(): AsyncReset = new AsyncReset
}

/** Data type representing asynchronous reset signals
  *
  * These signals are similar to [[Clock]]s in that they must be glitch-free for proper circuit
  * operation. [[Reg]]s defined with the implicit reset being an [[AsyncReset]] will be
  * asychronously reset registers.
  */
sealed class AsyncReset(private[chisel3] val width: Width = Width(1)) extends AsyncResetImpl with Reset {
  override def toString: String = stringAccessor("AsyncReset")

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo): AsyncReset = _asAsyncResetImpl

  /** @group SourceInfoTransformMacro */
  def do_asBool(implicit sourceInfo: SourceInfo): Bool = _asBoolImpl

  /** @group SourceInfoTransformMacro */
  def do_toBool(implicit sourceInfo: SourceInfo): Bool = _asBoolImpl
}

// REVIEW TODO: Why does this extend UInt and not Bits? Does defining airth
// operations on a Bool make sense?
/** A data type for booleans, defined as a single bit indicating true or false.
  *
  * @define coll [[Bool]]
  * @define numType $coll
  */
sealed class Bool() extends UInt(1.W) with BoolImpl with Reset {

  // REVIEW TODO: Why does this need to exist and have different conventions
  // than Bits?

  /** Bitwise and operator
    *
    * @param that a hardware $coll
    * @return the bitwise and of  this $coll and `that`
    * @group Bitwise
    */
  final def &(that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** Bitwise or operator
    *
    * @param that a hardware $coll
    * @return the bitwise or of this $coll and `that`
    * @group Bitwise
    */
  final def |(that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** Bitwise exclusive or (xor) operator
    *
    * @param that a hardware $coll
    * @return the bitwise xor of this $coll and `that`
    * @group Bitwise
    */
  final def ^(that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_&(that: Bool)(implicit sourceInfo: SourceInfo): Bool = _impl_&(that)

  /** @group SourceInfoTransformMacro */
  def do_|(that: Bool)(implicit sourceInfo: SourceInfo): Bool = _impl_|(that)

  /** @group SourceInfoTransformMacro */
  def do_^(that: Bool)(implicit sourceInfo: SourceInfo): Bool = _impl_^(that)

  /** @group SourceInfoTransformMacro */
  override def do_unary_~(implicit sourceInfo: SourceInfo): Bool = _impl_unary_~

  /** Logical or operator
    *
    * @param that a hardware $coll
    * @return the logical or of this $coll and `that`
    * @note this is equivalent to [[Bool!.|(that:chisel3\.Bool)* Bool.|)]]
    * @group Logical
    */
  def ||(that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_||(that: Bool)(implicit sourceInfo: SourceInfo): Bool = _impl_||(that)

  /** Logical and operator
    *
    * @param that a hardware $coll
    * @return the logical and of this $coll and `that`
    * @note this is equivalent to [[Bool!.&(that:chisel3\.Bool)* Bool.&]]
    * @group Logical
    */
  def &&(that: Bool): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_&&(that: Bool)(implicit sourceInfo: SourceInfo): Bool = _impl_&&(that)

  override def do_asBool(implicit sourceInfo: SourceInfo): Bool = _asBoolImpl

  /** Reinterprets this $coll as a clock */
  def asClock: Clock = macro SourceInfoTransform.noArg

  /** @group SourceInfoTransformMacro */
  def do_asClock(implicit sourceInfo: SourceInfo): Clock = _asClockImpl

  /** @group SourceInfoTransformMacro */
  def do_asAsyncReset(implicit sourceInfo: SourceInfo): AsyncReset = _asAsyncResetImpl
}
