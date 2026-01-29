// SPDX-License-Identifier: Apache-2.0

package chisel3.ltl

import chisel3._
import chisel3.layer.{block, Layer}
import chisel3.util.circt._
import chisel3.experimental.SourceInfo

/** An opaque sequence returned by an intrinsic.
  *
  * Due to the lack of opaque user-defined types in FIRRTL, the Linear Temporal
  * Logic (LTL) intrinsics operate on `Bool` values instead of a more desirable
  * `Sequence` type. To prevent abuse, the Chisel API here wraps these `Bool`s
  * in an opaque type over which we can provide a safe interface that cannot be
  * used to construct invalid IR.
  */
private case class OpaqueSequence(private[ltl] val inner: Bool) extends Sequence

/** An opaque property returned by an intrinsic.
  *
  * See `OpaqueSequence` for details.
  */
private case class OpaqueProperty(private[ltl] val inner: Bool) extends Property

/** A single item that may be used in the `Sequence(...)` convenience
  * constructor for sequences. These atoms may either be `Sequence`s themselves,
  * like `a` or `a and b`, or a `DelayAtom`, like `Delay`. Together they enable
  * a shorthand notation for sequences like:
  *
  * `Sequence(a, Delay(), b, Delay(2), c, Delay(3, 9), d, Delay(4, None), e)`.
  */
sealed trait SequenceAtom

/** A delay atom for the `Sequence(...)` convenience constructor. * */
private case class DelayAtom(val min: Int, val max: Option[Int]) extends SequenceAtom

/** The delay atoms available to users. Can be interleaved with actual sequences
  * in `Sequence(...)`. See `SequenceAtom` for details.
  */
object Delay {
  def apply():                             SequenceAtom = DelayAtom(1, Some(1))
  def apply(delay: Int):                   SequenceAtom = DelayAtom(delay, Some(delay))
  def apply(min:   Int, max: Int):         SequenceAtom = DelayAtom(min, Some(max))
  def apply(min:   Int, max: Option[Int]): SequenceAtom = DelayAtom(min, max)
}

/** A Linear Temporal Logic (LTL) sequence. */
sealed trait Sequence extends Property with SequenceIntf {

  protected def _delayImpl(delay: Int = 1)(implicit sourceInfo: SourceInfo): Sequence = Sequence._delay(this, delay)

  protected def _delayRangeImpl(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._delayRange(this, min, max)

  protected def _delayAtLeastImpl(delay: Int)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._delayAtLeast(this, delay)

  protected def _concatImpl(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = Sequence._concat(this, other)

  protected def _repeatImpl(n: Int = 1)(implicit sourceInfo: SourceInfo): Sequence = Sequence._repeat(this, n)

  protected def _repeatRangeImpl(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._repeatRange(this, min, max)

  protected def _repeatAtLeastImpl(n: Int)(implicit sourceInfo: SourceInfo): Sequence = Sequence._repeatAtLeast(this, n)

  protected def _gotoRepeatImpl(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._gotoRepeat(this, min, max)

  protected def _nonConsecutiveRepeatImpl(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._nonConsecutiveRepeat(this, min, max)

  protected def _andSeqImpl(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = Sequence._and(this, other)

  protected def _orSeqImpl(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = Sequence._or(this, other)

  protected def _intersectSeqImpl(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._intersect(this, other)

  protected def _untilSeqImpl(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._until(this, other)

  override protected def _clockImpl(clock: Clock)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._clock(this, clock)

  protected def _implicationImpl(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    Property.implication(this, prop)

  protected def _implicationNonOverlappingImpl(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    Property.implicationNonOverlapping(this, prop)

  // Convenience operators.

  protected def _impl_|->(prop: Property)(implicit sourceInfo: SourceInfo): Property = this.implication(prop)

  protected def _impl_|=>(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    this.implicationNonOverlapping(prop)

  protected def _impl_###(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = this.concat(other.delay())

  protected def _impl_##*(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence =
    this.concat(other.delayAtLeast(0))

  protected def _impl_##+(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence =
    this.concat(other.delayAtLeast(1))
}

/** Prefix-style utilities to work with sequences.
  *
  * This object exposes the primary API to create and compose sequences from
  * booleans and shorter sequences.
  */
object Sequence extends Sequence$Intf {

  implicit class BoolSequence(val inner: Bool) extends Sequence with SequenceAtom

  protected def _delay(seq: Sequence, delay: Int = 1)(implicit sourceInfo: SourceInfo): Sequence =
    OpaqueSequence(LTLDelayIntrinsic(delay, Some(0))(seq.inner))

  protected def _delayRange(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = {
    require(min <= max)
    OpaqueSequence(LTLDelayIntrinsic(min, Some(max - min))(seq.inner))
  }

  protected def _delayAtLeast(seq: Sequence, delay: Int)(implicit sourceInfo: SourceInfo): Sequence =
    OpaqueSequence(LTLDelayIntrinsic(delay, None)(seq.inner))

  protected def _concat(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueSequence(LTLConcatIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  protected def _repeat(seq: Sequence, n: Int = 1)(implicit sourceInfo: SourceInfo): Sequence = {
    require(0 < n)
    OpaqueSequence(LTLRepeatIntrinsic(n, Some(0))(seq.inner))
  }

  protected def _repeatRange(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = {
    require(min <= max)
    OpaqueSequence(LTLRepeatIntrinsic(min, Some(max - min))(seq.inner))
  }

  protected def _repeatAtLeast(seq: Sequence, n: Int)(implicit sourceInfo: SourceInfo): Sequence = {
    require(0 < n)
    OpaqueSequence(LTLRepeatIntrinsic(n, None)(seq.inner))
  }

  protected def _gotoRepeat(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = {
    require(0 <= min && min <= max)
    OpaqueSequence(LTLGoToRepeatIntrinsic(min, max - min)(seq.inner))
  }

  protected def _nonConsecutiveRepeat(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = {
    require(0 <= min && min <= max)
    OpaqueSequence(LTLNonConsecutiveRepeatIntrinsic(min, max - min)(seq.inner))
  }

  protected def _and(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueSequence(LTLAndIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  protected def _or(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueSequence(LTLOrIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  protected def _intersect(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueSequence(LTLIntersectIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  protected def _until(arg0: Sequence, arg1: Sequence)(implicit sourceInfo: SourceInfo): Sequence =
    OpaqueSequence(LTLUntilIntrinsic(arg0.inner, arg1.inner))

  protected def _clock(seq: Sequence, clock: Clock)(implicit sourceInfo: SourceInfo): Sequence =
    OpaqueSequence(LTLClockIntrinsic(seq.inner, clock))

  protected def _apply(atoms: SequenceAtom*)(implicit sourceInfo: SourceInfo): Sequence = {
    require(atoms.nonEmpty)
    def needDelayTail = {
      require(
        atoms.tail.nonEmpty,
        "`Delay` operator in `Sequence(...)` must be followed by an item to be delayed"
      )
      apply(atoms.tail: _*)
    }
    atoms.head match {
      case seq: Sequence if atoms.tail.nonEmpty => seq.concat(apply(atoms.tail: _*))
      case seq: Sequence                        => seq
      case DelayAtom(min, None)      => needDelayTail.delayAtLeast(min)
      case DelayAtom(min, Some(max)) => needDelayTail.delayRange(min, max)
    }
  }
}

/** A Linear Temporal Logic (LTL) property. */
sealed trait Property extends PropertyIntf {

  /** The underlying `Bool` that is returned and accepted by the LTL
    * intrinsics.
    */
  private[ltl] def inner: Bool

  protected def _notImpl(implicit sourceInfo: SourceInfo): Property = Property._not(this)

  protected def _eventuallyImpl(implicit sourceInfo: SourceInfo): Property = Property._eventually(this)

  protected def _andPropImpl(other: Property)(implicit sourceInfo: SourceInfo): Property = Property._and(this, other)

  protected def _orPropImpl(other: Property)(implicit sourceInfo: SourceInfo): Property = Property._or(this, other)

  protected def _intersectPropImpl(other: Property)(implicit sourceInfo: SourceInfo): Property =
    Property._intersect(this, other)

  protected def _untilPropImpl(other: Property)(implicit sourceInfo: SourceInfo): Property =
    Property._until(this, other)

  protected def _clockImpl(clock: Clock)(implicit sourceInfo: SourceInfo): Property = Property._clock(this, clock)
}

/** Prefix-style utilities to work with properties.
  *
  * This object exposes the primary API to create and compose properties from
  * booleans, sequences, and other properties.
  */
object Property extends Property$Intf {

  protected def _not(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    OpaqueProperty(LTLNotIntrinsic(prop.inner))

  protected def _implication(seq: Sequence, prop: Property)(implicit sourceInfo: SourceInfo): Property =
    OpaqueProperty(LTLImplicationIntrinsic(seq.inner, prop.inner))

  protected def _implicationNonOverlapping(seq: Sequence, prop: Property)(implicit sourceInfo: SourceInfo): Property = {
    import Sequence.BoolSequence
    implication(seq.concat(true.B.delay(1)), prop)
  }

  protected def _eventually(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    OpaqueProperty(LTLEventuallyIntrinsic(prop.inner))

  protected def _and(arg0: Property, argN: Property*)(implicit sourceInfo: SourceInfo): Property = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueProperty(LTLAndIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  protected def _or(arg0: Property, argN: Property*)(implicit sourceInfo: SourceInfo): Property = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueProperty(LTLOrIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  protected def _intersect(arg0: Property, argN: Property*)(implicit sourceInfo: SourceInfo): Property = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueProperty(LTLIntersectIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  protected def _until(arg0: Property, arg1: Property)(implicit sourceInfo: SourceInfo): Property =
    OpaqueProperty(LTLUntilIntrinsic(arg0.inner, arg1.inner))

  protected def _clock(prop: Property, clock: Clock)(implicit sourceInfo: SourceInfo): Property =
    OpaqueProperty(LTLClockIntrinsic(prop.inner, clock))
}

/** The base class for the `AssertProperty`, `AssumeProperty`, and
  * `CoverProperty` verification constructs.
  */
sealed abstract class AssertPropertyLike(defaultLayer: Layer) extends AssertPropertyLikeIntf {

  protected def _applyImpl(
    prop:    => Property,
    clock:   Option[Clock] = Module.clockOption,
    disable: Option[Disable] = Module.disableOption,
    label:   Option[String] = None
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = block(defaultLayer, skipIfAlreadyInBlock = true, skipIfLayersEnabled = true) {
    val _prop = prop // evaluate prop expression once
    val clocked = clock.fold(_prop)(_prop.clock(_))
    createIntrinsic(label)(sourceInfo)(clocked.inner, disable.map(!_.value))
  }

  protected def _applyCondImpl(
    cond: Bool
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    apply(Sequence.BoolSequence(cond))
  }

  protected def _applyCondLabelImpl(
    cond:  Bool,
    label: String
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    apply(Sequence.BoolSequence(cond), label = Some(label))
  }

  protected def _applyCondClockDisableLabelImpl(
    cond:    Bool,
    clock:   Clock,
    disable: Disable,
    label:   String
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    apply(Sequence.BoolSequence(cond), Some(clock), Some(disable), Some(label))
  }

  protected def createIntrinsic(label: Option[String])(implicit sourceInfo: SourceInfo): (Bool, Option[Bool]) => Unit
}

/** Assert that a property holds.
  *
  * Use like `AssertProperty(p)`. See `AssertPropertyLike.apply` for optional
  * clock, disable_iff, and label parameters.
  */
object AssertProperty extends AssertPropertyLike(defaultLayer = layers.Verification.Assert) {
  protected def createIntrinsic(label: Option[String])(implicit sourceInfo: SourceInfo) = VerifAssertIntrinsic(label)
}

/** Assume that a property holds.
  *
  * Use like `AssumeProperty(p)`. See `AssertPropertyLike.apply` for optional
  * clock, disable_iff, and label parameters.
  */
object AssumeProperty extends AssertPropertyLike(defaultLayer = layers.Verification.Assume) {
  protected def createIntrinsic(label: Option[String])(implicit sourceInfo: SourceInfo) = VerifAssumeIntrinsic(label)
}

/** Cover that a property holds.
  *
  * Use like `CoverProperty(p)`. See `AssertPropertyLike.apply` for optional
  * clock, disable_iff, and label parameters.
  */
object CoverProperty extends AssertPropertyLike(defaultLayer = layers.Verification.Cover) {
  protected def createIntrinsic(label: Option[String])(implicit sourceInfo: SourceInfo) = VerifCoverIntrinsic(label)
}

/** Require that a property holds as a pre-condition of a contract. Behaves like
  * an `AssertProperty` if used outside of a contract. When used inside of a
  * contract, the behavior differs depending on how the contract is used in a
  * formal proof:
  *
  * - During a proof that the contract is upheld by the surrounding circuit, the
  *   property given to `RequireProperty` is assumed to hold.
  * - During a larger proof where the contract is already assumed to be proven,
  *   the property given to `RequireProperty` is asserted to hold.
  *
  * Use like `RequireProperty(p)`. See `AssertPropertyLike.apply` for optional
  * clock, disable_iff, and label parameters.
  */
object RequireProperty extends AssertPropertyLike(defaultLayer = layers.Verification.Assume) {
  protected def createIntrinsic(label: Option[String])(implicit sourceInfo: SourceInfo) = VerifRequireIntrinsic(label)
}

/** Ensure that a property holds as a post-condition of a contract. Behaves like
  * an `AssertProperty` if used outside of a contract. When used inside of a
  * contract, the behavior differs depending on how the contract is used in a
  * formal proof:
  *
  * - During a proof that the contract is upheld by the surrounding circuit, the
  *   property given to `EnsureProperty` is asserted to hold.
  * - During a larger proof where the contract is already assumed to be proven,
  *   the property given to `EnsureProperty` is assumed to hold.
  *
  * Use like `EnsureProperty(p)`. See `AssertPropertyLike.apply` for optional
  * clock, disable_iff, and label parameters.
  */
object EnsureProperty extends AssertPropertyLike(defaultLayer = layers.Verification.Assert) {
  protected def createIntrinsic(label: Option[String])(implicit sourceInfo: SourceInfo) = VerifEnsureIntrinsic(label)
}
