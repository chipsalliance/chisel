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

  /** See `Sequence.delay`. */
  protected def _delayImpl(delay: Int = 1)(implicit sourceInfo: SourceInfo): Sequence = Sequence._delay(this, delay)

  /** See `Sequence.delayRange`. */
  protected def _delayRangeImpl(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._delayRange(this, min, max)

  /** See `Sequence.delayAtLeast`. */
  protected def _delayAtLeastImpl(delay: Int)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._delayAtLeast(this, delay)

  /** See `Sequence.concat`. */
  protected def _concatImpl(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = Sequence._concat(this, other)

  /** See `Sequence.repeat`. */
  protected def _repeatImpl(n: Int = 1)(implicit sourceInfo: SourceInfo): Sequence = Sequence._repeat(this, n)

  /** See `Sequence.repeatRange`. */
  protected def _repeatRangeImpl(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._repeatRange(this, min, max)

  /** See `Sequence.repeatAtLeast`. */
  protected def _repeatAtLeastImpl(n: Int)(implicit sourceInfo: SourceInfo): Sequence = Sequence._repeatAtLeast(this, n)

  /** See `Sequence.gotoRepeat`. */
  protected def _gotoRepeatImpl(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._gotoRepeat(this, min, max)

  /** See `Sequence.nonConsecutiveRepeat`. */
  protected def _nonConsecutiveRepeatImpl(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._nonConsecutiveRepeat(this, min, max)

  /** See `Sequence.and`. */
  protected def _andSeqImpl(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = Sequence._and(this, other)

  /** See `Sequence.or`. */
  protected def _orSeqImpl(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = Sequence._or(this, other)

  /** See `Sequence.intersect`. */
  protected def _intersectSeqImpl(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._intersect(this, other)

  /** See `Sequence.until`. */
  protected def _untilSeqImpl(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._until(this, other)

  /** See `Sequence.clock`. */
  override protected def _clockImpl(clock: Clock)(implicit sourceInfo: SourceInfo): Sequence =
    Sequence._clock(this, clock)

  /** See `Property.implication`. */
  protected def _implicationImpl(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    Property.implication(this, prop)

  /** See `Property.implication`. */
  protected def _implicationNonOverlappingImpl(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    Property.implicationNonOverlapping(this, prop)

  // Convenience operators.

  /** Equivalent to `|->` in SVA. */
  protected def _impl_|->(prop: Property)(implicit sourceInfo: SourceInfo): Property = this.implication(prop)

  /** Equivalent to `|=>` in SVA. */
  protected def _impl_|=>(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    this.implicationNonOverlapping(prop)

  /** Equivalent to `a ##1 b` in SVA. */
  protected def _impl_###(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = this.concat(other.delay())

  /** Equivalent to `a ##[*] b` in SVA. */
  protected def _impl_##*(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence =
    this.concat(other.delayAtLeast(0))

  /** Equivalent to `a ##[+] b` in SVA. */
  protected def _impl_##+(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence =
    this.concat(other.delayAtLeast(1))
}

/** Prefix-style utilities to work with sequences.
  *
  * This object exposes the primary API to create and compose sequences from
  * booleans and shorter sequences.
  */
object Sequence extends Sequence$Intf {

  /** Implicitly wraps a `Bool` and allows it to be used as a sequence or
    * property. Use via `import chisel3.ltl.Sequence.BoolSequence`.
    */
  implicit class BoolSequence(val inner: Bool) extends Sequence with SequenceAtom

  /** Delay a sequence by a fixed number of cycles. Equivalent to `##delay` in
    * SVA.
    */
  protected def _delay(seq: Sequence, delay: Int = 1)(implicit sourceInfo: SourceInfo): Sequence =
    OpaqueSequence(LTLDelayIntrinsic(delay, Some(0))(seq.inner))

  /** Delay a sequence by a bounded range of cycles. Equivalent to `##[min:max]`
    * in SVA.
    */
  protected def _delayRange(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = {
    require(min <= max)
    OpaqueSequence(LTLDelayIntrinsic(min, Some(max - min))(seq.inner))
  }

  /** Delay a sequence by an unbounded range of cycles. Equivalent to
    * `##[delay:$]` in SVA.
    */
  protected def _delayAtLeast(seq: Sequence, delay: Int)(implicit sourceInfo: SourceInfo): Sequence =
    OpaqueSequence(LTLDelayIntrinsic(delay, None)(seq.inner))

  /** Concatenate multiple sequences. Equivalent to
    * `arg0 ##0 arg1 ##0 ... ##0 argN` in SVA.
    */
  protected def _concat(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueSequence(LTLConcatIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  /** Repeat a sequence a fixed number of consecutive times. Equivalent to `s[n]` in
    * SVA.
    */
  protected def _repeat(seq: Sequence, n: Int = 1)(implicit sourceInfo: SourceInfo): Sequence = {
    require(0 < n)
    OpaqueSequence(LTLRepeatIntrinsic(n, Some(0))(seq.inner))
  }

  /** Repeat a sequence by a bounded range of consecutive times. Equivalent to `s[min:max]`
    * in SVA.
    */
  protected def _repeatRange(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = {
    require(min <= max)
    OpaqueSequence(LTLRepeatIntrinsic(min, Some(max - min))(seq.inner))
  }

  /** Repeat a sequence by an unbounded range of consecutive times. Equivalent to
    * `s[n:$]` in SVA.
    */
  protected def _repeatAtLeast(seq: Sequence, n: Int)(implicit sourceInfo: SourceInfo): Sequence = {
    require(0 < n)
    OpaqueSequence(LTLRepeatIntrinsic(n, None)(seq.inner))
  }

  /** GoTo-style repitition of a sequence a fixed number of non-consecutive times,
    * where the final evaluation of the sequence must hold, e.g.
    * a !b b b !b !b b c represents a matching observation of `gotoRepeat(b, 1, 3)`,
    * but a !b b b !b !b b !b c does not. Equivalent to `s[->min:max]` in SVA.
    */
  protected def _gotoRepeat(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = {
    require(0 <= min && min <= max)
    OpaqueSequence(LTLGoToRepeatIntrinsic(min, max - min)(seq.inner))
  }

  /** Repeat a sequence a fixed number of non-consecutive times,
    * where the final evaluation of the sequence does not have to hold, e.g.
    * both a !b b b !b !b b c and a !b b b !b !b b !b c represent matching
    * observations of `nonConsecutiveRepeat(b, 1, 3)`. Equivalent to `s[=min:max]` in SVA.
    */
  protected def _nonConsecutiveRepeat(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = {
    require(0 <= min && min <= max)
    OpaqueSequence(LTLNonConsecutiveRepeatIntrinsic(min, max - min)(seq.inner))
  }

  /** Form the conjunction of two sequences. Equivalent to
    * `arg0 and arg1 and ... and argN` in SVA.
    */
  protected def _and(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueSequence(LTLAndIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  /** Form the disjunction of two sequences. Equivalent to
    * `arg0 or arg1 or ... or argN` in SVA.
    */
  protected def _or(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueSequence(LTLOrIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  /** Form the conjunction of two sequences, where the start and end
    * times of both sequences must be identical. Equivalent to
    * `arg0 intersect arg1 intersect ... intersect argN` in SVA.
    */
  protected def _intersect(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueSequence(LTLIntersectIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  /** Check that a sequence holds untile another sequence holds.
    * This operator is weak: the property will hold even if input always
    * holds and condition never holds.
    */
  protected def _until(arg0: Sequence, arg1: Sequence)(implicit sourceInfo: SourceInfo): Sequence =
    OpaqueSequence(LTLUntilIntrinsic(arg0.inner, arg1.inner))

  /** Specify a `clock` relative to which all cycle delays within `seq` are
    * specified. Equivalent to `@(posedge clock) seq` in SVA.
    */
  protected def _clock(seq: Sequence, clock: Clock)(implicit sourceInfo: SourceInfo): Sequence =
    OpaqueSequence(LTLClockIntrinsic(seq.inner, clock))

  /** Convenience constructor for sequences. Allows for the following syntax:
    *
    * `Sequence(a, Delay(), b, Delay(2), c, Delay(3, 9), d, Delay(4, None), e)`
    *
    * See `SequenceAtom` for more details.
    */
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

  /** See `Property.not`. */
  protected def _notImpl(implicit sourceInfo: SourceInfo): Property = Property._not(this)

  /** See `Property.eventually`. */
  protected def _eventuallyImpl(implicit sourceInfo: SourceInfo): Property = Property._eventually(this)

  /** See `Property.and`. */
  protected def _andPropImpl(other: Property)(implicit sourceInfo: SourceInfo): Property = Property._and(this, other)

  /** See `Property.or`. */
  protected def _orPropImpl(other: Property)(implicit sourceInfo: SourceInfo): Property = Property._or(this, other)

  /** See `Property.intersect`. */
  protected def _intersectPropImpl(other: Property)(implicit sourceInfo: SourceInfo): Property =
    Property._intersect(this, other)

  /** See `Property.until`. */
  protected def _untilPropImpl(other: Property)(implicit sourceInfo: SourceInfo): Property =
    Property._until(this, other)

  /** See `Property.clock`. */
  protected def _clockImpl(clock: Clock)(implicit sourceInfo: SourceInfo): Property = Property._clock(this, clock)
}

/** Prefix-style utilities to work with properties.
  *
  * This object exposes the primary API to create and compose properties from
  * booleans, sequences, and other properties.
  */
object Property extends Property$Intf {

  /** Negate a property. Equivalent to `not prop` in SVA. */
  protected def _not(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    OpaqueProperty(LTLNotIntrinsic(prop.inner))

  /** Precondition the checking of a property (the consequent) on a sequence
    * (the antecedent). Equivalent to the overlapping implication `seq |-> prop`
    * in SVA.
    */
  protected def _implication(seq: Sequence, prop: Property)(implicit sourceInfo: SourceInfo): Property =
    OpaqueProperty(LTLImplicationIntrinsic(seq.inner, prop.inner))

  /** Non-overlapping variant of `Property.implication`. Equivalent to
    * `seq ##1 true |-> prop` and `seq |=> prop` in SVA.
    */
  protected def _implicationNonOverlapping(seq: Sequence, prop: Property)(implicit sourceInfo: SourceInfo): Property = {
    import Sequence.BoolSequence
    implication(seq.concat(true.B.delay(1)), prop)
  }

  /** Indicate that a property will eventually hold at a future point in time.
    * This is a *strong* eventually, so the property has to hold within a finite
    * number of cycles. The property does not trivially hold by waiting an
    * infinite number of cycles.
    *
    * Equivalent to `s_eventually prop` in SVA.
    */
  protected def _eventually(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    OpaqueProperty(LTLEventuallyIntrinsic(prop.inner))

  /** Form the conjunction of two properties. Equivalent to
    * `arg0 and arg1 and ... and argN` in SVA.
    */
  protected def _and(arg0: Property, argN: Property*)(implicit sourceInfo: SourceInfo): Property = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueProperty(LTLAndIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  /** Form the disjunction of two properties. Equivalent to
    * `arg0 or arg1 or ... or argN` in SVA.
    */
  protected def _or(arg0: Property, argN: Property*)(implicit sourceInfo: SourceInfo): Property = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueProperty(LTLOrIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  /** Form the conjunction of two properties, where the start and end
    * times of both sequences must be identical. Equivalent to
    * `arg0 intersect arg1 intersect ... intersect argN` in SVA.
    */
  protected def _intersect(arg0: Property, argN: Property*)(implicit sourceInfo: SourceInfo): Property = {
    var lhs = arg0
    for (rhs <- argN) {
      lhs = OpaqueProperty(LTLIntersectIntrinsic(lhs.inner, rhs.inner))
    }
    lhs
  }

  /** Check that a property holds untile another property holds.
    * This operator is weak: the property will hold even if input always
    * holds and condition never holds.
    */
  protected def _until(arg0: Property, arg1: Property)(implicit sourceInfo: SourceInfo): Property =
    OpaqueProperty(LTLUntilIntrinsic(arg0.inner, arg1.inner))

  /** Specify a `clock` relative to which all cycle delays within `prop` are
    * specified. Equivalent to `@(posedge clock) prop` in SVA.
    */
  protected def _clock(prop: Property, clock: Clock)(implicit sourceInfo: SourceInfo): Property =
    OpaqueProperty(LTLClockIntrinsic(prop.inner, clock))
}

/** The base class for the `AssertProperty`, `AssumeProperty`, and
  * `CoverProperty` verification constructs.
  */
sealed abstract class AssertPropertyLike(defaultLayer: Layer) extends AssertPropertyLikeIntf {

  /** Assert, assume, or cover that a property holds.
    *
    * @param prop: parameter can be a `Property`, `Sequence`, or simple `Bool`.
    * @param clock [optional]: specifies a clock with respect to which all cycle
    *   delays in the property are expressed. This is a shorthand for
    *   `prop.clock(clock)`.
    * @param disable [optional]: specifies a condition under which the evaluation
    *   of the property is disabled. This is a shorthand for
    *   `prop.disable(disable)`.
    * @param label [optional]: is used to assign a name to the assert, assume, or
    *   cover construct in the output language. In SystemVerilog, this is
    *   emitted as `label: assert(...)`.
    */
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

  /** Assert, assume, or cover that a boolean predicate holds.
    * @param cond: a boolean predicate that should be checked.
    * This will generate a boolean property that is clocked using the implicit clock
    * and disabled in the case where the design has not yet been reset.
    */
  protected def _applyCondImpl(
    cond: Bool
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    apply(Sequence.BoolSequence(cond))
  }

  /** Assert, assume, or cover that a boolean predicate holds.
    * @param cond: a boolean predicate that should be checked.
    * @param label: is used to assign a name to the assert, assume, or
    *   cover construct in the output language. In SystemVerilog, this is
    *   emitted as `label: assert(...)`.
    * This will generate a boolean property that is clocked using the implicit clock
    * and disabled in the case where the design has not yet been reset.
    */
  protected def _applyCondLabelImpl(
    cond:  Bool,
    label: String
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    apply(Sequence.BoolSequence(cond), label = Some(label))
  }

  /** Assert, assume, or cover that a boolean predicate holds.
    * @param cond: a boolean predicate that should be checked.
    * @param clock: specifies a clock with respect to which all cycle
    *   delays in the property are expressed. This is a shorthand for
    *   `prop.clock(clock)`.
    * @param disable: specifies a condition under which the evaluation
    *   of the property is disabled. This is a shorthand for
    *   `prop.disable(disable)`.
    * @param label: is used to assign a name to the assert, assume, or
    *   cover construct in the output language. In SystemVerilog, this is
    *   emitted as `label: assert(...)`.
    * This will generate a boolean property that is clocked using the implicit clock
    * and disabled in the case where the design has not yet been reset.
    */
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
