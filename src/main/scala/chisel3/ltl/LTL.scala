// SPDX-License-Identifier: Apache-2.0

package chisel3.ltl

import chisel3._
import chisel3.util.circt._
import chisel3.experimental.hierarchy.{Instance, Instantiate}

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
  def apply(): SequenceAtom = DelayAtom(1, Some(1))
  def apply(delay: Int): SequenceAtom = DelayAtom(delay, Some(delay))
  def apply(min:   Int, max: Int): SequenceAtom = DelayAtom(min, Some(max))
  def apply(min:   Int, max: Option[Int]): SequenceAtom = DelayAtom(min, max)
}

/** A Linear Temporal Logic (LTL) sequence. */
sealed trait Sequence extends Property {

  /** See `Sequence.delay`. */
  def delay(delay: Int = 1): Sequence = Sequence.delay(this, delay)

  /** See `Sequence.delayRange`. */
  def delayRange(min: Int, max: Int): Sequence = Sequence.delayRange(this, min, max)

  /** See `Sequence.delayAtLeast`. */
  def delayAtLeast(delay: Int): Sequence = Sequence.delayAtLeast(this, delay)

  /** See `Sequence.concat`. */
  def concat(other: Sequence): Sequence = Sequence.concat(this, other)

  /** See `Sequence.and`. */
  def and(other: Sequence): Sequence = Sequence.and(this, other)

  /** See `Sequence.or`. */
  def or(other: Sequence): Sequence = Sequence.or(this, other)

  /** See `Sequence.clock`. */
  override def clock(clock: Clock): Sequence = Sequence.clock(this, clock)

  /** See `Property.implication`. */
  def implication(prop: Property): Property = Property.implication(this, prop)

  /** See `Property.implication`. */
  def implicationNonOverlapping(prop: Property): Property = Property.implicationNonOverlapping(this, prop)

  // Convenience operators.

  /** Equivalent to `|->` in SVA. */
  def |->(prop: Property): Property = this.implication(prop)

  /** Equivalent to `|=>` in SVA. */
  def |=>(prop: Property): Property = this.implicationNonOverlapping(prop)

  /** Equivalent to `a ##1 b` in SVA. */
  def ###(other: Sequence): Sequence = this.concat(other.delay())

  /** Equivalent to `a ##[*] b` in SVA. */
  def ##*(other: Sequence): Sequence = this.concat(other.delayAtLeast(0))

  /** Equivalent to `a ##[+] b` in SVA. */
  def ##+(other: Sequence): Sequence = this.concat(other.delayAtLeast(1))
}

/** Prefix-style utilities to work with sequences.
  *
  * This object exposes the primary API to create and compose sequences from
  * booleans and shorter sequences.
  */
object Sequence {

  /** Implicitly wraps a `Bool` and allows it to be used as a sequence or
    * property. Use via `import chisel3.util.ltl.Sequence.BoolSequence`.
    */
  implicit class BoolSequence(val inner: Bool) extends Sequence with SequenceAtom

  /** Delay a sequence by a fixed number of cycles. Equivalent to `##delay` in
    * SVA.
    */
  def delay(seq: Sequence, delay: Int = 1): Sequence = 
    OpaqueSequence(LTLDelayIntrinsic(delay, Some(0))(seq.inner))

  /** Delay a sequence by a bounded range of cycles. Equivalent to `##[min:max]`
    * in SVA.
    */
  def delayRange(seq: Sequence, min: Int, max: Int): Sequence = {
    require(min <= max)
    OpaqueSequence(LTLDelayIntrinsic(min, Some(max - min))(seq.inner))
  }

  /** Delay a sequence by an unbounded range of cycles. Equivalent to
    * `##[delay:$]` in SVA.
    */
  def delayAtLeast(seq: Sequence, delay: Int): Sequence = 
    OpaqueSequence(LTLDelayIntrinsic(delay, None)(seq.inner))
  

  /** Concatenate multiple sequences. Equivalent to
    * `arg0 ##0 arg1 ##0 ... ##0 argN` in SVA.
    */
  def concat(arg0: Sequence, argN: Sequence*): Sequence = {
    var lhs = arg0
    for (rhs <- argN) 
      lhs = OpaqueSequence(LTLConcatIntrinsic(lhs.inner, rhs.inner))
    lhs
  }

  /** Form the conjunction of two sequences. Equivalent to
    * `arg0 and arg1 and ... and argN` in SVA.
    */
  def and(arg0: Sequence, argN: Sequence*): Sequence = {
    var lhs = arg0
    for (rhs <- argN) 
      lhs = OpaqueSequence(LTLAndIntrinsic(lhs.inner, rhs.inner))
    lhs
  }

  /** Form the disjunction of two sequences. Equivalent to
    * `arg0 or arg1 or ... or argN` in SVA.
    */
  def or(arg0: Sequence, argN: Sequence*): Sequence = {
    var lhs = arg0
    for (rhs <- argN) 
      lhs = OpaqueSequence(LTLOrIntrinsic(lhs.inner, rhs.inner))
    lhs
  }

  /** Specify a `clock` relative to which all cycle delays within `seq` are
    * specified. Equivalent to `@(posedge clock) seq` in SVA.
    */
  def clock(seq: Sequence, clock: Clock): Sequence = 
    OpaqueSequence(LTLClockIntrinsic(seq.inner, clock))
  

  /** Convenience constructor for sequences. Allows for the following syntax:
    *
    * `Sequence(a, Delay(), b, Delay(2), c, Delay(3, 9), d, Delay(4, None), e)`
    *
    * See `SequenceAtom` for more details.
    */
  def apply(atoms: SequenceAtom*): Sequence = {
    require(atoms.nonEmpty)
    def needDelayTail = {
      require(
        atoms.tail.nonEmpty,
        "`Delay` operator in `Sequence(...)` must be followed by an item to be delayed"
      )
      Sequence(atoms.tail: _*)
    }
    atoms.head match {
      case seq: Sequence if atoms.tail.nonEmpty => seq.concat(Sequence(atoms.tail: _*))
      case seq: Sequence                        => seq
      case DelayAtom(min, None)      => needDelayTail.delayAtLeast(min)
      case DelayAtom(min, Some(max)) => needDelayTail.delayRange(min, max)
    }
  }
}

/** A Linear Temporal Logic (LTL) property. */
sealed trait Property {

  /** The underlying `Bool` that is returned and accepted by the LTL
    * intrinsics.
    */
  private[ltl] def inner: Bool

  /** See `Property.not`. */
  def not: Property = Property.not(this)

  /** See `Property.eventually`. */
  def eventually: Property = Property.eventually(this)

  /** See `Property.and`. */
  def and(other: Property): Property = Property.and(this, other)

  /** See `Property.or`. */
  def or(other: Property): Property = Property.or(this, other)

  /** See `Property.clock`. */
  def clock(clock: Clock): Property = Property.clock(this, clock)

  /** See `Property.disable`. */
  def disable(cond: Disable): Property = Property.disable(this, cond)
}

/** Prefix-style utilities to work with properties.
  *
  * This object exposes the primary API to create and compose properties from
  * booleans, sequences, and other properties.
  */
object Property {

  /** Negate a property. Equivalent to `not prop` in SVA. */
  def not(prop: Property): Property = 
    OpaqueProperty(LTLNotIntrinsic(prop.inner))

  /** Precondition the checking of a property (the consequent) on a sequence
    * (the antecedent). Equivalent to the overlapping implication `seq |-> prop`
    * in SVA.
    */
  def implication(seq: Sequence, prop: Property): Property = 
    OpaqueProperty(LTLImplicationIntrinsic(seq.inner, prop.inner))

  /** Non-overlapping variant of `Property.implication`. Equivalent to
    * `seq ##1 true |-> prop` and `seq |=> prop` in SVA.
    */
  def implicationNonOverlapping(seq: Sequence, prop: Property): Property = {
    import Sequence.BoolSequence
    Property.implication(seq.concat(true.B.delay(1)), prop)
  }

  /** Indicate that a property will eventually hold at a future point in time.
    * This is a *strong* eventually, so the property has to hold within a finite
    * number of cycles. The property does not trivially hold by waiting an
    * infinite number of cycles.
    *
    * Equivalent to `s_eventually prop` in SVA.
    */
  def eventually(prop: Property): Property = 
    OpaqueProperty(LTLEventuallyIntrinsic(prop.inner))

  /** Form the conjunction of two properties. Equivalent to
    * `arg0 and arg1 and ... and argN` in SVA.
    */
  def and(arg0: Property, argN: Property*): Property = {
    var lhs = arg0
    for (rhs <- argN) 
      lhs = OpaqueProperty(LTLAndIntrinsic(lhs.inner, rhs.inner))
    lhs
  }

  /** Form the disjunction of two properties. Equivalent to
    * `arg0 or arg1 or ... or argN` in SVA.
    */
  def or(arg0: Property, argN: Property*): Property = {
    var lhs = arg0
    for (rhs <- argN) 
      lhs = OpaqueProperty(LTLOrIntrinsic(lhs.inner, rhs.inner))
    lhs
  }

  /** Specify a `clock` relative to which all cycle delays within `prop` are
    * specified. Equivalent to `@(posedge clock) prop` in SVA.
    */
  def clock(prop: Property, clock: Clock): Property = 
    OpaqueProperty(LTLClockIntrinsic(prop.inner, clock))

  /** Disable the checking of a property if a condition is true. If the
    * condition is true at any time during the evaluation of the property, the
    * evaluation is aborted. Equivalent to `disable iff (cond) prop` in SVA.
    */
  def disable(prop: Property, cond: Disable): Property = 
    OpaqueProperty(LTLDisableIntrinsic(prop.inner, cond.value))
}

/** The base class for the `AssertProperty`, `AssumeProperty`, and
  * `CoverProperty` verification constructs.
  */
sealed abstract class AssertPropertyLike {

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
  def apply(
    prop:    Property,
    clock:   Option[Clock] = Module.clockOption,
    disable: Option[Disable] = Module.disableOption,
    label:   Option[String] = None
  ): Unit = {
    val disabled = disable.fold(prop)(prop.disable(_))
    val clocked = clock.fold(disabled)(disabled.clock(_))
    createIntrinsic(label)(clocked.inner)
  }

  /** Assert, assume, or cover that a boolean predicate holds.
    * @param cond: a boolean predicate that should be checked.
    * This will generate a boolean property that is clocked using the implicit clock
    * and disabled in the case where the design has not yet been reset.
    */
  def apply(
    cond: Bool
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
  def apply(
    cond:  Bool,
    label: String
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
  def apply(
    cond:    Bool,
    clock:   Clock,
    disable: Disable,
    label:   String
  ): Unit = {
    apply(Sequence.BoolSequence(cond), Some(clock), Some(disable), Some(label))
  }

  def createIntrinsic(label: Option[String]): (Bool) => Unit
}

/** Assert that a property holds.
  *
  * Use like `AssertProperty(p)`. See `AssertPropertyLike.apply` for optional
  * clock, disable_iff, and label parameters.
  */
object AssertProperty extends AssertPropertyLike {
  def createIntrinsic(label: Option[String]) = VerifAssertIntrinsic(label)
}

/** Assume that a property holds.
  *
  * Use like `AssumeProperty(p)`. See `AssertPropertyLike.apply` for optional
  * clock, disable_iff, and label parameters.
  */
object AssumeProperty extends AssertPropertyLike {
  def createIntrinsic(label: Option[String]) = VerifAssumeIntrinsic(label)
}

/** Cover that a property holds.
  *
  * Use like `CoverProperty(p)`. See `AssertPropertyLike.apply` for optional
  * clock, disable_iff, and label parameters.
  */
object CoverProperty extends AssertPropertyLike {
  def createIntrinsic(label: Option[String]) = VerifCoverIntrinsic(label)
}
