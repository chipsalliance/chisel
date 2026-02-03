// SPDX-License-Identifier: Apache-2.0

package chisel3.ltl

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait SequenceIntf { self: Sequence =>

  /** See `Sequence.delay`. */
  def delay(delay: Int = 1)(using SourceInfo): Sequence = _delayImpl(delay)

  /** See `Sequence.delayRange`. */
  def delayRange(min: Int, max: Int)(using SourceInfo): Sequence = _delayRangeImpl(min, max)

  /** See `Sequence.delayAtLeast`. */
  def delayAtLeast(delay: Int)(using SourceInfo): Sequence = _delayAtLeastImpl(delay)

  /** See `Sequence.concat`. */
  def concat(other: Sequence)(using SourceInfo): Sequence = _concatImpl(other)

  /** See `Sequence.repeat`. */
  def repeat(n: Int = 1)(using SourceInfo): Sequence = _repeatImpl(n)

  /** See `Sequence.repeatRange`. */
  def repeatRange(min: Int, max: Int)(using SourceInfo): Sequence = _repeatRangeImpl(min, max)

  /** See `Sequence.repeatAtLeast`. */
  def repeatAtLeast(n: Int)(using SourceInfo): Sequence = _repeatAtLeastImpl(n)

  /** See `Sequence.gotoRepeat`. */
  def gotoRepeat(min: Int, max: Int)(using SourceInfo): Sequence = _gotoRepeatImpl(min, max)

  /** See `Sequence.nonConsecutiveRepeat`. */
  def nonConsecutiveRepeat(min: Int, max: Int)(using SourceInfo): Sequence =
    _nonConsecutiveRepeatImpl(min, max)

  /** See `Sequence.and`. */
  def and(other: Sequence)(using SourceInfo): Sequence = _andSeqImpl(other)

  /** See `Sequence.or`. */
  def or(other: Sequence)(using SourceInfo): Sequence = _orSeqImpl(other)

  /** See `Sequence.intersect`. */
  def intersect(other: Sequence)(using SourceInfo): Sequence = _intersectSeqImpl(other)

  /** See `Sequence.until`. */
  def until(other: Sequence)(using SourceInfo): Sequence = _untilSeqImpl(other)

  /** See `Sequence.clock`. */
  override def clock(clock: Clock)(using SourceInfo): Sequence = _clockImpl(clock)

  /** See `Property.implication`. */
  def implication(prop: Property)(using SourceInfo): Property = _implicationImpl(prop)

  /** See `Property.implication`. */
  def implicationNonOverlapping(prop: Property)(using SourceInfo): Property =
    _implicationNonOverlappingImpl(prop)

  // Convenience operators.

  /** Equivalent to `|->` in SVA. */
  def |->(prop: Property)(using SourceInfo): Property = _impl_|->(prop)

  /** Equivalent to `|=>` in SVA. */
  def |=>(prop: Property)(using SourceInfo): Property = _impl_|=>(prop)

  /** Equivalent to `a ##1 b` in SVA. */
  def ###(other: Sequence)(using SourceInfo): Sequence = _impl_###(other)

  /** Equivalent to `a ##[*] b` in SVA. */
  def ##*(other: Sequence)(using SourceInfo): Sequence = _impl_##*(other)

  /** Equivalent to `a ##[+] b` in SVA. */
  def ##+(other: Sequence)(using SourceInfo): Sequence = _impl_##+(other)
}

private[chisel3] trait PropertyIntf { self: Property =>

  /** See `Property.not`. */
  def not(using SourceInfo): Property = _notImpl

  /** See `Property.eventually`. */
  def eventually(using SourceInfo): Property = _eventuallyImpl

  /** See `Property.and`. */
  def and(other: Property)(using SourceInfo): Property = _andPropImpl(other)

  /** See `Property.or`. */
  def or(other: Property)(using SourceInfo): Property = _orPropImpl(other)

  /** See `Property.intersect`. */
  def intersect(other: Property)(using SourceInfo): Property = _intersectPropImpl(other)

  /** See `Property.until`. */
  def until(other: Property)(using SourceInfo): Property = _untilPropImpl(other)

  /** See `Property.clock`. */
  def clock(clock: Clock)(using SourceInfo): Property = _clockImpl(clock)
}

private[chisel3] trait Sequence$Intf { self: Sequence.type =>

  /** Delay a sequence by a fixed number of cycles. Equivalent to `##delay` in
    * SVA.
    */
  def delay(seq: Sequence, delay: Int = 1)(using SourceInfo): Sequence = _delay(seq, delay)

  /** Delay a sequence by a bounded range of cycles. Equivalent to `##[min:max]`
    * in SVA.
    */
  def delayRange(seq: Sequence, min: Int, max: Int)(using SourceInfo): Sequence =
    _delayRange(seq, min, max)

  /** Delay a sequence by an unbounded range of cycles. Equivalent to
    * `##[delay:$]` in SVA.
    */
  def delayAtLeast(seq: Sequence, delay: Int)(using SourceInfo): Sequence = _delayAtLeast(seq, delay)

  /** Concatenate multiple sequences. Equivalent to
    * `arg0 ##0 arg1 ##0 ... ##0 argN` in SVA.
    */
  def concat(arg0: Sequence, argN: Sequence*)(using SourceInfo): Sequence = _concat(arg0, argN: _*)

  /** Repeat a sequence a fixed number of consecutive times. Equivalent to `s[n]` in
    * SVA.
    */
  def repeat(seq: Sequence, n: Int = 1)(using SourceInfo): Sequence = _repeat(seq, n)

  /** Repeat a sequence by a bounded range of consecutive times. Equivalent to `s[min:max]`
    * in SVA.
    */
  def repeatRange(seq: Sequence, min: Int, max: Int)(using SourceInfo): Sequence =
    _repeatRange(seq, min, max)

  /** Repeat a sequence by an unbounded range of consecutive times. Equivalent to
    * `s[n:$]` in SVA.
    */
  def repeatAtLeast(seq: Sequence, n: Int)(using SourceInfo): Sequence = _repeatAtLeast(seq, n)

  /** GoTo-style repitition of a sequence a fixed number of non-consecutive times,
    * where the final evaluation of the sequence must hold, e.g.
    * a !b b b !b !b b c represents a matching observation of `gotoRepeat(b, 1, 3)`,
    * but a !b b b !b !b b !b c does not. Equivalent to `s[->min:max]` in SVA.
    */
  def gotoRepeat(seq: Sequence, min: Int, max: Int)(using SourceInfo): Sequence =
    _gotoRepeat(seq, min, max)

  /** Repeat a sequence a fixed number of non-consecutive times,
    * where the final evaluation of the sequence does not have to hold, e.g.
    * both a !b b b !b !b b c and a !b b b !b !b b !b c represent matching
    * observations of `nonConsecutiveRepeat(b, 1, 3)`. Equivalent to `s[=min:max]` in SVA.
    */
  def nonConsecutiveRepeat(seq: Sequence, min: Int, max: Int)(using SourceInfo): Sequence =
    _nonConsecutiveRepeat(seq, min, max)

  /** Form the conjunction of two sequences. Equivalent to
    * `arg0 and arg1 and ... and argN` in SVA.
    */
  def and(arg0: Sequence, argN: Sequence*)(using SourceInfo): Sequence = _and(arg0, argN: _*)

  /** Form the disjunction of two sequences. Equivalent to
    * `arg0 or arg1 or ... or argN` in SVA.
    */
  def or(arg0: Sequence, argN: Sequence*)(using SourceInfo): Sequence = _or(arg0, argN: _*)

  /** Form the conjunction of two sequences, where the start and end
    * times of both sequences must be identical. Equivalent to
    * `arg0 intersect arg1 intersect ... intersect argN` in SVA.
    */
  def intersect(arg0: Sequence, argN: Sequence*)(using SourceInfo): Sequence =
    _intersect(arg0, argN: _*)

  /** Check that a sequence holds untile another sequence holds.
    * This operator is weak: the property will hold even if input always
    * holds and condition never holds.
    */
  def until(arg0: Sequence, arg1: Sequence)(using SourceInfo): Sequence = _until(arg0, arg1)

  /** Specify a `clock` relative to which all cycle delays within `seq` are
    * specified. Equivalent to `@(posedge clock) seq` in SVA.
    */
  def clock(seq: Sequence, clock: Clock)(using SourceInfo): Sequence = _clock(seq, clock)

  /** Convenience constructor for sequences. Allows for the following syntax:
    *
    * `Sequence(a, Delay(), b, Delay(2), c, Delay(3, 9), d, Delay(4, None), e)`
    *
    * See `SequenceAtom` for more details.
    */
  def apply(atoms: SequenceAtom*)(using SourceInfo): Sequence = _apply(atoms: _*)
}

private[chisel3] trait Property$Intf { self: Property.type =>

  /** Negate a property. Equivalent to `not prop` in SVA. */
  def not(prop: Property)(using SourceInfo): Property = _not(prop)

  /** Precondition the checking of a property (the consequent) on a sequence
    * (the antecedent). Equivalent to the overlapping implication `seq |-> prop`
    * in SVA.
    */
  def implication(seq: Sequence, prop: Property)(using SourceInfo): Property = _implication(seq, prop)

  /** Non-overlapping variant of `Property.implication`. Equivalent to
    * `seq ##1 true |-> prop` and `seq |=> prop` in SVA.
    */
  def implicationNonOverlapping(seq: Sequence, prop: Property)(using SourceInfo): Property =
    _implicationNonOverlapping(seq, prop)

  /** Indicate that a property will eventually hold at a future point in time.
    * This is a *strong* eventually, so the property has to hold within a finite
    * number of cycles. The property does not trivially hold by waiting an
    * infinite number of cycles.
    *
    * Equivalent to `s_eventually prop` in SVA.
    */
  def eventually(prop: Property)(using SourceInfo): Property = _eventually(prop)

  /** Form the conjunction of two properties. Equivalent to
    * `arg0 and arg1 and ... and argN` in SVA.
    */
  def and(arg0: Property, argN: Property*)(using SourceInfo): Property = _and(arg0, argN: _*)

  /** Form the disjunction of two properties. Equivalent to
    * `arg0 or arg1 or ... or argN` in SVA.
    */
  def or(arg0: Property, argN: Property*)(using SourceInfo): Property = _or(arg0, argN: _*)

  /** Form the conjunction of two properties, where the start and end
    * times of both sequences must be identical. Equivalent to
    * `arg0 intersect arg1 intersect ... intersect argN` in SVA.
    */
  def intersect(arg0: Property, argN: Property*)(using SourceInfo): Property = _intersect(arg0, argN: _*)

  /** Check that a property holds untile another property holds.
    * This operator is weak: the property will hold even if input always
    * holds and condition never holds.
    */
  def until(arg0: Property, arg1: Property)(using SourceInfo): Property = _until(arg0, arg1)

  /** Specify a `clock` relative to which all cycle delays within `prop` are
    * specified. Equivalent to `@(posedge clock) prop` in SVA.
    */
  def clock(prop: Property, clock: Clock)(using SourceInfo): Property = _clock(prop, clock)
}

private[chisel3] trait AssertPropertyLikeIntf { self: AssertPropertyLike =>

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
    prop:    => Property,
    clock:   Option[Clock] = Module.clockOption,
    disable: Option[Disable] = Module.disableOption,
    label:   Option[String] = None
  )(using SourceInfo): Unit = _applyImpl(prop, clock, disable, label)

  /** Assert, assume, or cover that a boolean predicate holds.
    * @param cond: a boolean predicate that should be checked.
    * This will generate a boolean property that is clocked using the implicit clock
    * and disabled in the case where the design has not yet been reset.
    */
  def apply(cond: Bool)(using SourceInfo): Unit = _applyCondImpl(cond)

  /** Assert, assume, or cover that a boolean predicate holds.
    * @param cond: a boolean predicate that should be checked.
    * @param label: is used to assign a name to the assert, assume, or
    *   cover construct in the output language. In SystemVerilog, this is
    *   emitted as `label: assert(...)`.
    * This will generate a boolean property that is clocked using the implicit clock
    * and disabled in the case where the design has not yet been reset.
    */
  def apply(cond: Bool, label: String)(using SourceInfo): Unit = _applyCondLabelImpl(cond, label)

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
  )(using SourceInfo): Unit = _applyCondClockDisableLabelImpl(cond, clock, disable, label)
}
