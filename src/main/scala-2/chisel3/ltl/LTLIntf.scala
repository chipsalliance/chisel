// SPDX-License-Identifier: Apache-2.0

package chisel3.ltl

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait SequenceIntf { self: Sequence =>

  /** See `Sequence.delay`. */
  def delay(delay: Int = 1)(implicit sourceInfo: SourceInfo): Sequence = _delayImpl(delay)

  /** See `Sequence.delayRange`. */
  def delayRange(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = _delayRangeImpl(min, max)

  /** See `Sequence.delayAtLeast`. */
  def delayAtLeast(delay: Int)(implicit sourceInfo: SourceInfo): Sequence = _delayAtLeastImpl(delay)

  /** See `Sequence.concat`. */
  def concat(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = _concatImpl(other)

  /** See `Sequence.repeat`. */
  def repeat(n: Int = 1)(implicit sourceInfo: SourceInfo): Sequence = _repeatImpl(n)

  /** See `Sequence.repeatRange`. */
  def repeatRange(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = _repeatRangeImpl(min, max)

  /** See `Sequence.repeatAtLeast`. */
  def repeatAtLeast(n: Int)(implicit sourceInfo: SourceInfo): Sequence = _repeatAtLeastImpl(n)

  /** See `Sequence.gotoRepeat`. */
  def gotoRepeat(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence = _gotoRepeatImpl(min, max)

  /** See `Sequence.nonConsecutiveRepeat`. */
  def nonConsecutiveRepeat(min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    _nonConsecutiveRepeatImpl(min, max)

  /** See `Sequence.and`. */
  def and(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = _andSeqImpl(other)

  /** See `Sequence.or`. */
  def or(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = _orSeqImpl(other)

  /** See `Sequence.intersect`. */
  def intersect(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = _intersectSeqImpl(other)

  /** See `Sequence.until`. */
  def until(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = _untilSeqImpl(other)

  /** See `Sequence.clock`. */
  override def clock(clock: Clock)(implicit sourceInfo: SourceInfo): Sequence = _clockImpl(clock)

  /** See `Property.implication`. */
  def implication(prop: Property)(implicit sourceInfo: SourceInfo): Property = _implicationImpl(prop)

  /** See `Property.implication`. */
  def implicationNonOverlapping(prop: Property)(implicit sourceInfo: SourceInfo): Property =
    _implicationNonOverlappingImpl(prop)

  // Convenience operators.

  /** Equivalent to `|->` in SVA. */
  def |->(prop: Property)(implicit sourceInfo: SourceInfo): Property = _impl_|->(prop)

  /** Equivalent to `|=>` in SVA. */
  def |=>(prop: Property)(implicit sourceInfo: SourceInfo): Property = _impl_|=>(prop)

  /** Equivalent to `a ##1 b` in SVA. */
  def ###(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = _impl_###(other)

  /** Equivalent to `a ##[*] b` in SVA. */
  def ##*(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = _impl_##*(other)

  /** Equivalent to `a ##[+] b` in SVA. */
  def ##+(other: Sequence)(implicit sourceInfo: SourceInfo): Sequence = _impl_##+(other)
}

private[chisel3] trait PropertyIntf { self: Property =>

  /** See `Property.not`. */
  def not(implicit sourceInfo: SourceInfo): Property = _notImpl

  /** See `Property.eventually`. */
  def eventually(implicit sourceInfo: SourceInfo): Property = _eventuallyImpl

  /** See `Property.and`. */
  def and(other: Property)(implicit sourceInfo: SourceInfo): Property = _andPropImpl(other)

  /** See `Property.or`. */
  def or(other: Property)(implicit sourceInfo: SourceInfo): Property = _orPropImpl(other)

  /** See `Property.intersect`. */
  def intersect(other: Property)(implicit sourceInfo: SourceInfo): Property = _intersectPropImpl(other)

  /** See `Property.until`. */
  def until(other: Property)(implicit sourceInfo: SourceInfo): Property = _untilPropImpl(other)

  /** See `Property.clock`. */
  def clock(clock: Clock)(implicit sourceInfo: SourceInfo): Property = _clockImpl(clock)
}

private[chisel3] trait Sequence$Intf { self: Sequence.type =>

  /** Delay a sequence by a fixed number of cycles. */
  def delay(seq: Sequence, delay: Int = 1)(implicit sourceInfo: SourceInfo): Sequence = _delay(seq, delay)

  /** Delay a sequence by a bounded range of cycles. */
  def delayRange(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    _delayRange(seq, min, max)

  /** Delay a sequence by an unbounded range of cycles. */
  def delayAtLeast(seq: Sequence, delay: Int)(implicit sourceInfo: SourceInfo): Sequence = _delayAtLeast(seq, delay)

  /** Concatenate multiple sequences. */
  def concat(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = _concat(arg0, argN: _*)

  /** Repeat a sequence a fixed number of consecutive times. */
  def repeat(seq: Sequence, n: Int = 1)(implicit sourceInfo: SourceInfo): Sequence = _repeat(seq, n)

  /** Repeat a sequence by a bounded range of consecutive times. */
  def repeatRange(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    _repeatRange(seq, min, max)

  /** Repeat a sequence by an unbounded range of consecutive times. */
  def repeatAtLeast(seq: Sequence, n: Int)(implicit sourceInfo: SourceInfo): Sequence = _repeatAtLeast(seq, n)

  /** GoTo-style repitition of a sequence. */
  def gotoRepeat(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    _gotoRepeat(seq, min, max)

  /** Repeat a sequence a fixed number of non-consecutive times. */
  def nonConsecutiveRepeat(seq: Sequence, min: Int, max: Int)(implicit sourceInfo: SourceInfo): Sequence =
    _nonConsecutiveRepeat(seq, min, max)

  /** Form the conjunction of two sequences. */
  def and(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = _and(arg0, argN: _*)

  /** Form the disjunction of two sequences. */
  def or(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence = _or(arg0, argN: _*)

  /** Form the intersection of two sequences. */
  def intersect(arg0: Sequence, argN: Sequence*)(implicit sourceInfo: SourceInfo): Sequence =
    _intersect(arg0, argN: _*)

  /** Check that a sequence holds until another sequence holds. */
  def until(arg0: Sequence, arg1: Sequence)(implicit sourceInfo: SourceInfo): Sequence = _until(arg0, arg1)

  /** Specify a clock relative to which all cycle delays are specified. */
  def clock(seq: Sequence, clock: Clock)(implicit sourceInfo: SourceInfo): Sequence = _clock(seq, clock)

  /** Convenience constructor for sequences. */
  def apply(atoms: SequenceAtom*)(implicit sourceInfo: SourceInfo): Sequence = _apply(atoms: _*)
}

private[chisel3] trait Property$Intf { self: Property.type =>

  /** Negate a property. */
  def not(prop: Property)(implicit sourceInfo: SourceInfo): Property = _not(prop)

  /** Precondition the checking of a property on a sequence. */
  def implication(seq: Sequence, prop: Property)(implicit sourceInfo: SourceInfo): Property = _implication(seq, prop)

  /** Non-overlapping variant of implication. */
  def implicationNonOverlapping(seq: Sequence, prop: Property)(implicit sourceInfo: SourceInfo): Property =
    _implicationNonOverlapping(seq, prop)

  /** Indicate that a property will eventually hold. */
  def eventually(prop: Property)(implicit sourceInfo: SourceInfo): Property = _eventually(prop)

  /** Form the conjunction of two properties. */
  def and(arg0: Property, argN: Property*)(implicit sourceInfo: SourceInfo): Property = _and(arg0, argN: _*)

  /** Form the disjunction of two properties. */
  def or(arg0: Property, argN: Property*)(implicit sourceInfo: SourceInfo): Property = _or(arg0, argN: _*)

  /** Form the intersection of two properties. */
  def intersect(arg0: Property, argN: Property*)(implicit sourceInfo: SourceInfo): Property = _intersect(arg0, argN: _*)

  /** Check that a property holds until another property holds. */
  def until(arg0: Property, arg1: Property)(implicit sourceInfo: SourceInfo): Property = _until(arg0, arg1)

  /** Specify a clock relative to which all cycle delays are specified. */
  def clock(prop: Property, clock: Clock)(implicit sourceInfo: SourceInfo): Property = _clock(prop, clock)
}

private[chisel3] trait AssertPropertyLikeIntf { self: AssertPropertyLike =>

  def apply(
    prop:    => Property,
    clock:   Option[Clock] = Module.clockOption,
    disable: Option[Disable] = Module.disableOption,
    label:   Option[String] = None
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = _applyImpl(prop, clock, disable, label)

  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Unit = _applyCondImpl(cond)

  def apply(cond: Bool, label: String)(implicit sourceInfo: SourceInfo): Unit = _applyCondLabelImpl(cond, label)

  def apply(
    cond:    Bool,
    clock:   Clock,
    disable: Disable,
    label:   String
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = _applyCondClockDisableLabelImpl(cond, clock, disable, label)
}