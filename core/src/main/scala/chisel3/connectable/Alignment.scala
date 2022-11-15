// SPDX-License-Identifier: Apache-2.0

package chisel3.connectable

import chisel3.{Aggregate, Data, DontCare, SpecifiedDirection}
import chisel3.experimental.Analog
import chisel3.reflect.DataMirror
import chisel3.internal.{ChildBinding, TopBinding}

// Indicates whether the active side is aligned or flipped relative to the active side's root
// Internal datastructure used to compute connectable operator connections
private[chisel3] sealed trait Alignment {
  def base: Connectable[Data]

  // The member for whom this alignment is for
  def member: Data

  // Inverts the alignment
  def invert: Alignment

  // Coerces the alignment
  def coerce: Alignment

  // Indicates whether the alignment has been coerced
  def coerced: Boolean

  // String indicating either 'flipped' or 'aligned'
  def alignment: String

  // Use same alignment datastructure, but swap in a different member
  // Used for computing DontCare, where we take the alignment of the non-DontCare side, but stick in a DontCare
  def swap(d: Data): Alignment

  // Indicates whether this member is being used as a consumer or producer
  // Affects the error word
  def isConsumer: Boolean

  // Whether two alignments are aligned
  final def alignsWith(o: Alignment): Boolean = o.alignment == this.alignment

  private def skipWidth(x: Data): Boolean = x.isInstanceOf[Aggregate] || x.isInstanceOf[DontCare.type]

  final def mismatchedWidths(o: Alignment, op: Connection): Boolean = (this.member, o.member) match {
    case (x, y) if (skipWidth(x) || skipWidth(y)) => false
    case (x, y) =>
      val lAndROpt = computeLandR(this.member, o.member, op)
      lAndROpt.map {
        case (l, r) if !(base.squeezed.contains(r) || o.base.squeezed.contains(r)) =>
          (l.widthOption, r.widthOption) match {
            case (None, _)            => false // l will infer a large enough width
            case (Some(x), None)      => true // r could infer a larger width than l's width
            case (Some(lw), Some(rw)) => lw < rw
          }
        case (l, r) => false
      }.getOrElse(false)
  }

  // Returns loc and roc
  final def computeLandR(c: Data, p: Data, op: Connection): Option[(Data, Data)] = {
    (c, p, this, op.connectToConsumer, op.connectToProducer, op.alwaysConnectToConsumer) match {
      case (x: Analog, y: Analog, _, _, _, _) => Some((x, y))
      case (x: Analog, DontCare, _, _, _, _) => Some((x, DontCare))
      case (x, y, _: AlignedWithRoot, true, _, _) => Some((c, p))
      case (x, y, _: FlippedWithRoot, _, true, _) => Some((p, c))
      case (x, y, _, _, _, true) => Some((c, p))
      case other                 => None
    }
  }

  // Whether the current member is waived
  final def isWaived: Boolean = base.waived.contains(member)

  final def isSqueezed: Boolean = base.squeezed.contains(member)

  // Whether the current member is an aggregate
  final def isAgg: Boolean = member.isInstanceOf[Aggregate]

  // If erroring, determine the correct word to use
  final def errorWord(op: Connection): String =
    (isConsumer, op.connectToConsumer, op.connectToProducer, alignment) match {
      case (true, true, _, "aligned")  => "unconnected"
      case (false, _, true, "flipped") => "unconnected"
      case (true, _, true, "flipped")  => "dangling"
      case (false, true, _, "aligned") => "dangling"
      case other                       => "unmatched"
    }

  final def empty = EmptyAlignment(base, isConsumer)
}

// The alignment datastructure for a missing field
private[chisel3] case class EmptyAlignment(base: Connectable[Data], isConsumer: Boolean) extends Alignment {
  def member = DontCare
  def waived = Set.empty
  def squeezed = Set.empty
  def invert = this
  def coerced = false
  def coerce = this
  def swap(d: Data): Alignment = this
  def alignment: String = "none"
}

private[chisel3] sealed trait NonEmptyAlignment extends Alignment

private[chisel3] case class AlignedWithRoot(
  base:       Connectable[Data],
  member:     Data,
  coerced:    Boolean,
  isConsumer: Boolean)
    extends NonEmptyAlignment {
  def invert = if (coerced) this else FlippedWithRoot(base, member, coerced, isConsumer)
  def coerce = this.copy(base, member, true)
  def swap(d: Data): Alignment = this.copy(member = d)
  def alignment: String = "aligned"
}

private[chisel3] case class FlippedWithRoot(
  base:       Connectable[Data],
  member:     Data,
  coerced:    Boolean,
  isConsumer: Boolean)
    extends NonEmptyAlignment {
  def invert = if (coerced) this else AlignedWithRoot(base, member, coerced, isConsumer)
  def coerce = this.copy(base, member, true)
  def swap(d: Data): Alignment = this.copy(member = d)
  def alignment: String = "flipped"
}

object Alignment {

  private[chisel3] def apply(base: Connectable[Data], isConsumer: Boolean): Alignment =
    AlignedWithRoot(base, base.base, isCoercing(base.base), isConsumer)

  /** Indicates whether a member of a component or type is coercing
    * This occurs if the member or a parent of member is declared with an `Input` or `Output`
    *
    * @param member the member of a component or type that could be coercing
    */
  def isCoercing(member: Data): Boolean = {
    def recUp(x: Data): Boolean = x.binding match {
      case _ if isLocallyCoercing(x) => true
      case None                      => false
      case Some(t: TopBinding) => false
      case Some(ChildBinding(p)) => recUp(p)
      case other                 => throw new Exception(s"Unexpected $other! $x, $member")
    }
    def isLocallyCoercing(d: Data): Boolean = {
      val s = DataMirror.specifiedDirectionOf(d)
      (s == SpecifiedDirection.Input) || (s == SpecifiedDirection.Output)
    }
    val ret = recUp(member)
    ret
  }

  private[chisel3] def deriveChildAlignment(subMember: Data, parentAlignment: Alignment): Alignment = {
    (DataMirror.specifiedDirectionOf(subMember)) match {
      case (SpecifiedDirection.Unspecified) => parentAlignment.swap(subMember)
      case (SpecifiedDirection.Flip)        => parentAlignment.invert.swap(subMember)
      case (SpecifiedDirection.Output)      => parentAlignment.coerce.swap(subMember)
      case (SpecifiedDirection.Input)       => parentAlignment.invert.coerce.swap(subMember)
      case other                            => throw new Exception(s"Unexpected internal error! $other")
    }
  }

  private[chisel3] def matchingZipOfChildren(
    left:  Option[Alignment],
    right: Option[Alignment]
  ): Seq[(Option[Alignment], Option[Alignment])] = {
    Data.DataMatchingZipOfChildren.matchingZipOfChildren(left.map(_.member), right.map(_.member)).map {
      case (Some(l), None)    => (Some(deriveChildAlignment(l, left.get)), None)
      case (Some(l), Some(r)) => (Some(deriveChildAlignment(l, left.get)), Some(deriveChildAlignment(r, right.get)))
      case (None, Some(r))    => (None, Some(deriveChildAlignment(r, right.get)))
    }
  }
}
