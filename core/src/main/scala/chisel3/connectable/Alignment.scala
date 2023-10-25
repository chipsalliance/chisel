// SPDX-License-Identifier: Apache-2.0

package chisel3.connectable

import chisel3.{Aggregate, Data, DontCare, SpecifiedDirection}
import chisel3.experimental.Analog
import chisel3.reflect.DataMirror
import chisel3.internal.{ChildBinding, TopBinding}

// Represent aligned or flipped relative to an original root.
// Used for walking types and their alignment, accounting for coercion.
private[chisel3] sealed trait Alignment {
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

  // Whether the current member is an aggregate
  final def isAgg: Boolean = member.isInstanceOf[Aggregate]

  final def empty = EmptyAlignment(isConsumer)
}

// The alignment datastructure for a missing field
private[chisel3] case class EmptyAlignment(isConsumer: Boolean) extends Alignment {
  def member = DontCare
  def invert = this
  def coerced = false
  def coerce = this
  def swap(d: Data): Alignment = this
  def alignment: String = "none"
}

private[chisel3] sealed trait NonEmptyAlignment extends Alignment

private[chisel3] case class AlignedWithRoot(
  member:     Data,
  coerced:    Boolean,
  isConsumer: Boolean)
    extends NonEmptyAlignment {
  def invert = if (coerced) this else FlippedWithRoot(member, coerced, isConsumer)
  def coerce = this.copy(member, true)
  def swap(d: Data): Alignment = this.copy(member = d)
  def alignment: String = "aligned"
}

private[chisel3] case class FlippedWithRoot(
  member:     Data,
  coerced:    Boolean,
  isConsumer: Boolean)
    extends NonEmptyAlignment {
  def invert = if (coerced) this else AlignedWithRoot(member, coerced, isConsumer)
  def coerce = this.copy(member, true)
  def swap(d: Data): Alignment = this.copy(member = d)
  def alignment: String = "flipped"
}

object Alignment {
  private[chisel3] def apply(base: Data, isConsumer: Boolean): Alignment =
    AlignedWithRoot(base, isCoercing(base), isConsumer)

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
    Data.dataMatchingZipOfChildren.matchingZipOfChildren(left.map(_.member), right.map(_.member)).map {
      case (l, r) => l.map(deriveChildAlignment(_, left.get)) -> r.map(deriveChildAlignment(_, right.get))
    }
  }
}

// Track base of a connectable and its Alignment, with extra helpers for using in connections.
// Internal datastructure used to compute connectable operator connections
private[chisel3] case class ConnectableAlignment(base: Connectable[Data], align: Alignment) {

  private def skipWidth(x: Data): Boolean = x.isInstanceOf[Aggregate] || x.isInstanceOf[DontCare.type]

  // Returns loc and roc
  final def computeLandR(c: Data, p: Data, op: Connection): Option[(Data, Data)] = {
    (c, p, this.align, op.connectToConsumer, op.connectToProducer, op.alwaysConnectToConsumer) match {
      case (x: Analog, y: Analog, _, _, _, _) => Some((x, y))
      case (x: Analog, DontCare, _, _, _, _) => Some((x, DontCare))
      case (x, y, _: AlignedWithRoot, true, _, _) => Some((c, p))
      case (x, y, _: FlippedWithRoot, _, true, _) => Some((p, c))
      case (x, y, _, _, _, true) => Some((c, p))
      case other                 => None
    }
  }

  final def truncationRequired(o: ConnectableAlignment, op: Connection): Option[Data] =
    (align.member, o.align.member) match {
      case (x, y) if (skipWidth(x) || skipWidth(y)) => None
      case (x, y) =>
        val lAndROpt = computeLandR(align.member, o.align.member, op)
        lAndROpt.map {
          case (l, r) if !(base.squeezed.contains(r) || o.base.squeezed.contains(r)) =>
            (l.widthOption, r.widthOption) match {
              case (None, _)                       => None // l will infer a large enough width
              case (Some(x), None)                 => Some(r) // r could infer a larger width than l's width
              case (Some(lw), Some(rw)) if lw < rw => Some(r)
              case (Some(lw), Some(rw))            => None
            }
          case (l, r) => None
        }.getOrElse(None)
    }

  // Whether the current member is waived
  final def isWaived: Boolean = base.waived.contains(align.member)

  // Whether the current member is squeezed
  final def isSqueezed: Boolean = base.squeezed.contains(align.member)

  // Whether the current member is excluded
  final def isExcluded: Boolean = base.excluded.contains(align.member)

  // If erroring, determine the correct word to use
  final def errorWord(op: Connection): String =
    (align.isConsumer, op.connectToConsumer, op.connectToProducer, align.alignment) match {
      case (true, true, _, "aligned")  => "unconnected"
      case (false, _, true, "flipped") => "unconnected"
      case (true, _, true, "flipped")  => "dangling"
      case (false, true, _, "aligned") => "dangling"
      case other                       => "unmatched"
    }

  /// Expose some underlying Alignment methods for convenience.
  /// Define in terms of ConnectableAlignment but defer to Alignment's code.

  final def member: Data = align.member

  final def alignsWith(o: ConnectableAlignment): Boolean = align.alignsWith(o.align)

  // Whether the current member is an aggregate
  final def isAgg: Boolean = align.isAgg

  final def empty = ConnectableAlignment(base, align.empty)

  final def swap(d: Data): ConnectableAlignment = ConnectableAlignment(base, align.swap(d))
}

object ConnectableAlignment {
  private[chisel3] def apply(base: Connectable[Data], isConsumer: Boolean): ConnectableAlignment =
    ConnectableAlignment(base, Alignment(base.base, isConsumer))

  private[chisel3] def deriveChildAlignment(
    subMember:       Data,
    parentAlignment: ConnectableAlignment
  ): ConnectableAlignment = {
    ConnectableAlignment(parentAlignment.base, Alignment.deriveChildAlignment(subMember, parentAlignment.align))
  }
  private[chisel3] def matchingZipOfChildren(
    left:  Option[ConnectableAlignment],
    right: Option[ConnectableAlignment]
  ): Seq[(Option[ConnectableAlignment], Option[ConnectableAlignment])] = {
    Data.dataMatchingZipOfChildren.matchingZipOfChildren(left.map(_.align.member), right.map(_.align.member)).map {
      case (l, r) => l.map(deriveChildAlignment(_, left.get)) -> r.map(deriveChildAlignment(_, right.get))
    }
  }
}
