// SPDX-License-Identifier: Apache-2.0

package chisel3.connectable

import chisel3.{DataMirror, Data, SpecifiedDirection, DontCare, Aggregate}
import chisel3.internal.{ChildBinding, TopBinding}

// Indicates whether the active side is aligned or flipped relative to the active side's root
// Internal datastructure used to compute connectable operator assignments
private[chisel3] sealed trait Alignment { 
  // The member for whom this alignment is for
  def member: Data

  // Waivers for connectable operators
  def waivers: Set[Data]

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

  // Whether the current member is waived
  final def isWaived: Boolean = waivers.contains(member)

  // Whether the current member is an aggregate
  final def isAgg: Boolean = member.isInstanceOf[Aggregate]

  // If erroring, determine the correct word to use
  final def errorWord(op: ConnectionOperator): String = (isConsumer, op.assignToConsumer, op.assignToProducer, alignment) match {
    case (true,  true,   _,    "aligned") => "unassigned"
    case (false, _,      true, "flipped") => "unassigned"
    case (true,  _,      true, "flipped") => "dangling"
    case (false, true,   _,    "aligned") => "dangling"
    case other => "unmatched"
  }
}

// The alignment datastructure for a missing field
private[chisel3] case object EmptyAlignment extends Alignment {
  def member = DontCare
  def waivers = Set.empty
  def invert = this
  def coerced = false
  def coerce = this
  def swap(d: Data): Alignment = this
  def alignment: String = "none"
  def isConsumer = ??? // should never call this
}

private[chisel3] sealed trait NonEmptyAlignment extends Alignment

private[chisel3] case class AlignedWithRoot(member: Data, coerced: Boolean, waivers: Set[Data], isConsumer: Boolean) extends NonEmptyAlignment {
  def invert = if(coerced) this else FlippedWithRoot(member, coerced, waivers, isConsumer)
  def coerce = this.copy(member, true)
  def swap(d: Data): Alignment = this.copy(member = d)
  def alignment: String = "aligned"
}

private[chisel3] case class FlippedWithRoot(member: Data, coerced: Boolean, waivers: Set[Data], isConsumer: Boolean) extends NonEmptyAlignment {
  def invert = if(coerced) this else AlignedWithRoot(member, coerced, waivers, isConsumer)
  def coerce = this.copy(member, true)
  def swap(d: Data): Alignment = this.copy(member = d)
  def alignment: String = "flipped"
}

object Alignment {

  private[chisel3] def apply(base: Data, waivers: Set[Data], isConsumer: Boolean): Alignment = AlignedWithRoot(base, isCoercing(base), waivers, isConsumer)

  /** Indicates whether a member of a component or type is coercing
    * This occurs if the member or a parent of member is declared with an `Input` or `Output`
    *
    * @param member the member of a component or type that could be coercing
    */
  def isCoercing(member: Data): Boolean = {
    def recUp(x: Data): Boolean = x.binding match {
      case _ if isLocallyCoercing(x)      => true
      case None                           => false
      case Some(t: TopBinding)   => false
      case Some(ChildBinding(p)) => recUp(p)
      case other                          => throw new Exception(s"Unexpected $other! $x, $member")
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

  private[chisel3] def matchingZipOfChildren(left: Option[Alignment], right: Option[Alignment]): Seq[(Option[Alignment], Option[Alignment])] = {
    Data.DataMatchingZipOfChildren.matchingZipOfChildren(left.map(_.member), right.map(_.member)).map {
      case (Some(l), None)    => (Some(deriveChildAlignment(l, left.get)), None)
      case (Some(l), Some(r)) => (Some(deriveChildAlignment(l, left.get)), Some(deriveChildAlignment(r, right.get)))
      case (None, Some(r))    => (None, Some(deriveChildAlignment(r, right.get)))
    }
  }
}
