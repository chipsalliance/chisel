// SPDX-License-Identifier: Apache-2.0

package chisel3
package connectable

import chisel3.experimental.SourceInfo
import chisel3.reflect.DataMirror
import experimental.{prefix, requireIsHardware}

/** A data for whom members if left dangling or unassigned with not trigger an error
  * A waived member will still be connected to if present in both producer and consumer
  *
  * @param base The component being connected to
  * @param waived members of base who will not trigger an error if left dangling or unassigned
  * @param squeezed members of base who will not trigger an error if would end up being truncated
  */
final class Connectable[+T <: Data] private (
  val base:                      T,
  private[chisel3] val waived:   Set[Data],
  private[chisel3] val squeezed: Set[Data],
  private[chisel3] val excluded: Set[Data]) {
  requireIsHardware(base, s"Can only created Connectable of components, not unbound Chisel types")

  private[chisel3] def copy(
    waived:   Set[Data] = this.waived,
    squeezed: Set[Data] = this.squeezed,
    excluded: Set[Data] = this.excluded
  ): Connectable[T] =
    new Connectable(base, waived, squeezed, excluded)

  /** True if no members are waived or squeezed or excluded */
  def notWaivedOrSqueezedOrExcluded = waived.isEmpty && squeezed.isEmpty && excluded.isEmpty

  /** Static cast to a super type */
  def as[S <: Data](implicit ev: T <:< S): Connectable[S] = this.asInstanceOf[Connectable[S]]

  /** Connect to/from all fields regardless of Scala type, squeeze if necessary, and don't error if mismatched members */
  def unsafe: Connectable[Data] = waiveAll.squeezeAll.asInstanceOf[Connectable[Data]]

  /** Select members of base to waive
    *
    * @param members functions given the base return a member to waive
    */
  def waive(members: (T => Data)*): Connectable[T] = this.copy(waived = waived ++ members.map(f => f(base)).toSet)

  /** Select members of base to waive and static cast to a new type
    *
    * @param members functions given the base return a member to waive
    */
  def waiveAs[S <: Data](members: (T => Data)*)(implicit ev: T <:< S): Connectable[S] =
    this.copy(waived = waived ++ members.map(f => f(base)).toSet).asInstanceOf[Connectable[S]]

  /** Programmatically select members of base to waive and static cast to a new type
    *
    * @param members partial function applied to all recursive members of base, if match, can return a member to waive
    */
  def waiveEach[S <: Data](pf: PartialFunction[Data, Seq[Data]])(implicit ev: T <:< S): Connectable[S] = {
    val waivedMembers = DataMirror.collectMembers(base)(pf).flatten
    this.copy(waived = waived ++ waivedMembers.toSet).asInstanceOf[Connectable[S]]
  }

  /** Waive all members of base */
  def waiveAll: Connectable[T] = {
    val waivedMembers = DataMirror.collectMembers(base) { case x => x }
    this.copy(waived = waivedMembers.toSet) // not appending waived because we are collecting all members
  }

  /** Waive all members of base and static cast to a new type */
  def waiveAllAs[S <: Data](implicit ev: T <:< S): Connectable[S] = waiveAll.asInstanceOf[Connectable[S]]

  /** Adds base to squeezes */
  def squeeze: Connectable[T] = this.copy(squeezed = squeezed ++ addOpaque(Seq(base)))

  /** Select members of base to squeeze
    *
    * @param members functions given the base return a member to squeeze
    */
  def squeeze(members: (T => Data)*): Connectable[T] = {
    this.copy(squeezed = squeezed ++ addOpaque(members.map(f => f(base))))
  }

  /** Programmatically select members of base to squeeze
    *
    * @param members partial function applied to all recursive members of base, if match, can return a member to squeeze
    */
  def squeezeEach[S <: Data](pf: PartialFunction[Data, Seq[Data]]): Connectable[T] = {
    val squeezedMembers = addOpaque(DataMirror.collectMembers(base)(pf).flatten.toSeq)
    this.copy(squeezed = squeezed ++ squeezedMembers)
  }

  /** Squeeze all members of base */
  def squeezeAll: Connectable[T] = {
    val squeezedMembers = DataMirror.collectMembers(base) { case x => x }
    this.copy(squeezed = squeezedMembers.toSet) // not appending squeezed because we are collecting all members
  }

  /** Squeeze all members of base and upcast to super type */
  def squeezeAllAs[S <: Data](implicit ev: T <:< S): Connectable[S] = squeezeAll.asInstanceOf[Connectable[S]]

  /** Adds base to excludes */
  def exclude: Connectable[T] = this.copy(excluded = excluded ++ addOpaque(Seq(base)))

  /** Select members of base to exclude
    *
    * @param members functions given the base return a member to exclude
    */
  def exclude(members: (T => Data)*): Connectable[T] = this.copy(excluded = excluded ++ members.map(f => f(base)).toSet)

  /** Select members of base to exclude and static cast to a new type
    *
    * @param members functions given the base return a member to exclude
    */
  def excludeAs[S <: Data](members: (T => Data)*)(implicit ev: T <:< S): Connectable[S] =
    this.copy(excluded = excluded ++ members.map(f => f(base)).toSet).asInstanceOf[Connectable[S]]

  /** Programmatically select members of base to exclude and static cast to a new type
    *
    * @param members partial function applied to all recursive members of base, if match, can return a member to exclude
    */
  def excludeEach[S <: Data](pf: PartialFunction[Data, Seq[Data]])(implicit ev: T <:< S): Connectable[S] = {
    val excludedMembers = DataMirror.collectMembers(base)(pf).flatten
    this.copy(excluded = excluded ++ excludedMembers.toSet).asInstanceOf[Connectable[S]]
  }

  /** Exclude probes */
  def excludeProbes: Connectable[T] = excludeEach {
    case f if (DataMirror.hasProbeTypeModifier(f)) => Seq(f)
  }

  /** Add any elements of members that are OpaqueType */
  private def addOpaque(members: Seq[Data]): Seq[Data] = {
    members.flatMap {
      case x: Record if x._isOpaqueType => Seq(x, x.getElements.head)
      case o => Seq(o)
    }
  }
}

object Connectable {

  /** Create a Connectable from a Data */
  def apply[T <: Data](
    base:             T,
    waiveSelection:   Data => Boolean = { _ => false },
    squeezeSelection: Data => Boolean = { _ => false },
    excludeSelection: Data => Boolean = { _ => false }
  ): Connectable[T] = {
    val (waived, squeezed, excluded) = {
      val members = DataMirror.collectMembers(base) { case x => x }
      (
        members.filter(waiveSelection).toSet,
        members.filter(squeezeSelection).toSet,
        members.filter(excludeSelection).toSet
      )
    }
    new Connectable(base, waived, squeezed, excluded)
  }

  /** The default connection operators for Chisel hardware components
    *
    * @define colonHashEq The "mono-direction connection operator", aka the "coercion operator".
    *
    * For `consumer :#= producer`, all leaf members of consumer (regardless of relative flip) are driven by the corresponding leaf members of producer (regardless of relative flip)
    *
    * Identical to calling :<= and :>=, but swapping consumer/producer for :>= (order is irrelevant), e.g.:
    *   consumer :<= producer
    *   producer :>= consumer
    *
    * Symbol reference:
    *  - ':' is the consumer side
    *  - '=' is the producer side
    *  - '#' means to ignore flips, always drive from producer to consumer
    *
    * $chiselTypeRestrictions
    *
    * Additional notes:
    * - Connecting two `util.DecoupledIO`'s would connect `bits`, `valid`, AND `ready` from producer to consumer (despite `ready` being flipped)
    * - Functionally equivalent to chisel3.:=, but different than Chisel.:=
    *
    * @group connection
    *
    * @define colonLessEq The "aligned connection operator" between a producer and consumer.
    *
    * For `consumer :<= producer`, each of `consumer`'s leaf members which are aligned with respect to `consumer` are driven from the corresponding `producer` leaf member.
    * Only `consumer`'s leaf/branch alignments influence the connection.
    *
    * Symbol reference:
    *  - ':' is the consumer side
    *  - '=' is the producer side
    *  - '<' means to connect from producer to consumer
    *
    * $chiselTypeRestrictions
    *
    * Additional notes:
    *  - Connecting two `util.DecoupledIO`'s would connect `bits` and `valid` from producer to consumer, but leave `ready` unconnected
    *
    * @group connection
    *
    * @define colonGreaterEq The "flipped connection operator", or the "backpressure connection operator" between a producer and consumer.
    *
    * For `consumer :>= producer`, each of `producer`'s leaf members which are flipped with respect to `producer` are driven from the corresponding consumer leaf member
    * Only `producer`'s leaf/branch alignments influence the connection.
    *
    * Symbol reference:
    *  - ':' is the consumer side
    *  - '=' is the producer side
    *  - '>' means to connect from consumer to producer
    *
    * $chiselTypeRestrictions
    *
    * Additional notes:
    *  - Connecting two `util.DecoupledIO`'s would connect `ready` from consumer to producer, but leave `bits` and `valid` unconnected
    *
    * @group connection
    *
    * @define colonLessGreaterEq The "bi-direction connection operator", aka the "tur-duck-en operator"
    *
    * For `consumer :<>= producer`, both producer and consumer leafs could be driving or be driven-to.
    * The `consumer`'s members aligned w.r.t. `consumer` will be driven by corresponding members of `producer`;
    * the `producer`'s members flipped w.r.t. `producer` will be driven by corresponding members of `consumer`
    *
    * Identical to calling `:<=` and `:>=` in sequence (order is irrelevant), e.g. `consumer :<= producer` then `consumer :>= producer`
    *
    * Symbol reference:
    *  - ':' is the consumer side
    *  - '=' is the producer side
    *  - '<' means to connect from producer to consumer
    *  - '>' means to connect from consumer to producer
    *
    * $chiselTypeRestrictions
    * - An additional type restriction is that all relative orientations of `consumer` and `producer` must match exactly.
    *
    * Additional notes:
    *  - Connecting two wires of `util.DecoupledIO` chisel type would connect `bits` and `valid` from producer to consumer, and `ready` from consumer to producer.
    *  - If the types of consumer and producer also have identical relative flips, then we can emit FIRRTL.<= as it is a stricter version of chisel3.:<>=
    *  - "turk-duck-en" is a dish where a turkey is stuffed with a duck, which is stuffed with a chicken; `:<>=` is a `:=` stuffed with a `<>`
    *
    * @define chiselTypeRestrictions The following restrictions apply:
    *  - The Chisel type of consumer and producer must be the "same shape" recursively:
    *    - All ground types are the same (UInt and UInt are same, SInt and UInt are not), but widths can be different (implicit trunction/padding occurs)
    *    - All vector types are the same length
    *    - All bundle types have the same member names, but the flips of members can be different between producer and consumer
    *  - The leaf members that are ultimately assigned to, must be assignable. This means they cannot be module inputs or instance outputs.
    */
  private[chisel3] trait ConnectableDocs

  /** Create Connectable for consumer and producer whose unmatched members are waived
    *
    * @param consumer the consumer from whom to waive unmatched members
    * @param producer the producer from whom to waive unmatched members
    */
  def waiveUnmatched[T <: Data](
    consumer: Connectable[T],
    producer: Connectable[T]
  ): (Connectable[T], Connectable[T]) = {
    val result = DataMirror.collectMembersOverAllForAny(Some((consumer.base: Data)), Some((producer.base: Data))) {
      case x @ (Some(c), None) => x
      case x @ (None, Some(p)) => x
    }
    val cWaived = result.map(_._1).flatten
    val pWaived = result.map(_._2).flatten
    (consumer.copy(waived = cWaived.toSet), producer.copy(waived = pWaived.toSet))
  }

  implicit class ConnectableOpExtension[T <: Data](consumer: Connectable[T]) extends ConnectableDocs {
    import Connection.connect

    /** $colonLessEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=[S <: Data](lProducer: => S)(implicit evidence: T =:= S, sourceInfo: SourceInfo): Unit = {
      val producer = prefix(consumer.base) { lProducer }
      connect(consumer, producer, ColonLessEq)
    }

    /** $colonLessEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=[S <: Data](producer: Connectable[S])(implicit evidence: T =:= S, sourceInfo: SourceInfo): Unit = {
      prefix(consumer.base) {
        connect(consumer, producer, ColonLessEq)
      }
    }

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[S <: Data](lProducer: => S)(implicit evidence: T =:= S, sourceInfo: SourceInfo): Unit = {
      val producer = prefix(consumer.base) { lProducer }
      connect(consumer, producer, ColonGreaterEq)
    }

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[S <: Data](producer: Connectable[S])(implicit evidence: T =:= S, sourceInfo: SourceInfo): Unit = {
      prefix(consumer.base) {
        connect(consumer, producer, ColonGreaterEq)
      }
    }

    /** There are cases when we need to reverse a FIRRTL connection
      *
      * Namely, OpaqueTypes of flipped fields since the flip is not present in the emitted FIRRTL
      * @todo when refactoring internals, this firrtl-specific behavior should probably be moved
      *   into Chisel IR lowering to FIRRTL IR. Such a change will likely conflict with the logic in
      *   BiConnect which uses AbsoluteDirection to "Do the right thing"TM.
      */
    private def doFirrtlConnect[S <: Data](consumer: T, producer: S)(implicit sourceInfo: SourceInfo): Unit = {
      val flip = consumer match {
        case rec: Record if rec._isOpaqueType =>
          rec.elementsIterator.next().specifiedDirection == SpecifiedDirection.Flip
        case _ => false
      }
      val (lhs, rhs) = if (flip) (producer, consumer) else (consumer, producer)
      lhs.firrtlConnect(rhs)
    }

    /** $colonLessGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=[S <: Data](lProducer: => S)(implicit evidence: T =:= S, sourceInfo: SourceInfo): Unit = {
      val producer = prefix(consumer.base) { lProducer }
      if (ColonLessGreaterEq.canFirrtlConnect(consumer, producer)) {
        doFirrtlConnect(consumer.base, producer)
      } else {
        connect(consumer, producer, ColonLessGreaterEq)
      }
    }

    /** $colonLessGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=[S <: Data](producer: Connectable[S])(implicit evidence: T =:= S, sourceInfo: SourceInfo): Unit = {
      prefix(consumer.base) {
        if (ColonLessGreaterEq.canFirrtlConnect(consumer, producer)) {
          doFirrtlConnect(consumer.base, producer.base)
        } else {
          connect(consumer, producer, ColonLessGreaterEq)
        }
      }
    }

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    final def :#=[S <: Data](lProducer: => S)(implicit evidence: T =:= S, sourceInfo: SourceInfo): Unit = {
      val producer = prefix(consumer.base) { lProducer }
      connect(consumer, producer, ColonHashEq)
    }

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    final def :#=[S <: Data](producer: Connectable[S])(implicit evidence: T =:= S, sourceInfo: SourceInfo): Unit = {
      prefix(consumer.base) {
        connect(consumer, producer, ColonHashEq)
      }
    }

    /** $colonLessEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      connect(consumer, producer, ColonLessEq)
    }

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      connect(consumer, producer, ColonGreaterEq)
    }

    /** $colonLessGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      connect(consumer, producer, ColonLessGreaterEq)
    }

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    final def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      connect(consumer, producer, ColonHashEq)
    }

  }
}
