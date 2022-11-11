// SPDX-License-Identifier: Apache-2.0

package chisel3
package connectable

import chisel3.internal.sourceinfo.SourceInfo
import experimental.{prefix, requireIsHardware}

/** A data for whom members if left dangling or unassigned with not trigger an error
  * A waived member will still be connected to if present in both producer and consumer
  *
  * @param base The component being connected to
  * @param waived members of base who will not trigger an error if left dangling or unassigned
  * @param squeezed members of base who will not trigger an error if would end up being truncated
  */
final case class ConnectableData[+T <: Data](base: T, waived: Set[Data], squeezed: Set[Data]) {
  requireIsHardware(base, s"Can only created ConnectableData of components, not unbound Chisel types")

  def notSpecial = waived.isEmpty && squeezed.isEmpty

  /** Select members of base to waive
    *
    * @param members functions given the base return a member to waive
    */
  def waive(members: (T => Data)*): ConnectableData[T] = this.copy(waived = waived ++ members.map(f => f(base)).toSet)

  /** Select members of base to waive and static cast to a new type
    *
    * @param members functions given the base return a member to waive
    */
  def waiveAs[S <: Data](members: (T => Data)*): ConnectableData[S] =
    this.copy(waived = waived ++ members.map(f => f(base)).toSet).asInstanceOf[ConnectableData[S]]

  /** Programmatically select members of base to waive
    *
    * @param members partial function applied to all recursive members of base, if match, can return a member to waive
    */
  def waiveEach[S <: Data](pf: PartialFunction[Data, Data]): ConnectableData[T] = {
    val waivedMembers = DataMirror.collectDeep(base)(pf)
    this.copy(waived = waived ++ waivedMembers.toSet)
  }

  /** Waive all members of base */
  def waiveAll: ConnectableData[T] = {
    val waivedMembers = DataMirror.collectDeep(base) { case x => x }
    this.copy(waived = waivedMembers.toSet) // not appending waived because we are collecting all members
  }

  /** Select members of base to squeeze
    *
    * @param members functions given the base return a member to squeeze
    */
  def squeeze(members: (T => Data)*): ConnectableData[T] = this.copy(squeezed = squeezed ++ members.map(f => f(base)).toSet)

  /** Programmatically select members of base to squeeze
    *
    * @param members partial function applied to all recursive members of base, if match, can return a member to squeeze
    */
  def squeezeEach[S <: Data](pf: PartialFunction[Data, Data]): ConnectableData[T] = {
    val squeezedMembers = DataMirror.collectDeep(base)(pf)
    this.copy(squeezed = squeezed ++ squeezedMembers.toSet)
  }

  /** Squeeze all members of base */
  def squeezeAll: ConnectableData[T] = {
    val squeezedMembers = DataMirror.collectDeep(base) { case x => x }
    this.copy(squeezed = squeezedMembers.toSet) // not appending squeezed because we are collecting all members
  }
}

object ConnectableData {
  def apply[T <: Data](base: T): ConnectableData[T] = ConnectableData(base, Set.empty[Data], Set.empty[Data])

  /** Create ConnectableData for consumer and producer whose unmatched members are waived
    *
    * @param consumer the consumer from whom to waive unmatched members
    * @param producer the producer from whom to waive unmatched members
    */
  def waiveUnmatched[T <: Data](consumer: ConnectableData[T], producer: ConnectableData[T]): (ConnectableData[T], ConnectableData[T]) = {
    val result = DataMirror.collectDeepOverAllForAny(Some((consumer.base: Data)), Some((producer.base: Data))) {
      case x @ (Some(c), None) => x
      case x @ (None, Some(p)) => x
    }
    val cWaived = result.map(_._1).flatten
    val pWaived = result.map(_._2).flatten
    (consumer.copy(waived = cWaived.toSet), producer.copy(waived = pWaived.toSet))
  }

  implicit class ConnectableForConnectableData[T <: Data](consumer: ConnectableData[T]) extends connectable.ConnectableDocs {
    import ConnectionFunctions.assign

    /** $colonLessEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=[S <: Data](lProducer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      val producer = prefix(consumer.base) { lProducer }
      assign(consumer, producer, ColonLessEq)
    }

    /** $colonLessEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=[S <: Data](producer: ConnectableData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer.base) {
        assign(consumer, producer, ColonLessEq)
      }
    }

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[S <: Data](lProducer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      val producer = prefix(consumer.base) { lProducer }
      assign(consumer, producer, ColonGreaterEq)
    }

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[S <: Data](producer: ConnectableData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer.base) {
        assign(consumer, producer, ColonGreaterEq)
      }
    }

    /** $colonLessGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=[S <: Data](lProducer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      val producer = prefix(consumer.base) { lProducer }
      if (ColonLessGreaterEq.canFirrtlConnect(consumer, producer)) {
        consumer.base.firrtlConnect(producer)
      } else {
        assign(consumer, producer, ColonLessGreaterEq)
      }
    }

    /** $colonLessGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=[S <: Data](producer: ConnectableData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer.base) {
        if (ColonLessGreaterEq.canFirrtlConnect(consumer, producer)) {
          consumer.base.firrtlConnect(producer.base)
        } else {
          assign(consumer, producer, ColonLessGreaterEq)
        }
      }
    }

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    final def :#=[S <: Data](lProducer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      val producer = prefix(consumer.base) { lProducer }
      assign(consumer, producer, ColonHashEq)
    }

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    final def :#=[S <: Data](producer: ConnectableData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer.base) {
        assign(consumer, producer, ColonHashEq)
      }
    }

    /** $colonLessEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      assign(consumer, producer, ColonLessEq)
    }

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      assign(consumer, producer, ColonGreaterEq)
    }

    /** $colonLessGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      assign(consumer, producer, ColonLessGreaterEq)
    }

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    final def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      assign(consumer, producer, ColonHashEq)
    }

  }
}
