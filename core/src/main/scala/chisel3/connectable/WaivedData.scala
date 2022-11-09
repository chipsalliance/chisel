// SPDX-License-Identifier: Apache-2.0

package chisel3
package connectable

import chisel3.internal.sourceinfo.SourceInfo
import experimental.{prefix, requireIsHardware}

/** A data for whom members if left dangling or unassigned with not trigger an error
  * A waived member will still be connected to if present in both producer and consumer
  * 
  * @param base The component being connected to
  * @param waivers members of base who will not trigger an error if left dangling or unassigned
  */
final case class WaivedData[+T <: Data](base: T, waivers: Set[Data]) {
  requireIsHardware(base, s"Can only created WaivedData of components, not unbound Chisel types")

  /** Select members of base to waive
    * 
    * @param members functions given the base return a member to waive
    */
  def waive(members: (T => Data)*): WaivedData[T] = this.copy(waivers = waivers ++ members.map(f => f(base)).toSet)

  /** Select members of base to waive and static cast to a new type
    * 
    * @param members functions given the base return a member to waive
    */
  def waiveAs[S <: Data](members: (T => Data)*): WaivedData[S] = this.copy(waivers = waivers ++ members.map(f => f(base)).toSet).asInstanceOf[WaivedData[S]]

  /** Programmatically select members of base to waive
    * 
    * @param members partial function applied to all recursive members of base, if match, can return a member to waive
    */
  def waiveEach[S <: Data](pf: PartialFunction[Data, Data]): WaivedData[T] = {
    val waivedMembers = DataMirror.collectDeep(base)(pf)
    this.copy(waivers = waivers ++ waivedMembers.toSet)
  }

  /** Waive all members of base */
  def waiveAll: WaivedData[T] = {
    val waivedMembers = DataMirror.collectDeep(base) { case x => x }
    this.copy(waivers = waivedMembers.toSet) // not appending waivers because we are collecting all members
  }
}

object WaivedData {
  def apply[T <: Data](base: T): WaivedData[T] = WaivedData(base, Set.empty[Data])

  /** Create WaivedData for consumer and producer whose unmatched members are waived
    *
    * @param consumer the consumer from whom to waive unmatched members
    * @param producer the producer from whom to waive unmatched members
    */
  def waiveUnmatched[T <: Data](consumer: T, producer: T): (WaivedData[T], WaivedData[T]) = {
    val result = DataMirror.collectDeepOverAllForAny(Some((consumer: Data)), Some((producer: Data))) {
      case x@(Some(c), None) => x
      case x@(None, Some(p)) => x
    }
    val cWaived = result.map(_._1).flatten
    val pWaived = result.map(_._2).flatten
    (WaivedData(consumer, cWaived.toSet), WaivedData(producer, pWaived.toSet))
  }

  implicit class ConnectableForWaivedData[T <: Data](wd: WaivedData[T]) extends connectable.ConnectableDocs {
    import ConnectionFunctions.assign

    val consumer = wd.base
    val cWaivers = wd.waivers

    /** $colonLessEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=[S <: Data](producer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        assign(consumer, producer, ColonLessEq, cWaivers, Set.empty[Data])
      }
    }

    /** $colonLessEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    final def :<=[S <: Data](pWaived: WaivedData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        assign(consumer, pWaived.base, ColonLessEq, cWaivers, pWaived.waivers)
      }
    }

    /** $colonGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[S <: Data](producer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        assign(consumer, producer, ColonGreaterEq, cWaivers, Set.empty[Data])
      }
    }

    /** $colonGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[S <: Data](pWaived: WaivedData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        assign(consumer, pWaived.base, ColonGreaterEq, cWaivers, pWaived.waivers)
      }
    }

    /** $colonLessGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=[S <: Data](producer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        // cannot call :<= and :>= directly because otherwise prefix is called twice
        assign(consumer, producer, ColonLessGreaterEq, cWaivers, Set.empty[Data])
      }
    }

    /** $colonLessGreaterEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    final def :<>=[S <: Data](pWaived: WaivedData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        assign(consumer, pWaived.base, ColonLessGreaterEq, cWaivers, pWaived.waivers)
      }
    }

    /** $colonHashEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    final def :#=[S <: Data](producer: => S)(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        assign(consumer, producer, ColonHashEq, cWaivers, Set.empty[Data])
      }
    }

    /** $colonHashEq
      * 
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    final def :#=[S <: Data](pWaived: WaivedData[S])(implicit evidence: S =:= T, sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        assign(consumer, pWaived.base, ColonHashEq, cWaivers, pWaived.waivers)
      }
    }
  }
}
