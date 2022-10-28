// SPDX-License-Identifier: Apache-2.0

package chisel3
package experimental

import chisel3.internal.prefix 
import chisel3.internal.sourceinfo.SourceInfo

final case class WaivedData[T <: Data](d: T, waivers: Set[Data])

object WaivedData {

  def waiveUnmatched[T <: Data](consumer: T, producer: T): (WaivedData[T], WaivedData[T]) = {
    def apply(l: Option[Data], r: Option[Data]): Option[(Option[Data], Option[Data])] = (l, r) match {
      case x@(Some(c), None) => Some(x)
      case x@(None, Some(p)) => Some(x)
      case other => None
    }
    val result = DataMirror.collectDeepOverAllForAny(Some(consumer), Some(producer))(apply _)
    val cWaived = result.map(_._1).flatten
    val pWaived = result.map(_._2).flatten
    (WaivedData(consumer, cWaived.toSet), WaivedData(producer, pWaived.toSet))
  }

  implicit class WaivableData[T <: Data](d: T) {
    def waive(fields: (T => Data)*): WaivedData[T] = WaivedData(d, fields.map(f => f(d)).toSet)

    def waiveAs[S <: Data](fields: (T => Data)*): WaivedData[S] = WaivedData(d, fields.map(f => f(d)).toSet).asInstanceOf[WaivedData[S]]

    def waiveAll[S <: Data](pf: PartialFunction[Data, Data]): WaivedData[T] = {
      val waivedMembers = DataMirror.collectDeep(d)(pf)
      WaivedData(d, waivedMembers.toSet)
    }
  }

  implicit class ConnectableForWaivedData[T <: Data](wd: WaivedData[T]) {
    val consumer = wd.d
    val cWaivers = wd.waivers

    /** The "aligned connection operator" between a producer and consumer.
      *
      * For `consumer :<= producer`, each of consumer's leaf fields WHO ARE ALIGNED WITH RESPECT TO CONSUMER are driven from the corresponding producer leaf field
      * All producer's leaf/branch alignments (with respect to producer) do not influence the connection.
      *
      * The following restrictions apply:
      *  - The Chisel type of consumer and producer must be the "same shape" recursively:
      *    - All ground types are the same (UInt and UInt are same, SInt and UInt are not), but widths can be different
      *    - All vector types are the same length
      *    - All bundle types have the same field names, but the flips of fields can be different between producer and consumer
      *  - The leaf fields that are ultimately assigned to, must be assignable. This means they cannot be module inputs or instance outputs.
      *
      * @note Connecting two [[Decoupled]]'s would connect `bits` and `valid` from producer to consumer, but leave `ready` unconnected
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      *
      * @param consumer the left-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connection ("aligned connection")
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      * @param sourceInfo
      * @group connection
      */
    final def :<=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessEq, cWaivers, Set.empty[Data])
      }
    }
    final def :<=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonLessEq, cWaivers, pWaived.waivers)
      }
    }

    /** The "flipped connection operator" between a producer and consumer.
      *
      * For `consumer :>= producer`, each of producers's leaf fields WHO ARE FLIPPED WITH RESPECT TO PRODUCER are driven from the corresponding consumer leaf field
      * All consumer's leaf/branch alignments (with respect to consumer) do not influence the connection.
      *
      * The following restrictions apply:
      *  - The Chisel type of consumer and producer must be the "same shape":
      *    - All ground types are the same (UInt and UInt are same, SInt and UInt are not), but widths can be different
      *    - All vector types are the same length
      *    - All bundle types have the same field names, but the flips of fields can be different
      *  - The leaf fields that are ultimately assigned to, must be assignable. This means they cannot be module inputs or instance outputs.
      *
      * @note Connecting two [[Decoupled]]'s would connect `ready` from consumer to producer, but leave `bits` and `valid` unconnected
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      *
      * @param consumer the left-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("flipped connection")
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      * @param sourceInfo
      * @group connection
      */
    final def :>=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonGreaterEq, cWaivers, Set.empty[Data])
      }
    }
    final def :>=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonGreaterEq, cWaivers, pWaived.waivers)
      }
    }

    /** The "bi-direction connection operator", aka the "tur-duck-en operator"
      *
      * For `consumer :<>= producer`, both producer and consumer leafs could be driving or be driven-to:
      *   - consumer's fields aligned w.r.t. consumer will be driven by corresponding fields of producer
      *   - producer's fields flipped w.r.t. producer will be driven by corresponding fields of consumer
      *
      * Identical to calling both :<= and :>= in sequence (order is irrelevant), e.g.:
      *   consumer :<= producer
      *   consumer :>= producer
      *
      * @note Connecting two [[Decoupled]]'s would connect `bits` and `valid` from producer to consumer, and `ready` from consumer to producer.
      * @note This may have surprising-to-new-users behavior if the flips of consumer and producer do not match. Save yourself the headache and internalize what
      * :<= and :>= do, and then you'll be able to reason your way to understanding what's happening :)
      * @note If the types of consumer and producer also have identical relative flips, then we can emit FIRRTL.<= as it is a stricter version of chisel3.:<>=
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      * @note "turk-duck-en" is a meme where a turkey is stuffed with a duck, which is stuffed with a chicken; `:<>=` is a `:=` stuffed with a `<>`
      *
      * @param consumer the left-hand-side of the connection (read above comment for more info)
      * @param producer the right-hand-side of the connection (read above comment for more info)
      * @param sourceInfo
      * @group connection
      */
    final def :<>=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        // cannot call :<= and :>= directly because otherwise prefix is called twice
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonLessGreaterEq, cWaivers, Set.empty[Data])
      }
    }
    final def :<>=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonLessGreaterEq, cWaivers, pWaived.waivers)
      }
    }

    /** The "mono-direction connection operator", aka the "coercion operator"
      *
      * For `consumer :#= producer`, all leaf fields of consumer (regardless of relative flip) are driven by the corresponding leaf fields of producer (regardless of relative flip)
      *
      * Identical to calling :<= and :>=, but swapping consumer/producer for :>= (order is irrelevant), e.g.:
      *   consumer :<= producer
      *   producer :>= consumer
      *
      * @note Connecting two [[Decoupled]]'s would connect `bits`, `valid`, AND `ready` from producer to consumer (despite `ready` being flipped)
      * @note Functionally equivalent to chisel3.:=, but different than Chisel.:=
      * @note If the widths differ between consumer/producer, the assignment will still occur and truncation, if necessary, is implicit
      *
      * @param consumer the left-hand-side of the connection, all fields will be driven-to
      * @param producer the right-hand-side of the connection, all fields will be driving, none will be driven-to
      * @param sourceInfo
      * @group connection
      */
    final def :#=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ColonHashEq, cWaivers, Set.empty[Data])
      }
    }
    final def :#=(pWaived: WaivedData[T])(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        DirectionalConnectionFunctions.assign(consumer, pWaived.d, DirectionalConnectionFunctions.ColonHashEq, cWaivers, pWaived.waivers)
      }
    }

  }
}