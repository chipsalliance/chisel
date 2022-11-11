// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.{prefix, BiConnect, Builder}
import chisel3.internal.sourceinfo.SourceInfo

package object connectable {

  import ConnectionFunctions.assign

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
    * $chiselTypeRestrictions
    *
    * Additional notes:
    * - Connecting two [[util.DecoupledIO]]'s would connect `bits`, `valid`, AND `ready` from producer to consumer (despite `ready` being flipped)
    * - Functionally equivalent to chisel3.:=, but different than Chisel.:=
    *
    * @group connection
    *
    * @define colonLessEq The "aligned connection operator" between a producer and consumer.
    *
    * For `consumer :<= producer`, each of `consumer`'s leaf members which are aligned with respect to `consumer` are driven from the corresponding `producer` leaf member.
    * Only `consumer`'s leaf/branch alignments influence the connection.
    *
    * $chiselTypeRestrictions
    *
    * Additional notes:
    *  - Connecting two [[util.DecoupledIO]]'s would connect `bits` and `valid` from producer to consumer, but leave `ready` unconnected
    *
    * @group connection
    *
    * @define colonGreaterEq The "flipped connection operator", or the "backpressure connection operator" between a producer and consumer.
    *
    * For `consumer :>= producer`, each of `producer`'s leaf members which are flipped with respect to `producer` are driven from the corresponding consumer leaf member
    * Only `producer`'s leaf/branch alignments influence the connection.
    *
    * $chiselTypeRestrictions
    *
    * Additional notes:
    *  - Connecting two [[util.DecoupledIO]]'s would connect `ready` from consumer to producer, but leave `bits` and `valid` unconnected
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
    * $chiselTypeRestrictions
    * - An additional type restriction is that all relative orientations of `consumer` and `producer` must match exactly.
    *
    * Additional notes:
    *  - Connecting two wires of [[util.DecoupledIO]] chisel type would connect `bits` and `valid` from producer to consumer, and `ready` from consumer to producer.
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
  trait ConnectableDocs

  /** ConnectableData Typeclass defines the following operators on all subclasses of Data: :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableDataOperators[T <: Data](consumer: T) extends ConnectableData.ConnectableForConnectableData(ConnectableData(consumer))

  /** ConnectableVec Typeclass defines the following operators on between a (consumer: Vec) and (producer: Seq): :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableVecOperators[T <: Data](consumer: Vec[T]) {

    /** $colonLessEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always drive leaf connections, and never get driven by leaf connections ("aligned connection")
      */
    def :<=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :<= b }
    }

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    def :>=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :>= b }
    }

    /** $colonLessGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection
      */
    def :<>=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :<>= b }
    }

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    def :#=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :#= b }
    }

    /** $colonHashEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection, all members will be driving, none will be driven-to
      */
    def :#=(producer: DontCare.type)(implicit sourceInfo: SourceInfo): Unit = {
      for (a <- consumer) { a :#= DontCare }
    }
  }

  implicit class ConnectableDontCare(consumer: DontCare.type) {

    /** $colonGreaterEq
      *
      * @group connection
      * @param producer the right-hand-side of the connection; will always be driven by leaf connections, and never drive leaf connections ("flipped connection")
      */
    final def :>=[T <: Data](producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        assign(consumer, producer, ColonGreaterEq)
      }
    }
  }
}
