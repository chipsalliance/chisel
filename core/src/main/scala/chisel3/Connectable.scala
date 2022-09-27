// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.{prefix, BiConnect, Builder}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.{Analog, DataMirror}

/** The default connection operators for Chisel hardware components */
object Connectable {

  /** ConnectableData Typeclass defines the following operators on all subclasses of Data: :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableData[T <: Data](consumer: T) {

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
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ConsumerIsActive)
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
        DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ProducerIsActive)
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
      *
      * @param consumer the left-hand-side of the connection (read above comment for more info)
      * @param producer the right-hand-side of the connection (read above comment for more info)
      * @param sourceInfo
      * @group connection
      */
    final def :<>=(producer: => T)(implicit sourceInfo: SourceInfo): Unit = {
      prefix(consumer) {
        val canFirrtlConnect =
          try {
            BiConnect.canFirrtlConnectData(
              consumer,
              producer,
              sourceInfo,
              DirectionalConnectionFunctions.compileOptions,
              Builder.referenceUserModule
            )
          } catch {
            // For some reason, an error is thrown if its a View; since this is purely an optimization, any actual error would get thrown
            //  when calling DirectionConnectionFunctions.assign. Hence, we can just default to false to take the non-optimized emission path
            case e: Throwable => false
          }
        if (canFirrtlConnect) {
          consumer.firrtlConnect(producer)
        } else {
          // cannot call :<= and :>= directly because otherwise prefix is called twice
          DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ProducerIsActive)
          DirectionalConnectionFunctions.assign(consumer, producer, DirectionalConnectionFunctions.ConsumerIsActive)
        }
      }
    }

    /** The "mono-direction connection operator", aka the "coercion operator"
      *
      * For `consumer :#= producer`, all leaf fields of consumer (regardless of relative flip) are driven by the corresponding leaf fields of producer (regardless of relative flip)
      *
      * Identical to calling :<= and :>=, but swapping consumer/producer for :>=: (order is irrelevant), e.g.:
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
      consumer.:<=(producer)
      producer.:>=(consumer)
    }
  }

  /** ConnectableVec Typeclass defines the following operators on between a (consumer: Vec) and (producer: Seq): :<=, :>=, :<>=, :#=
    *
    * @param consumer the left-hand-side of the connection
    */
  implicit class ConnectableVec[T <: Data](consumer: Vec[T]) {

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
    def :<=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :<= b }
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
    def :>=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :>= b }
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
      *
      * @param consumer the left-hand-side of the connection (read above comment for more info)
      * @param producer the right-hand-side of the connection (read above comment for more info)
      * @param sourceInfo
      * @group connection
      */
    def :<>=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :<>= b }
    }

    /** The "mono-direction connection operator", aka the "coercion operator"
      *
      * For `consumer :#= producer`, all leaf fields of consumer (regardless of relative flip) are driven by the corresponding leaf fields of producer (regardless of relative flip)
      *
      * Identical to calling :<= and :>=, but swapping consumer/producer for :>=: (order is irrelevant), e.g.:
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
    def :#=(producer: Seq[T])(implicit sourceInfo: SourceInfo): Unit = {
      if (consumer.length != producer.length)
        Builder.error(
          s"Vec (size ${consumer.length}) and Seq (size ${producer.length}) being connected have different lengths!"
        )
      for ((a, b) <- consumer.zip(producer)) { a :#= b }
    }
  }
}

private[chisel3] object DirectionalConnectionFunctions {
  // Consumed by the := operator, set to what chisel3 will eventually become.
  implicit val compileOptions = new CompileOptions {

    val connectFieldsMustMatch:      Boolean = true
    val declaredTypeMustBeUnbound:   Boolean = true
    val dontTryConnectionsSwapped:   Boolean = false
    val dontAssumeDirectionality:    Boolean = true
    val checkSynthesizable:          Boolean = true
    val explicitInvalidate:          Boolean = true
    val inferModuleReset:            Boolean = true
    override def emitStrictConnects: Boolean = true
  }

  // Indicates whether the active side is aligned or flipped relative to the active side's root
  sealed trait RelativeOrientation { def invert: RelativeOrientation }
  case object AlignedWithRoot extends RelativeOrientation { def invert = FlippedWithRoot }
  case object FlippedWithRoot extends RelativeOrientation { def invert = AlignedWithRoot }

  // Indicates whether the consumer or producer is the active side
  sealed trait ActiveSide
  case object ProducerIsActive extends ActiveSide
  case object ConsumerIsActive extends ActiveSide

  /** Assignment function which implements both :<= and :>=
    *
    * For example, given a connection like so:
    *  c :<= p
    * We can reason the following:
    *  - c is the consumerRoot
    *  - p is the producerRoot
    *  - The '<' indicates that the consumer side (left hand side) is the active side
    *
    * @param consumerRoot the original expression on the left-hand-side of the connection operator
    * @param producerRoot the original expression on the right-hand-side of the connection operator
    * @param activeSide indicates if the connection was a :<= (consumer is active) or :>= (producer is active)
    * @param sourceInfo source info for where the assignment occurred
    */
  def assign(consumerRoot: Data, producerRoot: Data, activeSide: ActiveSide)(implicit sourceInfo: SourceInfo): Unit = {

    val activeRoot = if (activeSide == ProducerIsActive) producerRoot else consumerRoot
    require(
      activeRoot != DontCare,
      s"Cannot have the active side be a DontCare! Use _ := DontCare or _ :<= DontCare or DontCare :>= _"
    )

    /** Determines the aligned/flipped of activeSubMember with respect to activeRoot
      *
      * Due to Chisel/chisel3 differences, its a little complicated to calculate the RelativeOrientation, as the information
      *   is captured with both ActualDirection and SpecifiedDirection. Fortunately, all this complexity is captured in this
      *   one function.
      *
      * References activeRoot, defined earlier in the function
      *
      * @param activeSubMember a subfield/subindex of activeRoot (or sub-sub, or sub-sub-sub etc)
      * @param orientation aligned/flipped of d's direct parent aggregate with respect to activeRoot
      * @return orientation aligned/flipped of d with respect to activeRoot
      */
    def deriveOrientation(activeSubMember: Data, orientation: RelativeOrientation): RelativeOrientation = {
      (activeRoot.direction, activeSubMember.direction, DataMirror.specifiedDirectionOf(activeSubMember)) match {
        case (ActualDirection.Output, ActualDirection.Output, _) => AlignedWithRoot
        case (ActualDirection.Input, ActualDirection.Input, _)   => AlignedWithRoot
        case (ActualDirection.Output, ActualDirection.Input, _)  => FlippedWithRoot
        case (ActualDirection.Input, ActualDirection.Output, _)  => FlippedWithRoot
        case (_, _, SpecifiedDirection.Unspecified)              => orientation
        case (_, _, SpecifiedDirection.Flip)                     => orientation.invert
        case (_, _, SpecifiedDirection.Output)                   => orientation
        case (_, _, SpecifiedDirection.Input)                    => orientation.invert
        case other                                               => throw new Exception(s"Unexpected internal error! $other")
      }
    }

    /** Recurses down our consumer and producer to connect leaf subfield/subindexes, depending on the final orientation
      *
      * @param consumer subfield/subindex of consumerRoot (or just is, for the first recursive case)
      * @param producer subfield/subindex of producerRoot (or just is, for the first recursive case)
      * @param orientation whether the active side's subfield/subindex (either consumer or producer) is flipped/aligned relative to activeRoot
      */
    def recursiveAssign(consumer: Data, producer: Data, orientation: RelativeOrientation): Unit = {
      (consumer, producer) match {
        case (vc: Vec[_], vp: Vec[_]) => {
          val (active, inactive) = if (activeSide == ProducerIsActive) (vp, vc) else (vc, vp)
          val (defaultableSubIndexes, unassignedSubIndexes) = if (active.size > inactive.size) {
            active(0) match {
              case d: experimental.Defaulting[_] =>
                ((inactive.size until active.size).map { i => active(i) }, Nil)
              case o =>
                (Nil, (inactive.size until active.size).map { i => active(i) })
            }
          } else (Nil, Nil)
          if (unassignedSubIndexes.nonEmpty) {
            Builder.error(s"Connection has unassigned subindexes ${unassignedSubIndexes.mkString(", ")}")
          } else {
            (vc.zip(vp)).foreach { case (ec, ep) => recursiveAssign(ec, ep, orientation) }
            defaultableSubIndexes.map {
              case d: experimental.Defaulting[Data] =>
                if (activeSide == ProducerIsActive) recursiveAssign(d.default, d.underlying, orientation)
                else recursiveAssign(d.underlying, d.default, orientation)
            }
          }
        }
        case (rc: Record, rp: Record) => {
          val (active, inactive) = if (activeSide == ProducerIsActive) (rp, rc) else (rc, rp)
          val (defaultableKeys, unassignableKeys) = {
            val missingKeys = active.elements.keySet -- inactive.elements.keySet
            missingKeys.partition { k => active.elements(k).isInstanceOf[experimental.Defaulting[_]] }
          }
          if (unassignableKeys.nonEmpty) {
            val unassignableFields = unassignableKeys.map(k => active.elements(k))
            Builder.error(
              s"Connection between $consumer and $producer has unassigned fields ${unassignableFields.mkString(", ")} in $active."
            )
          } else {
            active.asInstanceOf[Record].elements.foreach {
              case (key, ea: experimental.Defaulting[Data]) if defaultableKeys.contains(key) =>
                val field = active.elements(key).asInstanceOf[experimental.Defaulting[Data]]
                val elementOrientation = deriveOrientation(ea, orientation)
                if (activeSide == ProducerIsActive) recursiveAssign(ea.default, ea.underlying, elementOrientation)
                else recursiveAssign(ea.underlying, ea.default, elementOrientation)
              case (key, ea) =>
                val elementOrientation = deriveOrientation(ea, orientation)
                val ec = rc.elements(key)
                val ep = rp.elements(key)
                recursiveAssign(ec, ep, elementOrientation)
            }
          }
        }
        // Active side must be vc due to earlier requirement
        case (vc: Vec[_], DontCare) => vc.foreach { case ec => recursiveAssign(ec, DontCare, orientation) }
        // Active side must be vp due to earlier requirement
        case (DontCare, vp: Vec[_]) => vp.foreach { case ep => recursiveAssign(DontCare, ep, orientation) }
        // Active side must be rc due to earlier requirement
        case (rc: Record, DontCare) =>
          rc.elements.foreach { case (_, ec) => recursiveAssign(ec, DontCare, deriveOrientation(ec, orientation)) }
        // Active side must be rp due to earlier requirement
        case (DontCare, rp: Record) =>
          rp.elements.foreach { case (_, ep) => recursiveAssign(DontCare, ep, deriveOrientation(ep, orientation)) }

        // Analog cases
        case (ac: Analog, ap: Analog) => assignAnalog(ac, ap)
        case (ac: Analog, DontCare) => assignAnalog(ac, DontCare)
        case (DontCare, ap: Analog) => assignAnalog(ap, DontCare)

        // Only non-Analog Ground types are left
        case _ =>
          (orientation, activeSide) match {
            case (AlignedWithRoot, ConsumerIsActive) => consumer := producer // assign leaf fields (UInt/etc)
            case (FlippedWithRoot, ProducerIsActive) => producer := consumer // assign leaf fields (UInt/etc)
            case _                                   =>
          }
      }
    }

    // For the base recursive case, we start with the roots and they are aligned with themselves
    recursiveAssign(consumerRoot, producerRoot, AlignedWithRoot)
  }

  def checkAnalog(as: Analog*)(implicit sourceInfo: SourceInfo): Unit = {
    val currentModule = Builder.currentModule.get.asInstanceOf[RawModule]
    try {
      as.foreach { a => BiConnect.markAnalogConnected(sourceInfo, a, currentModule) }
    } catch { // convert attach exceptions to BiConnectExceptions
      case experimental.attach.AttachException(message) => Builder.error(message)
    }
  }

  def assignAnalog(a: Analog, b: Data)(implicit sourceInfo: SourceInfo): Unit = b match {
    case (ba: Analog) => {
      checkAnalog(a, ba)
      val currentModule = Builder.currentModule.get.asInstanceOf[RawModule]
      experimental.attach.impl(Seq(a, ba), currentModule)(sourceInfo)
    }
    case (DontCare) => {
      checkAnalog(a)
      pushCommand(DefInvalid(sourceInfo, a.lref))
    }
  }
}
