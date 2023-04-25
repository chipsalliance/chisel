// SPDX-License-Identifier: Apache-2.0

package chisel3.connectable

import chisel3.{Aggregate, BiConnectException, Data, DontCare, InternalErrorException, RawModule, Vec}
import chisel3.internal.{BiConnect, Builder}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.DefInvalid
import chisel3.experimental.{prefix, SourceInfo, UnlocatableSourceInfo}
import chisel3.experimental.{attach, Analog}
import chisel3.reflect.DataMirror.hasProbeTypeModifier
import Alignment.matchingZipOfChildren

import scala.collection.mutable

// Datastructure capturing the semantics of each connectable operator
private[chisel3] sealed trait Connection {
  val noDangles:               Boolean
  val noUnconnected:           Boolean
  val mustMatch:               Boolean
  val noWrongOrientations:     Boolean
  val noMismatchedWidths:      Boolean
  val connectToConsumer:       Boolean
  val connectToProducer:       Boolean
  val alwaysConnectToConsumer: Boolean
}

private[chisel3] case object ColonLessEq extends Connection {
  val noDangles:               Boolean = true
  val noUnconnected:           Boolean = true
  val mustMatch:               Boolean = true
  val noWrongOrientations:     Boolean = true
  val noMismatchedWidths:      Boolean = true
  val connectToConsumer:       Boolean = true
  val connectToProducer:       Boolean = false
  val alwaysConnectToConsumer: Boolean = false
}

private[chisel3] case object ColonGreaterEq extends Connection {
  val noDangles:               Boolean = true
  val noUnconnected:           Boolean = true
  val mustMatch:               Boolean = true
  val noWrongOrientations:     Boolean = true
  val noMismatchedWidths:      Boolean = true
  val connectToConsumer:       Boolean = false
  val connectToProducer:       Boolean = true
  val alwaysConnectToConsumer: Boolean = false
}

private[chisel3] case object ColonLessGreaterEq extends Connection {
  val noDangles:               Boolean = true
  val noUnconnected:           Boolean = true
  val mustMatch:               Boolean = true
  val noWrongOrientations:     Boolean = true
  val noMismatchedWidths:      Boolean = true
  val connectToConsumer:       Boolean = true
  val connectToProducer:       Boolean = true
  val alwaysConnectToConsumer: Boolean = false
  def canFirrtlConnect(consumer: Connectable[Data], producer: Connectable[Data]) = {
    val typeEquivalent =
      try {
        BiConnect.canFirrtlConnectData(
          consumer.base,
          producer.base,
          UnlocatableSourceInfo,
          Builder.referenceUserModule
        ) && consumer.base.typeEquivalent(producer.base)
      } catch {
        // For some reason, an error is thrown if its a View; since this is purely an optimization, any actual error would get thrown
        //  when calling DirectionConnection.connect. Hence, we can just default to false to take the non-optimized emission path
        case e: Throwable => false
      }
    (typeEquivalent && consumer.notWaivedOrSqueezedOrExcluded && producer.notWaivedOrSqueezedOrExcluded)
  }
}

private[chisel3] case object ColonHashEq extends Connection {
  val noDangles:               Boolean = true
  val noUnconnected:           Boolean = true
  val mustMatch:               Boolean = true
  val noWrongOrientations:     Boolean = false
  val noMismatchedWidths:      Boolean = true
  val connectToConsumer:       Boolean = true
  val connectToProducer:       Boolean = false
  val alwaysConnectToConsumer: Boolean = true
}

private[chisel3] object Connection {

  /** Connection function which implements both :<= and :>=
    *
    * For example, given a connection like so:
    *  consumer :<= producer
    * We can reason the following:
    *  - consumer is the consumerRoot
    *  - producer is the producerRoot
    *  - The '<' indicates that the consumer side (left hand side) is the active side
    *
    * @param consumerRoot the original expression on the left-hand-side of the connection operator
    * @param producerRoot the original expression on the right-hand-side of the connection operator
    * @param activeSide indicates if the connection was a :<= (consumer is active) or :>= (producer is active)
    * @param sourceInfo source info for where the connection occurred
    */
  def connect[T <: Data](
    cRoot: Connectable[T],
    pRoot: Connectable[T],
    cOp:   Connection
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    doConnection(cRoot, pRoot, cOp)
  }

  private def leafConnect(
    consumer:     Data,
    producer:     Data,
    alignment:    Alignment,
    connectionOp: Connection
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    (
      consumer,
      producer,
      alignment,
      connectionOp.connectToConsumer,
      connectionOp.connectToProducer,
      connectionOp.alwaysConnectToConsumer
    ) match {
      case (x: Analog, y: Analog, _, _, _, _) => connectAnalog(x, y)
      case (x: Analog, DontCare, _, _, _, _) => connectAnalog(x, DontCare)
      case (x, y, _: AlignedWithRoot, true, _, _) => consumer := producer
      case (x, y, _: FlippedWithRoot, _, true, _) => producer := consumer
      case (x, y, _, _, _, true) => consumer := producer
      case other                 =>
    }
  }

  private def connect(
    l: Data,
    r: Data
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    (l, r) match {
      case (x: Analog, y: Analog) => connectAnalog(x, y)
      case (x: Analog, DontCare) => connectAnalog(x, DontCare)
      case (_, _) => l := r
    }
  }

  private def doConnection[T <: Data](
    consumer:     Connectable[T],
    producer:     Connectable[T],
    connectionOp: Connection
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {

    var errors: List[String] = Nil
    import Alignment.deriveChildAlignment

    def doConnection(
      conAlign: Alignment,
      proAlign: Alignment
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      (conAlign, proAlign) match {
        // Base Case 0: should probably never happen
        case (_: EmptyAlignment, _: EmptyAlignment) => ()

        // Base Case 1: early exit if dangling/unconnected is excluded
        case (conAlign: Alignment, proAlign: Alignment) if conAlign.isExcluded && proAlign.isExcluded => ()

        // Base Case 2(A,B): early exit if dangling/unconnected is wavied or excluded
        case (conAlign: NonEmptyAlignment, _: EmptyAlignment) if conAlign.isWaived || conAlign.isExcluded => ()
        case (_: EmptyAlignment, proAlign: NonEmptyAlignment) if proAlign.isWaived || proAlign.isExcluded => ()

        // Base Case 3: early exit if dangling/unconnected is wavied
        case (conAlign: NonEmptyAlignment, proAlign: NonEmptyAlignment) if conAlign.isExcluded || proAlign.isExcluded =>
          val (excluded, included) =
            if (conAlign.isExcluded) (conAlign, proAlign)
            else (proAlign, conAlign)
          errors = (s"excluded field ${excluded.member} has matching non-excluded field ${included.member}") +: errors

        // Base Case 4: early exit if operator requires matching orientations, but they don't align
        case (conAlign: NonEmptyAlignment, proAlign: NonEmptyAlignment)
            if (!conAlign.alignsWith(proAlign)) && (connectionOp.noWrongOrientations) =>
          errors = (s"inversely oriented fields ${conAlign.member} and ${proAlign.member}") +: errors

        // Base Case 5: early exit if operator requires matching widths, but they aren't the same
        case (conAlign: NonEmptyAlignment, proAlign: NonEmptyAlignment)
            if (conAlign.truncationRequired(proAlign, connectionOp).nonEmpty) && (connectionOp.noMismatchedWidths) =>
          val mustBeTruncated = conAlign.truncationRequired(proAlign, connectionOp).get
          errors =
            (s"mismatched widths of ${conAlign.member} and ${proAlign.member} might require truncation of $mustBeTruncated") +: errors

        // Base Case 6: operator error on dangling/unconnected fields
        case (consumer: NonEmptyAlignment, _: EmptyAlignment) =>
          errors = (s"${consumer.errorWord(connectionOp)} consumer field ${conAlign.member}") +: errors
        case (_: EmptyAlignment, producer: NonEmptyAlignment) =>
          errors = (s"${producer.errorWord(connectionOp)} producer field ${proAlign.member}") +: errors

        // Recursive Case 4: non-empty orientations
        case (conAlign: NonEmptyAlignment, proAlign: NonEmptyAlignment) =>
          (conAlign.member, proAlign.member) match {
            // Check for zero-width Vectors: both Vecs must be type equivalent, e.g.
            // a UInt<8>[0] should not be connectable with a SInt<8>[0]
            // TODO: This is a "band-aid" fix and needs to be unified with the existing logic in a
            // more generalized and robust way
            case (consumer: Vec[Data @unchecked], producer: Vec[Data @unchecked])
                if (consumer.length == 0 && !consumer.typeEquivalent(producer)) =>
              errors =
                (s"Consumer (${consumer.cloneType.toString}) and producer (${producer.cloneType.toString}) have different types.") +: errors

            case (consumer: Aggregate, producer: Aggregate) =>
              matchingZipOfChildren(Some(conAlign), Some(proAlign)).foreach {
                case (ceo, peo) =>
                  doConnection(ceo.getOrElse(conAlign.empty), peo.getOrElse(proAlign.empty))
              }
            case (consumer: Aggregate, DontCare) =>
              consumer.getElements.foreach {
                case f =>
                  doConnection(
                    deriveChildAlignment(f, conAlign),
                    deriveChildAlignment(f, conAlign).swap(DontCare)
                  )
              }
            case (DontCare, producer: Aggregate) =>
              producer.getElements.foreach {
                case f =>
                  doConnection(
                    deriveChildAlignment(f, proAlign).swap(DontCare),
                    deriveChildAlignment(f, proAlign)
                  )
              }
            // Check that neither consumer nor producer contains probes
            case (consumer: Data, producer: Data)
                if (hasProbeTypeModifier(consumer) || hasProbeTypeModifier(producer)) =>
              errors = "Cannot use connectables with probe types. Omit them prior to connection." +: errors
            case (consumer, producer) =>
              val alignment = (
                conAlign.alignsWith(proAlign),
                (!conAlign.alignsWith(
                  proAlign
                ) && connectionOp.connectToConsumer && !connectionOp.connectToProducer),
                (!conAlign.alignsWith(
                  proAlign
                ) && !connectionOp.connectToConsumer && connectionOp.connectToProducer)
              ) match {
                case (true, _, _) => conAlign
                case (_, true, _) => conAlign
                case (_, _, true) => proAlign
                case other        => throw new Exception(other.toString)
              }
              val lAndROpt = alignment.computeLandR(consumer, producer, connectionOp)
              lAndROpt.foreach { case (l, r) => connect(l, r) }
          }
        case other => throw new Exception(other.toString + " " + connectionOp)
      }
    }

    // Start recursive connection
    doConnection(Alignment(consumer, true), Alignment(producer, false))

    // If any errors are collected, error.
    if (errors.nonEmpty) {
      Builder.error(errors.mkString("\n"))
    }
  }

  private def checkAnalog(as: Analog*)(implicit sourceInfo: SourceInfo): Unit = {
    val currentModule = Builder.currentModule.get.asInstanceOf[RawModule]
    try {
      as.toList match {
        case List(a) => BiConnect.markAnalogConnected(sourceInfo, a, DontCare, currentModule)
        case List(a, b) =>
          BiConnect.markAnalogConnected(sourceInfo, a, b, currentModule)
          BiConnect.markAnalogConnected(sourceInfo, b, a, currentModule)
        case _ => throw new InternalErrorException("Match error: as.toList=${as.toList}")
      }
    } catch { // convert Exceptions to Builder.error's so compilation can continue
      case attach.AttachException(message) => Builder.error(message)
      case BiConnectException(message)     => Builder.error(message)
    }
  }

  private def connectAnalog(a: Analog, b: Data)(implicit sourceInfo: SourceInfo): Unit = {
    b match {
      case (ba: Analog) => {
        checkAnalog(a, ba)
        val currentModule = Builder.currentModule.get.asInstanceOf[RawModule]
        attach.impl(Seq(a, ba), currentModule)(sourceInfo)
      }
      case (DontCare) => {
        checkAnalog(a)
        pushCommand(DefInvalid(sourceInfo, a.lref))
      }
    }
  }
}
