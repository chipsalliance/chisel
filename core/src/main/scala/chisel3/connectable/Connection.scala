// SPDX-License-Identifier: Apache-2.0

package chisel3.connectable

import chisel3.{Aggregate, BiConnectException, Data, DontCare, InternalErrorException, RawModule}
import chisel3.internal.{BiConnect, Builder}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.DefInvalid
import chisel3.experimental.{prefix, SourceInfo, UnlocatableSourceInfo}
import chisel3.experimental.{attach, Analog}
import chisel3.reflect.DataMirror.hasProbeTypeModifier
import ConnectableAlignment.matchingZipOfChildren

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

  private def connect(
    l: Data,
    r: Data
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    try {
      (l, r) match {
        case (x: Analog, y: Analog) => connectAnalog(x, y)
        case (x: Analog, DontCare) => connectAnalog(x, DontCare)
        case (_, _) => l := r
      }
    } catch {
      case e: Exception => Builder.error(e.getMessage)
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
    import ConnectableAlignment.deriveChildAlignment

    def doConnection(
      conAlign: ConnectableAlignment,
      proAlign: ConnectableAlignment
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      (conAlign.align, proAlign.align) match {
        // Base Case 0: should probably never happen
        case (_: EmptyAlignment, _: EmptyAlignment) => ()

        // Base Case 1: early exit if dangling/unconnected is excluded
        case (_: Alignment, _: Alignment) if conAlign.isExcluded && proAlign.isExcluded => ()

        // Base Case 2(A,B): early exit if dangling/unconnected is wavied or excluded
        case (_: NonEmptyAlignment, _: EmptyAlignment) if conAlign.isWaived || conAlign.isExcluded => ()
        case (_: EmptyAlignment, _: NonEmptyAlignment) if proAlign.isWaived || proAlign.isExcluded => ()

        // Base Case 3: early exit if dangling/unconnected is wavied
        case (_: NonEmptyAlignment, _: NonEmptyAlignment) if conAlign.isExcluded || proAlign.isExcluded =>
          val (excluded, included) =
            if (conAlign.isExcluded) (conAlign, proAlign)
            else (proAlign, conAlign)
          errors = (s"excluded field ${excluded.member} has matching non-excluded field ${included.member}") +: errors

        // Base Case 4: early exit if operator requires matching orientations, but they don't align
        case (_: NonEmptyAlignment, _: NonEmptyAlignment)
            if (!conAlign.alignsWith(proAlign)) && (connectionOp.noWrongOrientations) =>
          errors = (s"inversely oriented fields ${conAlign.member} and ${proAlign.member}") +: errors

        // Base Case 5: early exit if operator requires matching widths, but they aren't the same
        case (_: NonEmptyAlignment, _: NonEmptyAlignment)
            if (conAlign.truncationRequired(proAlign, connectionOp).nonEmpty) && (connectionOp.noMismatchedWidths) =>
          val mustBeTruncated = conAlign.truncationRequired(proAlign, connectionOp).get
          errors =
            (s"mismatched widths of ${conAlign.member} and ${proAlign.member} might require truncation of $mustBeTruncated") +: errors

        // Base Case 6: operator error on dangling/unconnected fields
        case (_: NonEmptyAlignment, _: EmptyAlignment) =>
          errors = (s"${conAlign.errorWord(connectionOp)} consumer field ${conAlign.member}") +: errors
        case (_: EmptyAlignment, _: NonEmptyAlignment) =>
          errors = (s"${proAlign.errorWord(connectionOp)} producer field ${proAlign.member}") +: errors

        // Recursive Case 4: non-empty orientations
        case (_: NonEmptyAlignment, _: NonEmptyAlignment) =>
          (conAlign.member, proAlign.member) match {
            case (consumer: Aggregate, producer: Aggregate)
                if !hasProbeTypeModifier(consumer) && !hasProbeTypeModifier(producer) =>
              matchingZipOfChildren(Some(conAlign), Some(proAlign)).foreach {
                case (ceo, peo) =>
                  doConnection(ceo.getOrElse(conAlign.empty), peo.getOrElse(proAlign.empty))
              }
            case (consumer: Aggregate, DontCare) if !hasProbeTypeModifier(consumer) =>
              consumer.getElements.foreach {
                case f =>
                  doConnection(
                    deriveChildAlignment(f, conAlign),
                    deriveChildAlignment(f, conAlign).swap(DontCare)
                  )
              }
            case (DontCare, producer: Aggregate) if !hasProbeTypeModifier(producer) =>
              producer.getElements.foreach {
                case f =>
                  doConnection(
                    deriveChildAlignment(f, proAlign).swap(DontCare),
                    deriveChildAlignment(f, proAlign)
                  )
              }
            // Check that neither consumer nor producer contains probes
            case (consumer: Data, producer: Data)
                if (hasProbeTypeModifier(consumer) ^ hasProbeTypeModifier(producer)) =>
              errors = s"mismatched probe/non-probe types in ${consumer} and ${producer}." +: errors
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
    doConnection(ConnectableAlignment(consumer, true), ConnectableAlignment(producer, false))

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
