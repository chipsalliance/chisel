// SPDX-License-Identifier: Apache-2.0

package chisel3.connectable

import chisel3.{Aggregate, BiConnectException, Data, DontCare, RawModule}
import chisel3.internal.{prefix, BiConnect, Builder}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.DefInvalid
import chisel3.internal.sourceinfo.{SourceInfo, UnlocatableSourceInfo}
import chisel3.experimental.{attach, Analog}
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
          Connection.chisel5CompileOptions,
          Builder.referenceUserModule
        ) && consumer.base.typeEquivalent(producer.base)
      } catch {
        // For some reason, an error is thrown if its a View; since this is purely an optimization, any actual error would get thrown
        //  when calling DirectionConnection.connect. Hence, we can just default to false to take the non-optimized emission path
        case e: Throwable => false
      }
    (typeEquivalent && consumer.notSpecial && producer.notSpecial)
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
    *  c :<= p
    * We can reason the following:
    *  - c is the consumerRoot
    *  - p is the producerRoot
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

  // Consumed by the := operator, set to what chisel3 will eventually become.
  implicit val chisel5CompileOptions = new chisel3.CompileOptions {
    val connectFieldsMustMatch:      Boolean = true
    val declaredTypeMustBeUnbound:   Boolean = true
    val dontTryConnectionsSwapped:   Boolean = false
    val dontAssumeDirectionality:    Boolean = true
    val checkSynthesizable:          Boolean = true
    val explicitInvalidate:          Boolean = true
    val inferModuleReset:            Boolean = true
    override def emitStrictConnects: Boolean = true
  }

  private def leafConnect(
    c:  Data,
    p:  Data,
    o:  Alignment,
    op: Connection
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    (c, p, o, op.connectToConsumer, op.connectToProducer, op.alwaysConnectToConsumer) match {
      case (x: Analog, y: Analog, _, _, _, _) => connectAnalog(x, y)
      case (x: Analog, DontCare, _, _, _, _) => connectAnalog(x, DontCare)
      case (x, y, _: AlignedWithRoot, true, _, _) => c := p
      case (x, y, _: FlippedWithRoot, _, true, _) => p := c
      case (x, y, _, _, _, true) => c := p
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
    consumer: Connectable[T],
    producer: Connectable[T],
    op:       Connection
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {

    val errors = mutable.ArrayBuffer[String]()
    import Alignment.deriveChildAlignment

    def doConnection(co: Alignment, po: Alignment)(implicit sourceInfo: SourceInfo): Unit = {
      (co, po) match {
        // Base Case 0: should probably never happen
        case (_: EmptyAlignment, _: EmptyAlignment) => ()

        // Base Case 1: early exit if dangling/unconnected is wavied
        case (co: NonEmptyAlignment, _: EmptyAlignment) if co.isWaived => ()
        case (_: EmptyAlignment, po: NonEmptyAlignment) if po.isWaived => ()

        // Base Case 2: early exit if operator requires matching orientations, but they don't align
        case (co: NonEmptyAlignment, po: NonEmptyAlignment) if (!co.alignsWith(po)) && (op.noWrongOrientations) =>
          errors += (s"inversely oriented fields ${co.member} and ${po.member}")

        // Base Case 3: early exit if operator requires matching widths, but they aren't the same
        case (co: NonEmptyAlignment, po: NonEmptyAlignment)
            if (co.mismatchedWidths(po, op)) && (op.noMismatchedWidths) =>
          errors += (s"mismatched widths of ${co.member} and ${po.member}")

        // Base Case 3: operator error on dangling/unconnected fields
        case (c: NonEmptyAlignment, _: EmptyAlignment) => errors += (s"${c.errorWord(op)} consumer field ${co.member}")
        case (_: EmptyAlignment, p: NonEmptyAlignment) => errors += (s"${p.errorWord(op)} producer field ${po.member}")

        // Recursive Case 4: non-empty orientations
        case (co: NonEmptyAlignment, po: NonEmptyAlignment) =>
          (co.member, po.member) match {
            case (c: Aggregate, p: Aggregate) =>
              matchingZipOfChildren(Some(co), Some(po)).foreach {
                case (ceo, peo) => doConnection(ceo.getOrElse(co.empty), peo.getOrElse(po.empty))
              }
            case (c: Aggregate, DontCare) =>
              c.getElements.foreach {
                case f => doConnection(deriveChildAlignment(f, co), deriveChildAlignment(f, co).swap(DontCare))
              }
            case (DontCare, p: Aggregate) =>
              p.getElements.foreach {
                case f => doConnection(deriveChildAlignment(f, po).swap(DontCare), deriveChildAlignment(f, po))
              }
            case (c, p) =>
              val o = (
                co.alignsWith(po),
                (!co.alignsWith(po) && op.connectToConsumer && !op.connectToProducer),
                (!co.alignsWith(po) && !op.connectToConsumer && op.connectToProducer)
              ) match {
                case (true, _, _) => co
                case (_, true, _) => co
                case (_, _, true) => po
              }
              val lAndROpt = o.computeLandR(c, p, op)
              lAndROpt.map { case (l, r) => connect(l, r) }
          }
        case other => throw new Exception(other.toString + " " + op)
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
