// SPDX-License-Identifier: Apache-2.0

package chisel3.connectable

import chisel3.{Aggregate, BiConnectException, Data, DontCare, RawModule}
import chisel3.internal.{prefix, BiConnect, Builder}
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.DefInvalid
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.{attach, Analog}
import Alignment.matchingZipOfChildren

import scala.collection.mutable

private[chisel3] object ConnectionFunctions {

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
  def assign[T <: Data](
    cRoot:    ConnectableData[T],
    pRoot:    ConnectableData[T],
    cOp:      ConnectionOperator
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    doAssignment(cRoot, pRoot, cOp)
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
    op: ConnectionOperator
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    (c, p, o, op.assignToConsumer, op.assignToProducer, op.alwaysAssignToConsumer) match {
      case (x: Analog, y: Analog, _, _, _, _) => assignAnalog(x, y)
      case (x: Analog, DontCare, _, _, _, _) => assignAnalog(x, DontCare)
      case (x, y, _: AlignedWithRoot, true, _, _) => c := p
      case (x, y, _: FlippedWithRoot, _, true, _) => p := c
      case (x, y, _, _, _, true) => c := p
      case other                 =>
    }
  }

  private def connect(
    l:  Data,
    r:  Data,
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    (l, r) match {
      case (x: Analog, y: Analog) => assignAnalog(x, y)
      case (x: Analog, DontCare) => assignAnalog(x, DontCare)
      case (_, _) => l := r
    }
  }


  private def doAssignment[T <: Data](
    consumer: ConnectableData[T],
    producer: ConnectableData[T],
    op:       ConnectionOperator
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {

    val errors = mutable.ArrayBuffer[String]()
    import Alignment.deriveChildAlignment

    def doAssignment(co: Alignment, po: Alignment)(implicit sourceInfo: SourceInfo): Unit = {
      (co, po) match {
        // Base Case 0: should probably never happen
        case (_: EmptyAlignment, _: EmptyAlignment) => ()

        // Base Case 1: early exit if dangling/unassigned is wavied
        case (co: NonEmptyAlignment, _: EmptyAlignment) if co.isWaived => ()
        case (_: EmptyAlignment, po: NonEmptyAlignment) if po.isWaived => ()

        // Base Case 2: early exit if operator requires matching orientations, but they don't align
        case (co: NonEmptyAlignment, po: NonEmptyAlignment) if (!co.alignsWith(po)) && (op.noWrongOrientations) =>
          errors += (s"inversely oriented fields ${co.member} and ${po.member}")

        // Base Case 3: early exit if operator requires matching widths, but they aren't the same
        case (co: NonEmptyAlignment, po: NonEmptyAlignment) if (co.truncates(po, op)) && (op.noMismatchedWidths) =>
          errors += (s"mismatched widths of ${co.member} and ${po.member}")

        // Base Case 3: operator error on dangling/unassigned fields
        case (c: NonEmptyAlignment, _: EmptyAlignment) => errors += (s"${c.errorWord(op)} consumer field ${co.member}")
        case (_: EmptyAlignment, p: NonEmptyAlignment) => errors += (s"${p.errorWord(op)} producer field ${po.member}")

        // Recursive Case 4: non-empty orientations
        case (co: NonEmptyAlignment, po: NonEmptyAlignment) =>
          (co.member, po.member) match {
            case (c: Aggregate, p: Aggregate) =>
              matchingZipOfChildren(Some(co), Some(po)).foreach {
                case (ceo, peo) => doAssignment(ceo.getOrElse(co.empty), peo.getOrElse(po.empty))
              }
            case (c: Aggregate, DontCare) =>
              c.getElements.foreach {
                case f => doAssignment(deriveChildAlignment(f, co), deriveChildAlignment(f, co).swap(DontCare))
              }
            case (DontCare, p: Aggregate) =>
              p.getElements.foreach {
                case f => doAssignment(deriveChildAlignment(f, po).swap(DontCare), deriveChildAlignment(f, po))
              }
            case (c, p) =>
              val o = (co.alignsWith(po), (!co.alignsWith(po) && op.assignToConsumer && !op.assignToProducer), (!co.alignsWith(po) && !op.assignToConsumer && op.assignToProducer)) match {
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

    // Start recursive assignment
    doAssignment(Alignment(consumer, true), Alignment(producer, false))

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

  private def assignAnalog(a: Analog, b: Data)(implicit sourceInfo: SourceInfo): Unit = {
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
