// SPDX-License-Identifier: Apache-2.0

package chisel3.connectable

import chisel3.internal.{BiConnect, Builder}
import chisel3.internal.sourceinfo.UnlocatableSourceInfo
import chisel3.Data
// Datastructure capturing the semantics of each connectable operator
private[chisel3] sealed trait ConnectionOperator {
  val noDangles:              Boolean
  val noUnassigned:           Boolean
  val mustMatch:              Boolean
  val noWrongOrientations:    Boolean
  val noMismatchedWidths:     Boolean
  val assignToConsumer:       Boolean
  val assignToProducer:       Boolean
  val alwaysAssignToConsumer: Boolean
}

private[chisel3] case object ColonLessEq extends ConnectionOperator {
  val noDangles:              Boolean = true
  val noUnassigned:           Boolean = true
  val mustMatch:              Boolean = true
  val noWrongOrientations:    Boolean = true
  val noMismatchedWidths:     Boolean = true
  val assignToConsumer:       Boolean = true
  val assignToProducer:       Boolean = false
  val alwaysAssignToConsumer: Boolean = false
}

private[chisel3] case object ColonGreaterEq extends ConnectionOperator {
  val noDangles:              Boolean = true
  val noUnassigned:           Boolean = true
  val mustMatch:              Boolean = true
  val noWrongOrientations:    Boolean = true
  val noMismatchedWidths:     Boolean = true
  val assignToConsumer:       Boolean = false
  val assignToProducer:       Boolean = true
  val alwaysAssignToConsumer: Boolean = false
}

private[chisel3] case object ColonLessGreaterEq extends ConnectionOperator {
  val noDangles:              Boolean = true
  val noUnassigned:           Boolean = true
  val mustMatch:              Boolean = true
  val noWrongOrientations:    Boolean = true
  val noMismatchedWidths:     Boolean = true
  val assignToConsumer:       Boolean = true
  val assignToProducer:       Boolean = true
  val alwaysAssignToConsumer: Boolean = false
  def canFirrtlConnect(consumer: ConnectableData[Data], producer: ConnectableData[Data]) = {
    val typeEquivalent = try {
      BiConnect.canFirrtlConnectData(
        consumer.base,
        producer.base,
        UnlocatableSourceInfo,
        ConnectionFunctions.chisel5CompileOptions,
        Builder.referenceUserModule
      ) && consumer.base.typeEquivalent(producer.base)
    } catch {
      // For some reason, an error is thrown if its a View; since this is purely an optimization, any actual error would get thrown
      //  when calling DirectionConnectionFunctions.assign. Hence, we can just default to false to take the non-optimized emission path
      case e: Throwable => false
    }
    (typeEquivalent && consumer.notSpecial && producer.notSpecial)
  }
}

private[chisel3] case object ColonHashEq extends ConnectionOperator {
  val noDangles:              Boolean = true
  val noUnassigned:           Boolean = true
  val mustMatch:              Boolean = true
  val noWrongOrientations:    Boolean = false
  val noMismatchedWidths:     Boolean = true
  val assignToConsumer:       Boolean = true
  val assignToProducer:       Boolean = false
  val alwaysAssignToConsumer: Boolean = true
}
