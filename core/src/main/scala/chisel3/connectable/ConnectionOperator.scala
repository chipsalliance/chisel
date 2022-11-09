// SPDX-License-Identifier: Apache-2.0

package chisel3.connectable

// Datastructure capturing the semantics of each connectable operator
private[chisel3] sealed trait ConnectionOperator {
  val noDangles: Boolean
  val noUnassigned: Boolean
  val mustMatch: Boolean
  val noWrongOrientations: Boolean
  val assignToConsumer: Boolean
  val assignToProducer: Boolean
  val alwaysAssignToConsumer: Boolean
}

private[chisel3] case object ColonLessEq extends ConnectionOperator {
  val noDangles: Boolean = true
  val noUnassigned: Boolean = true
  val mustMatch: Boolean = true
  val noWrongOrientations: Boolean = true
  val assignToConsumer: Boolean = true
  val assignToProducer: Boolean = false
  val alwaysAssignToConsumer: Boolean = false
}

private[chisel3] case object ColonGreaterEq extends ConnectionOperator {
  val noDangles: Boolean = true
  val noUnassigned: Boolean = true
  val mustMatch: Boolean = true
  val noWrongOrientations: Boolean = true
  val assignToConsumer: Boolean = false
  val assignToProducer: Boolean = true
  val alwaysAssignToConsumer: Boolean = false
}

private[chisel3] case object ColonLessGreaterEq extends ConnectionOperator {
  val noDangles: Boolean = true
  val noUnassigned: Boolean = true
  val mustMatch: Boolean = true
  val noWrongOrientations: Boolean = true
  val assignToConsumer: Boolean = true
  val assignToProducer: Boolean = true
  val alwaysAssignToConsumer: Boolean = false
}

private[chisel3] case object ColonHashEq extends ConnectionOperator {
  val noDangles: Boolean = true
  val noUnassigned: Boolean = true
  val mustMatch: Boolean = true
  val noWrongOrientations: Boolean = false
  val assignToConsumer: Boolean = true
  val assignToProducer: Boolean = false
  val alwaysAssignToConsumer: Boolean = true
}
