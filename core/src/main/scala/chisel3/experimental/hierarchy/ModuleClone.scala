// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import chisel3.experimental.BaseModule
import chisel3.internal.{HasId, PseudoModule}
import chisel3.internal.firrtl.{Component, ModuleCloneIO, Ref}
import chisel3.internal.{throwException, Namespace}
import chisel3._

// Private internal class to serve as a _parent for Data in cloned ports
private[chisel3] class ModuleClone[T <: BaseModule](val getProto: T) extends PseudoModule with core.IsClone[T] {
  override def toString = s"ModuleClone(${getProto})"
  // Do not call default addId function, which may modify a module that is already "closed"
  override def addId(d: HasId): Unit = ()
  def getPorts = _portsRecord
  // ClonePorts that hold the bound ports for this module
  // Used for setting the refs of both this module and the Record
  private[chisel3] var _portsRecord: Record = _
  // This is necessary for correctly supporting .toTarget on a Module Clone. If it is made from the
  // Instance/Definition API, it should return an instanceTarget. If made from CMAR, it should return a
  // ModuleTarget.
  private[chisel3] var _madeFromDefinition: Boolean = false
  // Don't generate a component, but point to the one for the cloned Module
  private[chisel3] def generateComponent(): Option[Component] = {
    require(!_closed, "Can't generate module more than once")
    _closed = true
    _component = getProto._component
    None
  }
  // Maps proto ports to module clone's ports
  private[chisel3] lazy val ioMap: Map[Data, Data] = {
    getProto match {
      // BlackBox needs special handling for its pseduo-io Bundle
      case protoBB: BlackBox =>
        Map(protoBB._io.get -> getPorts.elements("io"))
      case _ =>
        val name2Port = getPorts.elements
        getProto.getChiselPorts.map { case (name, data) => data -> name2Port(name) }.toMap
    }
  }
  // This module doesn't actually exist in the FIRRTL so no initialization to do
  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()

  // Name of this instance's module is the same as the proto's name
  override def desiredName: String = getProto.name

  private[chisel3] def setRefAndPortsRef(namespace: Namespace): Unit = {
    val record = _portsRecord
    // Use .forceName to re-use default name resolving behavior
    record.forceName(default = this.desiredName, namespace)
    // Now take the Ref that forceName set and convert it to the correct Arg
    val instName = record.getRef match {
      case Ref(name) => name
      case bad       => throwException(s"Internal Error! Cloned-module Record $record has unexpected ref $bad")
    }
    // Set both the record and the module to have the same instance name
    val ref = ModuleCloneIO(getProto, instName)
    record.setRef(ref, force = true) // force because we did .forceName first
    getProto match {
      // BlackBox needs special handling for its pseduo-io Bundle
      case _: BlackBox =>
        // Override the io Bundle's ref so that it thinks it is the top for purposes of
        // generating FIRRTL
        record.elements("io").setRef(ref, force = true)
      case _ => // Do nothing
    }

    this.setRef(Ref(instName))
  }
}
