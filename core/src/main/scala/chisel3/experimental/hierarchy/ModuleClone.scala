// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy
import chisel3.experimental.hierarchy.core._
import chisel3.internal.firrtl._
import chisel3.internal.{throwException, HasId, Namespace, PseudoModule}
import chisel3.experimental.BaseModule
import chisel3._
import Utils._

/** Represents a unique instance which is DIFFERENT from the underlying proto
  * It has a different instance name and ports
  * We do not mock up its parental lineage; to do that, we use StandInHierarchy
  *
  * Private internal class to serve as a _parent for Data in cloned ports
  *
  * @param proto
  * @param contexts
  */
private[chisel3] final class ModuleClone[T <: BaseModule](
  val genesis: ModuleDefinition[T],
  val lenses:  Seq[Lense[T]])
    extends PseudoModule
    with Clone[T] {
  // _parent is set outside, just like a normal module

  // ======== THINGS TO MAKE CHISEL WORK ========

  override def toString = s"experimental.hierarchy.ModuleClone(${proto})"
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
    _component = proto._component
    None
  }
  // Maps proto ports to module clone's ports
  private[chisel3] lazy val ioMap: Map[Data, Data] = {
    val name2Port = getPorts.elements
    proto.getChiselPorts.map { case (name, data) => data -> name2Port(name) }.toMap
  }
  // This module doesn't actually exist in the FIRRTL so no initialization to do
  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()

  // Name of this instance's module is the same as the proto's name
  override def desiredName: String = proto.name

  private[chisel3] def setRefAndPortsRef(namespace: Namespace): Unit = {
    val record = _portsRecord
    // Use .forceName to re-use default name resolving behavior
    record.forceName(None, default = this.desiredName, namespace)
    // Now take the Ref that forceName set and convert it to the correct Arg
    val instName = record.getRef match {
      case Ref(name) => name
      case bad       => throwException(s"Internal Error! Cloned-module Record $record has unexpected ref $bad")
    }
    // Set both the record and the module to have the same instance name
    record.setRef(ModuleCloneIO(proto, instName), force = true) // force because we did .forceName first
    this.setRef(Ref(instName))
  }
}
