package chisel3.experimental.hierarchy
import chisel3.experimental.hierarchy.core._
import chisel3.internal.firrtl._
import chisel3.internal.{PseudoModule, Namespace, HasId, throwException}
import chisel3.experimental.BaseModule
import chisel3._
import Utils._

/** Represents an instance contained in a specific proto context, e.g. a child module
  *
  * Private internal class to serve as a _parent for Data in cloned ports
  *
  * @param proto
  * @param contexts
  */
private[chisel3] final class StandInModule[T <: BaseModule](val proto: T, val parent: Option[IsHierarchicable]) extends PseudoModule with IsStandIn[T] {
  _parent = parent match {
    case Some(b: BaseModule) => Some(b)
    case other => None
  }

  def toInstance:   core.Instance[T] = new core.Instance(StandIn(this))
  def toDefinition: core.Definition[T] = new core.Definition(StandIn(StandInDefinition(proto, getCircuit)))

  // ======== THINGS TO MAKE CHISEL WORK ========

  override def toString = s"experimental.hierarchy.StandInModule(${proto})"
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
