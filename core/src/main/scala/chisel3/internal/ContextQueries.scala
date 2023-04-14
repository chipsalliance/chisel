package chisel3.internal
import chisel3.experimental.hierarchy.{Definition, Instance}
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.{IsMember, CircuitTarget, CompleteTarget, ModuleTarget, IsModule}

object ContextQuery {
  def isProto(c: Context): Boolean      = c.getValue.map(_.isInstanceOf[BaseModule]).getOrElse(false)
  def isDefinition(c: Context): Boolean = c.getValue.map(_.isInstanceOf[Definition[_]]).getOrElse(false)
  def isInstance(c: Context): Boolean   = c.getValue.map(_.isInstanceOf[Instance[_]]).getOrElse(false)
  def isModule(c: Context): Boolean     = isProto(c) || isDefinition(c) || isInstance(c)
  def isCircuit(c: Context): Boolean    = c.getValue.map(_.isInstanceOf[DynamicContext]).getOrElse(false)

  // Could be an instance, definition, or proto
  def localParentModule(c: Context): Option[Context] = {
    c.parentCollectFirst {
      case p if isModule(p) => p
    }
  }
  def localParentProto(c: Context): Option[BaseModule] = localParentModule(c).map(_.origin.value.asInstanceOf[BaseModule])
  def localParentProtoModuleName(c: Context): Option[String] = localParentProto(c).map(_.name)

  def localParentProtoModuleIdentifier(c: Context): Option[String] = localParentProto(c).map(_.definitionIdentifier)

  def identifierTarget(c: Context): Option[CompleteTarget] = {
    if(isCircuit(c)) Some(CircuitTarget(c.valueAs[DynamicContext].main.get.definitionIdentifier))
    else if(isModule(c)) {
      val defnId = c.valueAs[BaseModule].definitionIdentifier
      val instId = c.valueAs[BaseModule].instanceIdentifier
      identifierTarget(c.parent.get) match {
        case Some(c: CircuitTarget) => Some(c.module(defnId))
        case Some(m: IsModule) => Some(m.instOf(instId, defnId))
        case _ => None
      }
    } else None
  }

  def nameTarget(c: Context): Option[CompleteTarget] = {
    if(isCircuit(c)) Some(CircuitTarget(c.valueAs[DynamicContext].main.get.name))
    else if(isModule(c)) {
      val defnName = c.valueAs[BaseModule].name
      val instName = c.valueAs[BaseModule].instanceName
      identifierTarget(c.parent.get) match {
        case Some(c: CircuitTarget) => Some(c.module(defnName))
        case Some(m: IsModule) => Some(m.instOf(instName, defnName))
        case _ => None
      }
    } else None
  }

  def isATopModule(c: Context): Boolean = c.parent.map(isCircuit).getOrElse(false)

}