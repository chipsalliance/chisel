package Chisel
import scala.util.DynamicVariable
import scala.collection.mutable.{ArrayBuffer, HashMap}

private class IdGen {
  private var counter = -1L
  def next: Long = {
    counter += 1
    counter
  }
}

class RefMap {
  private val _refmap = new HashMap[Long,Immediate]()

  def setRef(id: Id, ref: Immediate): Unit =
    _refmap(id._id) = ref

  def setRefForId(id: Id, name: String): Unit =
    if (!_refmap.contains(id._id))
      setRef(id, Ref(Builder.globalNamespace.name(name)))

  def setFieldForId(parentid: Id, id: Id, name: String): Unit = {
    _refmap(id._id) = Slot(Alias(parentid), name)
  }

  def setIndexForId(parentid: Id, id: Id, index: Int): Unit =
    _refmap(id._id) = Index(Alias(parentid), index)

  def apply(id: Id): Immediate = _refmap(id._id)
}

private class DynamicContext {
  val idGen = new IdGen
  val globalNamespace = new FIRRTLNamespace
  val globalRefMap = new RefMap
  val components = ArrayBuffer[Component]()
  val currentModuleVar = new DynamicVariable[Option[Module]](None)
  val currentParamsVar = new DynamicVariable[Parameters](Parameters.empty)
  val parameterDump = new ParameterDump

  def getCurrentModule = currentModuleVar.value
  def moduleScope[T](body: => T): T = {
    currentModuleVar.withValue(getCurrentModule)(body)
  }
  def forceCurrentModule[T](m: Module) { // Used in Module constructor
    currentModuleVar.value = Some(m)
  }
  def pushCommand(c: Command) {
    currentModuleVar.value.foreach(_._commands += c)
  }

  def getParams: Parameters = currentParamsVar.value
  def paramsScope[T](p: Parameters)(body: => T): T = {
    currentParamsVar.withValue(p)(body)
  }
}

private object Builder {
  // All global mutable state must be referenced via dynamicContextVar!!
  private val dynamicContextVar = new DynamicVariable[Option[DynamicContext]](None)

  def dynamicContext = dynamicContextVar.value.get
  def idGen = dynamicContext.idGen
  def globalNamespace = dynamicContext.globalNamespace
  def globalRefMap = dynamicContext.globalRefMap
  def components = dynamicContext.components
  def parameterDump = dynamicContext.parameterDump

  def pushCommand(c: Command) = dynamicContext.pushCommand(c)
  def pushOp[T <: Data](cmd: DefPrim[T]) = {
    pushCommand(cmd)
    cmd.id
  }

  def build[T <: Module](f: => T): Circuit = {
    dynamicContextVar.withValue(Some(new DynamicContext)) {
      val mod = f
      globalRefMap.setRefForId(mod, mod.name)
      Circuit(components.last.name, components, globalRefMap, parameterDump)
    }
  }
}
