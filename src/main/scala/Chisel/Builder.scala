package Chisel
import scala.util.DynamicVariable
import scala.collection.mutable.{ArrayBuffer, HashMap}

private class Namespace(parent: Option[Namespace], keywords: Option[Set[String]]) {
  private var i = 0L
  private val names = collection.mutable.HashSet[String]()
  def forbidden =  keywords.getOrElse(Set()) ++ names

  private def rename(n: String) = { i += 1; s"${n}_${i}" }

  def contains(elem: String): Boolean = {
    forbidden.contains(elem) ||
      parent.map(_ contains elem).getOrElse(false)
  }

  def name(elem: String): String = {
    val res = if(forbidden contains elem) rename(elem) else elem
    names += res
    res
  }

  def child(ks: Option[Set[String]]): Namespace = new Namespace(Some(this), ks)
  def child: Namespace = new Namespace(Some(this), None)
}

private class FIRRTLNamespace extends Namespace(None, Some(Set("mem", "node", "wire", "reg", "inst")))

private class IdGen {
  private var counter = -1L
  def next: Long = {
    counter += 1
    counter
  }
}

trait HasId {
  private[Chisel] val _id = Builder.idGen.next
  def setRef() =  Builder.globalRefMap.setRef(this, s"T_${_id}")
  def setRef(imm: Immediate) = Builder.globalRefMap.setRef(this, imm)
  def setRef(name: String) = Builder.globalRefMap.setRef(this, name)
  def setRef(parent: HasId, name: String) = Builder.globalRefMap.setField(parent, this, name)
  def setRef(parent: HasId, index: Int) = Builder.globalRefMap.setIndex(parent, this, index)
}

class RefMap {
  private val _refmap = new HashMap[Long,Immediate]()

  def setRef(id: HasId, ref: Immediate): Unit =
    _refmap(id._id) = ref

  def setRef(id: HasId, name: String): Unit =
    if (!_refmap.contains(id._id))
      setRef(id, Ref(Builder.globalNamespace.name(name)))

  def setField(parentid: HasId, id: HasId, name: String): Unit = {
    _refmap(id._id) = Slot(Alias(parentid), name)
  }

  def setIndex(parentid: HasId, id: HasId, index: Int): Unit =
    _refmap(id._id) = Index(Alias(parentid), index)

  def apply(id: HasId): Immediate = _refmap(id._id)
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
      mod.setRef(mod.name)
      Circuit(components.last.name, components, globalRefMap, parameterDump)
    }
  }
}
