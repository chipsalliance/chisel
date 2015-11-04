// See LICENSE for license details.

package Chisel
import scala.util.DynamicVariable
import scala.collection.mutable.{ArrayBuffer, HashMap}

private class Namespace(parent: Option[Namespace], keywords: Set[String]) {
  private var i = 0L
  private val names = collection.mutable.HashSet[String]()

  private def rename(n: String) = { i += 1; s"${n}_${i}" }

  def contains(elem: String): Boolean = {
    keywords.contains(elem) || names.contains(elem) ||
      parent.map(_ contains elem).getOrElse(false)
  }

  def name(elem: String): String = {
    if (this contains elem) {
      name(rename(elem))
    } else {
      names += elem
      elem
    }
  }

  def child(kws: Set[String]): Namespace = new Namespace(Some(this), kws)
  def child: Namespace = child(Set())
}

private class IdGen {
  private var counter = -1L
  def next: Long = {
    counter += 1
    counter
  }
}

private[Chisel] trait HasId {
  private[Chisel] def _onModuleClose {}
  private[Chisel] val _parent = Builder.dynamicContext.currentModule
  _parent.foreach(_.addId(this))

  private[Chisel] val _refMap = Builder.globalRefMap
  private[Chisel] val _id = Builder.idGen.next
  private[Chisel] def setRef(imm: Arg) = _refMap.setRef(this, imm)
  private[Chisel] def setRef(name: String) = _refMap.setRef(this, name)
  private[Chisel] def setRef(parent: HasId, name: String) = _refMap.setField(parent, this, name)
  private[Chisel] def setRef(parent: HasId, index: Int) = _refMap.setIndex(parent, this, index)
  private[Chisel] def getRef = _refMap(this)
}

class RefMap {
  private val _refmap = new HashMap[Long,Arg]()

  private[Chisel] def setRef(id: HasId, ref: Arg): Unit =
    _refmap(id._id) = ref

  private[Chisel] def setRef(id: HasId, name: String): Unit =
    if (!_refmap.contains(id._id)) setRef(id, Ref(name))

  private[Chisel] def setField(parentid: HasId, id: HasId, name: String): Unit =
    _refmap(id._id) = Slot(Node(parentid), name)

  private[Chisel] def setIndex(parentid: HasId, id: HasId, index: Int): Unit =
    _refmap(id._id) = Index(Node(parentid), index)

  def apply(id: HasId): Arg = _refmap(id._id)
}

private class DynamicContext {
  val idGen = new IdGen
  val globalNamespace = new Namespace(None, Set())
  val globalRefMap = new RefMap
  val components = ArrayBuffer[Component]()
  var currentModule: Option[Module] = None
  val errors = new ErrorLog
}

private object Builder {
  // All global mutable state must be referenced via dynamicContextVar!!
  private val dynamicContextVar = new DynamicVariable[Option[DynamicContext]](None)

  def dynamicContext: DynamicContext = dynamicContextVar.value.get
  def idGen: IdGen = dynamicContext.idGen
  def globalNamespace: Namespace = dynamicContext.globalNamespace
  def globalRefMap: RefMap = dynamicContext.globalRefMap
  def components: ArrayBuffer[Component] = dynamicContext.components

  def pushCommand[T <: Command](c: T): T = {
    dynamicContext.currentModule.foreach(_._commands += c)
    c
  }
  def pushOp[T <: Data](cmd: DefPrim[T]): T = pushCommand(cmd).id

  def errors: ErrorLog = dynamicContext.errors
  def error(m: => String): Unit = errors.error(m)

  def build[T <: Module](f: => T): Circuit = {
    dynamicContextVar.withValue(Some(new DynamicContext)) {
      errors.info("Elaborating design...")
      val mod = f
      mod.setRef(globalNamespace.name(mod.name))
      errors.checkpoint()
      errors.info("Done elaborating.")

      Circuit(components.last.name, components, globalRefMap)
    }
  }
}
