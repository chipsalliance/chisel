// See LICENSE for license details.

package Chisel.internal

import java.util.IdentityHashMap
import scala.util.DynamicVariable
import scala.collection.mutable.{ArrayBuffer, HashMap}

import Chisel._
import Chisel.internal.firrtl._

private[Chisel] class Namespace(parent: Option[Namespace], keywords: Set[String]) {
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

private[Chisel] trait HasId {
  private[Chisel] def _onModuleClose {}
  private[Chisel] val _parent = Builder.dynamicContext.currentModule
  _parent.foreach(_.addId(this))

  private[Chisel] val _refMap = Builder.globalRefMap
  private[Chisel] def setRef(imm: Arg) = _refMap.setRef(this, imm)
  private[Chisel] def setRef(name: String) = _refMap.setRef(this, name)
  private[Chisel] def setRef(parent: HasId, name: String) = _refMap.setField(parent, this, name)
  private[Chisel] def setRef(parent: HasId, index: Int) = _refMap.setIndex(parent, this, ILit(index))
  private[Chisel] def setRef(parent: HasId, index: UInt) = _refMap.setIndex(parent, this, index.ref)
  private[Chisel] def getRef = _refMap(this)
}

class RefMap {
  private val _refmap = new IdentityHashMap[HasId,Arg]()

  private[Chisel] def setRef(id: HasId, ref: Arg): Unit =
    _refmap.put(id, ref)

  private[Chisel] def setRef(id: HasId, name: String): Unit =
    if (!_refmap.containsKey(id)) setRef(id, Ref(name))

  private[Chisel] def setField(parentid: HasId, id: HasId, name: String): Unit =
    _refmap.put(id, Slot(Node(parentid), name))

  private[Chisel] def setIndex(parentid: HasId, id: HasId, index: Arg): Unit =
    _refmap.put(id, Index(Node(parentid), index))

  def apply(id: HasId): Arg = {
    val rtn = _refmap.get(id)
    require(rtn != null)
    return rtn
  }
}

private[Chisel] class DynamicContext {
  val globalNamespace = new Namespace(None, Set())
  val globalRefMap = new RefMap
  val components = ArrayBuffer[Component]()
  var currentModule: Option[Module] = None
  val errors = new ErrorLog
}

private[Chisel] object Builder {
  // All global mutable state must be referenced via dynamicContextVar!!
  private val dynamicContextVar = new DynamicVariable[Option[DynamicContext]](None)

  def dynamicContext: DynamicContext = dynamicContextVar.value.get
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
