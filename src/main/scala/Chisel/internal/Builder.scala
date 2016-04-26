// See LICENSE for license details.

package Chisel.internal

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

private[Chisel] class IdGen {
  private var counter = -1L
  def next: Long = {
    counter += 1
    counter
  }
}

private[Chisel] trait HasId {
  private[Chisel] def _onModuleClose {} // scalastyle:ignore method.name
  private[Chisel] val _parent = Builder.dynamicContext.currentModule
  _parent.foreach(_.addId(this))

  private[Chisel] val _id = Builder.idGen.next
  override def hashCode: Int = _id.toInt
  override def equals(that: Any): Boolean = that match {
    case x: HasId => _id == x._id
    case _ => false
  }

  private var _ref: Option[Arg] = None
  private[Chisel] def setRef(imm: Arg): Unit = _ref = Some(imm)
  private[Chisel] def setRef(name: => String): Unit = if (_ref.isEmpty) setRef(Ref(name))
  private[Chisel] def setRef(parent: HasId, name: String): Unit = setRef(Slot(Node(parent), name))
  private[Chisel] def setRef(parent: HasId, index: Int): Unit = setRef(Index(Node(parent), ILit(index)))
  private[Chisel] def setRef(parent: HasId, index: UInt): Unit = setRef(Index(Node(parent), index.ref))
  private[Chisel] def getRef: Arg = _ref.get
}

private[Chisel] class DynamicContext {
  val idGen = new IdGen
  val globalNamespace = new Namespace(None, Set())
  val components = ArrayBuffer[Component]()
  var currentModule: Option[Module] = None
  val errors = new ErrorLog
}

private[Chisel] object Builder {
  // All global mutable state must be referenced via dynamicContextVar!!
  private val dynamicContextVar = new DynamicVariable[Option[DynamicContext]](None)

  def dynamicContext: DynamicContext = dynamicContextVar.value.get
  def idGen: IdGen = dynamicContext.idGen
  def globalNamespace: Namespace = dynamicContext.globalNamespace
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

      Circuit(components.last.name, components)
    }
  }
}
