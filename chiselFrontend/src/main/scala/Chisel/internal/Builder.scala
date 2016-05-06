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

  // Facilities for 'suggesting' a name to this.
  // Post-name hooks called to carry the suggestion to other candidates as needed
  private var suggested_name: Option[String] = None
  private val postname_hooks = scala.collection.mutable.ListBuffer.empty[String=>Unit]
  // Only takes the first suggestion!
  def suggestName(name: =>String): this.type = {
    if(suggested_name.isEmpty) suggested_name = Some(name)
    for(hook <- postname_hooks) { hook(name) }
    this
  }
  private[Chisel] def addPostnameHook(hook: String=>Unit): Unit = postname_hooks += hook

  // Uses a namespace to convert suggestion into a true name
  // Will not do any naming if the reference already assigned.
  // (e.g. tried to suggest a name to part of a Bundle)
  private[Chisel] def forceName(default: =>String, namespace: Namespace): Unit =
    if(_ref.isEmpty) {
      val candidate_name = suggested_name.getOrElse(default)
      val available_name = namespace.name(candidate_name)
      setRef(Ref(available_name))
    }

  private var _ref: Option[Arg] = None
  private[Chisel] def setRef(imm: Arg): Unit = _ref = Some(imm)
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

  def dynamicContext: DynamicContext =
    dynamicContextVar.value getOrElse (new DynamicContext)
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
      mod.forceName(mod.name, globalNamespace)
      errors.checkpoint()
      errors.info("Done elaborating.")

      Circuit(components.last.name, components)
    }
  }
}
