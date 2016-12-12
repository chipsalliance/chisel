// See LICENSE for license details.

package chisel3.internal

import scala.util.DynamicVariable
import scala.collection.mutable.{ArrayBuffer, HashMap}

import chisel3._
import core._
import firrtl._

private[chisel3] class Namespace(parent: Option[Namespace], keywords: Set[String]) {
  private val names = collection.mutable.HashMap[String, Long]()
  for (keyword <- keywords)
    names(keyword) = 1

  private def rename(n: String): String = {
    val index = names.getOrElse(n, 1L)
    val tryName = s"${n}_${index}"
    names(n) = index + 1
    if (this contains tryName) rename(n) else tryName
  }

  private def sanitize(s: String): String = {
    // TODO what character set does FIRRTL truly support? using ANSI C for now
    def legalStart(c: Char) = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'
    def legal(c: Char) = legalStart(c) || (c >= '0' && c <= '9')
    val res = s filter legal
    if (res.isEmpty || !legalStart(res.head)) s"_$res" else res
  }

  def contains(elem: String): Boolean = {
    names.contains(elem) || parent.map(_ contains elem).getOrElse(false)
  }

  def name(elem: String): String = {
    val sanitized = sanitize(elem)
    if (this contains sanitized) {
      name(rename(sanitized))
    } else {
      names(sanitized) = 1
      sanitized
    }
  }

  def child(kws: Set[String]): Namespace = new Namespace(Some(this), kws)
  def child: Namespace = child(Set())
}

private[chisel3] class IdGen {
  private var counter = -1L
  def next: Long = {
    counter += 1
    counter
  }
}

/** Public API to access Node/Signal names.
  * currently, the node's name, the full path name, and references to its parent Module and component.
  * These are only valid once the design has been elaborated, and should not be used during its construction.
  */
trait InstanceId {
  def instanceName: String
  def pathName: String
  def parentPathName: String
  def parentModName: String
}

private[chisel3] trait HasId extends InstanceId {
  private[chisel3] def _onModuleClose: Unit = {} // scalastyle:ignore method.name
  private[chisel3] val _parent: Option[Module] = Builder.currentModule
  _parent.foreach(_.addId(this))

  private[chisel3] val _id: Long = Builder.idGen.next
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
  private[chisel3] def addPostnameHook(hook: String=>Unit): Unit = postname_hooks += hook

  // Uses a namespace to convert suggestion into a true name
  // Will not do any naming if the reference already assigned.
  // (e.g. tried to suggest a name to part of a Bundle)
  private[chisel3] def forceName(default: =>String, namespace: Namespace): Unit =
    if(_ref.isEmpty) {
      val candidate_name = suggested_name.getOrElse(default)
      val available_name = namespace.name(candidate_name)
      setRef(Ref(available_name))
    }

  private var _ref: Option[Arg] = None
  private[chisel3] def setRef(imm: Arg): Unit = _ref = Some(imm)
  private[chisel3] def setRef(parent: HasId, name: String): Unit = setRef(Slot(Node(parent), name))
  private[chisel3] def setRef(parent: HasId, index: Int): Unit = setRef(Index(Node(parent), ILit(index)))
  private[chisel3] def setRef(parent: HasId, index: UInt): Unit = setRef(Index(Node(parent), index.ref))
  private[chisel3] def getRef: Arg = _ref.get

  // Implementation of public methods.
  def instanceName: String = _parent match {
    case Some(p) => p._component match {
      case Some(c) => getRef fullName c
      case None => throwException("signalName/pathName should be called after circuit elaboration")
    }
    case None => throwException("this cannot happen")
  }
  def pathName: String = _parent match {
    case None => instanceName
    case Some(p) => s"${p.pathName}.$instanceName"
  }
  def parentPathName: String = _parent match {
    case Some(p) => p.pathName
    case None => throwException(s"$instanceName doesn't have a parent")
  }
  def parentModName: String = _parent match {
    case Some(p) => p.name
    case None => throwException(s"$instanceName doesn't have a parent")
  }

  private[chisel3] def getPublicFields(rootClass: Class[_]): Seq[java.lang.reflect.Method] = {
    // Suggest names to nodes using runtime reflection
    def getValNames(c: Class[_]): Set[String] = {
      if (c == rootClass) Set()
      else getValNames(c.getSuperclass) ++ c.getDeclaredFields.map(_.getName)
    }
    val valNames = getValNames(this.getClass)
    def isPublicVal(m: java.lang.reflect.Method) =
      m.getParameterTypes.isEmpty && valNames.contains(m.getName)
    this.getClass.getMethods.sortWith(_.getName < _.getName).filter(isPublicVal(_))
  }
}

private[chisel3] class DynamicContext() {
  val idGen = new IdGen
  val globalNamespace = new Namespace(None, Set())
  val components = ArrayBuffer[Component]()
  val annotations = ArrayBuffer[ChiselAnnotation]()
  var currentModule: Option[Module] = None
  // Set by object Module.apply before calling class Module constructor
  // Used to distinguish between no Module() wrapping, multiple wrappings, and rewrapping
  var readyForModuleConstr: Boolean = false
  val errors = new ErrorLog
}

private[chisel3] object Builder {
  // All global mutable state must be referenced via dynamicContextVar!!
  private val dynamicContextVar = new DynamicVariable[Option[DynamicContext]](None)
  private def dynamicContext: DynamicContext =
    dynamicContextVar.value.getOrElse(new DynamicContext)

  def idGen: IdGen = dynamicContext.idGen
  def globalNamespace: Namespace = dynamicContext.globalNamespace
  def components: ArrayBuffer[Component] = dynamicContext.components
  def annotations: ArrayBuffer[ChiselAnnotation] = dynamicContext.annotations

  def currentModule: Option[Module] = dynamicContext.currentModule
  def currentModule_=(target: Option[Module]): Unit = {
    dynamicContext.currentModule = target
  }
  def forcedModule: Module = currentModule match {
    case Some(module) => module
    case None => throw new Exception(
      "Error: Not in a Module. Likely cause: Missed Module() wrap or bare chisel API call."
      // A bare api call is, e.g. calling Wire() from the scala console).
    )
  }
  def readyForModuleConstr: Boolean = dynamicContext.readyForModuleConstr
  def readyForModuleConstr_=(target: Boolean): Unit = {
    dynamicContext.readyForModuleConstr = target
  }

  // TODO(twigg): Ideally, binding checks and new bindings would all occur here
  // However, rest of frontend can't support this yet.
  def pushCommand[T <: Command](c: T): T = {
    forcedModule match {
      case _: BlackBox => throwException("Cannot add hardware to a BlackBox")
      case m => m._commands += c
    }
    c
  }
  def pushOp[T <: Data](cmd: DefPrim[T]): T = {
    // Bind each element of the returned Data to being a Op
    Binding.bind(cmd.id, OpBinder(forcedModule), "Error: During op creation, fresh result")
    pushCommand(cmd).id
  }

  def errors: ErrorLog = dynamicContext.errors
  def error(m: => String): Unit = errors.error(m)
  def warning(m: => String): Unit = errors.warning(m)
  def deprecated(m: => String): Unit = errors.deprecated(m)

  def build[T <: Module](f: => T): Circuit = {
    dynamicContextVar.withValue(Some(new DynamicContext())) {
      errors.info("Elaborating design...")
      val mod = f
      mod.forceName(mod.name, globalNamespace)
      errors.checkpoint()
      errors.info("Done elaborating.")

      Circuit(components.last.name, components, annotations.map(_.toFirrtl))
    }
  }
}
