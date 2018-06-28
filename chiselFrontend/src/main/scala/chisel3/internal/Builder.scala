// See LICENSE for license details.

package chisel3.internal

import scala.util.DynamicVariable
import scala.collection.mutable.{ArrayBuffer, HashMap}

import chisel3._
import core._
import firrtl._
import _root_.firrtl.annotations.{CircuitName, ComponentName, ModuleName, Named}

private[chisel3] class Namespace(keywords: Set[String]) {
  private val names = collection.mutable.HashMap[String, Long]()
  for (keyword <- keywords)
    names(keyword) = 1

  private def rename(n: String): String = {
    val index = names.getOrElse(n, 1L)
    val tryName = s"${n}_${index}"
    names(n) = index + 1
    if (this contains tryName) rename(n) else tryName
  }

  private def sanitize(s: String, leadingDigitOk: Boolean = false): String = {
    // TODO what character set does FIRRTL truly support? using ANSI C for now
    def legalStart(c: Char) = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'
    def legal(c: Char) = legalStart(c) || (c >= '0' && c <= '9')
    val res = s filter legal
    val headOk = (!res.isEmpty) && (leadingDigitOk || legalStart(res.head))
    if (headOk) res else s"_$res"
  }

  def contains(elem: String): Boolean = names.contains(elem)

  // leadingDigitOk is for use in fields of Records
  def name(elem: String, leadingDigitOk: Boolean = false): String = {
    val sanitized = sanitize(elem, leadingDigitOk)
    if (this contains sanitized) {
      name(rename(sanitized))
    } else {
      names(sanitized) = 1
      sanitized
    }
  }
}

private[chisel3] object Namespace {
  /** Constructs an empty Namespace */
  def empty: Namespace = new Namespace(Set.empty[String])
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
  /** Returns a FIRRTL Named that refers to this object in the elaborated hardware graph */
  def toNamed: Named

}

private[chisel3] trait HasId extends InstanceId {
  private[chisel3] def _onModuleClose: Unit = {} // scalastyle:ignore method.name
  private[chisel3] val _parent: Option[BaseModule] = Builder.currentModule
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
  private[chisel3] def suggestedName: Option[String] = suggested_name
  private[chisel3] def addPostnameHook(hook: String=>Unit): Unit = postname_hooks += hook

  // Uses a namespace to convert suggestion into a true name
  // Will not do any naming if the reference already assigned.
  // (e.g. tried to suggest a name to part of a Record)
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
  // TODO Should this be public?
  protected def circuitName: String = _parent match {
    case None => instanceName
    case Some(p) => p.circuitName
  }

  private[chisel3] def getPublicFields(rootClass: Class[_]): Seq[java.lang.reflect.Method] = {
    // Suggest names to nodes using runtime reflection
    def getValNames(c: Class[_]): Set[String] = {
      if (c == rootClass) Set()
      else getValNames(c.getSuperclass) ++ c.getDeclaredFields.map(_.getName)
    }
    val valNames = getValNames(this.getClass)
    def isPublicVal(m: java.lang.reflect.Method) =
      m.getParameterTypes.isEmpty && valNames.contains(m.getName) && !m.getDeclaringClass.isAssignableFrom(rootClass)
    this.getClass.getMethods.sortWith(_.getName < _.getName).filter(isPublicVal(_))
  }
}
/** Holds the implementation of toNamed for Data and MemBase */
private[chisel3] trait NamedComponent extends HasId {
  /** Returns a FIRRTL ComponentName that references this object
    * @note Should not be called until circuit elaboration is complete
    */
  final def toNamed: ComponentName =
    ComponentName(this.instanceName, ModuleName(this.parentModName, CircuitName(this.circuitName)))
}

private[chisel3] class DynamicContext() {
  val idGen = new IdGen
  val globalNamespace = Namespace.empty
  val components = ArrayBuffer[Component]()
  val annotations = ArrayBuffer[ChiselAnnotation]()
  var currentModule: Option[BaseModule] = None
  // Set by object Module.apply before calling class Module constructor
  // Used to distinguish between no Module() wrapping, multiple wrappings, and rewrapping
  var readyForModuleConstr: Boolean = false
  var whenDepth: Int = 0 // Depth of when nesting
  var currentClockAndReset: Option[ClockAndReset] = None
  val errors = new ErrorLog
  val namingStack = new internal.naming.NamingStack
  // Record the Bundle instance, class name, method name, and reverse stack trace position of open Bundles
  val bundleStack: ArrayBuffer[(Bundle, String, String, Int)] = ArrayBuffer()
}

private[chisel3] object Builder {
  // All global mutable state must be referenced via dynamicContextVar!!
  private val dynamicContextVar = new DynamicVariable[Option[DynamicContext]](None)
  private def dynamicContext: DynamicContext =
    dynamicContextVar.value.getOrElse(new DynamicContext)

  // Initialize any singleton objects before user code inadvertently inherits them.
  private def initializeSingletons(): Unit = {
    val dummy = core.DontCare
  }
  def idGen: IdGen = dynamicContext.idGen
  def globalNamespace: Namespace = dynamicContext.globalNamespace
  def components: ArrayBuffer[Component] = dynamicContext.components
  def annotations: ArrayBuffer[ChiselAnnotation] = dynamicContext.annotations
  def namingStack: internal.naming.NamingStack = dynamicContext.namingStack

  def currentModule: Option[BaseModule] = dynamicContext.currentModule
  def currentModule_=(target: Option[BaseModule]): Unit = {
    dynamicContext.currentModule = target
  }
  def forcedModule: BaseModule = currentModule match {
    case Some(module) => module
    case None => throwException(
      "Error: Not in a Module. Likely cause: Missed Module() wrap or bare chisel API call."
      // A bare api call is, e.g. calling Wire() from the scala console).
    )
  }
  def forcedUserModule: UserModule = currentModule match {
    case Some(module: UserModule) => module
    case _ => throwException(
      "Error: Not in a UserModule. Likely cause: Missed Module() wrap, bare chisel API call, or attempting to construct hardware inside a BlackBox."
      // A bare api call is, e.g. calling Wire() from the scala console).
    )
  }
  def readyForModuleConstr: Boolean = dynamicContext.readyForModuleConstr
  def readyForModuleConstr_=(target: Boolean): Unit = {
    dynamicContext.readyForModuleConstr = target
  }
  def whenDepth: Int = dynamicContext.whenDepth
  def whenDepth_=(target: Int): Unit = {
    dynamicContext.whenDepth = target
  }
  def currentClockAndReset: Option[ClockAndReset] = dynamicContext.currentClockAndReset
  def currentClockAndReset_=(target: Option[ClockAndReset]): Unit = {
    dynamicContext.currentClockAndReset = target
  }
  def forcedClockAndReset: ClockAndReset = currentClockAndReset match {
    case Some(clockAndReset) => clockAndReset
    case None => throwException("Error: No implicit clock and reset.")
  }
  def forcedClock: Clock = forcedClockAndReset.clock
  def forcedReset: Reset = forcedClockAndReset.reset

  // TODO(twigg): Ideally, binding checks and new bindings would all occur here
  // However, rest of frontend can't support this yet.
  def pushCommand[T <: Command](c: T): T = {
    forcedUserModule.addCommand(c)
    c
  }
  def pushOp[T <: Data](cmd: DefPrim[T]): T = {
    // Bind each element of the returned Data to being a Op
    cmd.id.bind(OpBinding(forcedUserModule))
    pushCommand(cmd).id
  }

  // Called when Bundle construction begins, used to record a stack of open Bundle constructors to
  // record candidates for Bundle autoclonetype. This is a best-effort guess.
  // Returns the current stack of open Bundles
  // Note: elt will NOT have finished construction, its elements cannot be accessed
  def updateBundleStack(elt: Bundle): Seq[Bundle] = {
    val stackElts = Thread.currentThread().getStackTrace()
        .reverse  // so stack frame numbers are deterministic across calls
        .dropRight(2)  // discard Thread.getStackTrace and updateBundleStack

    // Determine where we are in the Bundle stack
    val eltClassName = elt.getClass.getName
    val eltStackPos = stackElts.map(_.getClassName).lastIndexOf(eltClassName)

    // Prune the existing Bundle stack of closed Bundles
    // If we know where we are in the stack, discard frames above that
    val stackEltsTop = if (eltStackPos >= 0) eltStackPos else stackElts.size
    val pruneLength = dynamicContext.bundleStack.reverse.prefixLength { case (_, cname, mname, pos) =>
      pos >= stackEltsTop || stackElts(pos).getClassName != cname || stackElts(pos).getMethodName != mname
    }
    dynamicContext.bundleStack.trimEnd(pruneLength)

    // Return the stack state before adding the most recent bundle
    val lastStack = dynamicContext.bundleStack.map(_._1).toSeq

    // Append the current Bundle to the stack, if it's on the stack trace
    if (eltStackPos >= 0) {
      val stackElt = stackElts(eltStackPos)
      dynamicContext.bundleStack.append((elt, eltClassName, stackElt.getMethodName, eltStackPos))
    }
    // Otherwise discard the stack frame, this shouldn't fail noisily

    lastStack
  }

  def errors: ErrorLog = dynamicContext.errors
  def error(m: => String): Unit = errors.error(m)
  def warning(m: => String): Unit = errors.warning(m)
  def deprecated(m: => String, location: Option[String] = None): Unit = errors.deprecated(m, location)

  /** Record an exception as an error, and throw it.
    *
    * @param m exception message
    */
  @throws(classOf[ChiselException])
  def exception(m: => String): Unit = {
    error(m)
    throwException(m)
  }

  def build[T <: UserModule](f: => T): Circuit = {
    dynamicContextVar.withValue(Some(new DynamicContext())) {
      errors.info("Elaborating design...")
      val mod = f
      mod.forceName(mod.name, globalNamespace)
      errors.checkpoint()
      errors.info("Done elaborating.")

      Circuit(components.last.name, components, annotations)
    }
  }
  initializeSingletons()
}

/** Allows public access to the naming stack in Builder / DynamicContext.
  * Necessary because naming macros expand in user code and don't have access into private[chisel3]
  * objects.
  */
object DynamicNamingStack {
  def apply() = Builder.namingStack
}
