// See LICENSE for license details.

package chisel3.internal

import scala.util.DynamicVariable
import scala.collection.mutable.ArrayBuffer
import chisel3._
import chisel3.experimental._
import chisel3.internal.firrtl._
import chisel3.internal.naming._
import _root_.firrtl.annotations.{CircuitName, ComponentName, IsMember, ModuleName, Named, ReferenceTarget}

import scala.collection.mutable

private[chisel3] class Namespace(keywords: Set[String]) {
  private val names = collection.mutable.HashMap[String, Long]()
  for (keyword <- keywords)
    names(keyword) = 1

  private def rename(n: String): String = {
    val index = names(n)
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
  /** Returns a FIRRTL IsMember that refers to this object in the elaborated hardware graph */
  def toTarget: IsMember
  /** Returns a FIRRTL IsMember that refers to the absolute path to this object in the elaborated hardware graph */
  def toAbsoluteTarget: IsMember
}

private[chisel3] trait HasId extends InstanceId {
  private[chisel3] def _onModuleClose: Unit = {} // scalastyle:ignore method.name
  private[chisel3] val _parent: Option[BaseModule] = Builder.currentModule
  _parent.foreach(_.addId(this))

  private[chisel3] val _id: Long = Builder.idGen.next

  // TODO: remove this, but its removal seems to cause a nasty Scala compiler crash.
  override def hashCode: Int = super.hashCode()
  override def equals(that: Any): Boolean = super.equals(that)

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
  private[chisel3] def getOptionRef: Option[Arg] = _ref

  // Implementation of public methods.
  def instanceName: String = _parent match {
    case Some(p) => p._component match {
      case Some(c) => _ref match {
        case Some(arg) => arg fullName c
        case None => suggested_name.getOrElse("??")
      }
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
      if (c == rootClass) {
        Set()
      } else {
        getValNames(c.getSuperclass) ++ c.getDeclaredFields.map(_.getName)
      }
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

  /** Returns a FIRRTL ReferenceTarget that references this object
    * @note Should not be called until circuit elaboration is complete
    */
  final def toTarget: ReferenceTarget = {
    val name = this.instanceName
    import _root_.firrtl.annotations.{Target, TargetToken}
    Target.toTargetTokens(name).toList match {
      case TargetToken.Ref(r) :: components => ReferenceTarget(this.circuitName, this.parentModName, Nil, r, components)
      case other =>
        throw _root_.firrtl.annotations.Target.NamedException(s"Cannot convert $name into [[ReferenceTarget]]: $other")
    }
  }

  final def toAbsoluteTarget: ReferenceTarget = {
    val localTarget = toTarget
    _parent match {
      case Some(parent) => parent.toAbsoluteTarget.ref(localTarget.ref).copy(component = localTarget.component)
      case None => localTarget
    }
  }
}

// Mutable global state for chisel that can appear outside a Builder context
private[chisel3] class ChiselContext() {
  val idGen = new IdGen

  // Record the Bundle instance, class name, method name, and reverse stack trace position of open Bundles
  val bundleStack: ArrayBuffer[(Bundle, String, String, Int)] = ArrayBuffer()
}

private[chisel3] class DynamicContext() {
  val globalNamespace = Namespace.empty
  val components = ArrayBuffer[Component]()
  val annotations = ArrayBuffer[ChiselAnnotation]()
  var currentModule: Option[BaseModule] = None

  /** Contains a mapping from a elaborated module to their aspect
    * Set by [[ModuleAspect]]
    */
  val aspectModule: mutable.HashMap[BaseModule, BaseModule] = mutable.HashMap.empty[BaseModule, BaseModule]

  // Set by object Module.apply before calling class Module constructor
  // Used to distinguish between no Module() wrapping, multiple wrappings, and rewrapping
  var readyForModuleConstr: Boolean = false
  var whenDepth: Int = 0 // Depth of when nesting
  var currentClock: Option[Clock] = None
  var currentReset: Option[Reset] = None
  val errors = new ErrorLog
  val namingStack = new NamingStack
}

//scalastyle:off number.of.methods
private[chisel3] object Builder {
  // All global mutable state must be referenced via dynamicContextVar!!
  private val dynamicContextVar = new DynamicVariable[Option[DynamicContext]](None)
  private def dynamicContext: DynamicContext = {
    require(dynamicContextVar.value.isDefined, "must be inside Builder context")
    dynamicContextVar.value.get
  }

  // Ensure we have a thread-specific ChiselContext
  private val chiselContext = new ThreadLocal[ChiselContext]{
    override def initialValue: ChiselContext = {
      new ChiselContext
    }
  }

  // Initialize any singleton objects before user code inadvertently inherits them.
  private def initializeSingletons(): Unit = {
    // This used to contain:
    //    val dummy = core.DontCare
    //  but this would occasionally produce hangs due to static initialization deadlock
    //  when Builder initialization collided with chisel3.package initialization of the DontCare value.
    // See:
    //  http://ternarysearch.blogspot.com/2013/07/static-initialization-deadlock.html
    //  https://bugs.openjdk.java.net/browse/JDK-8037567
    //  https://stackoverflow.com/questions/28631656/runnable-thread-state-but-in-object-wait
  }

  def namingStackOption: Option[NamingStack] = dynamicContextVar.value.map(_.namingStack)

  def idGen: IdGen = chiselContext.get.idGen

  def globalNamespace: Namespace = dynamicContext.globalNamespace
  def components: ArrayBuffer[Component] = dynamicContext.components
  def annotations: ArrayBuffer[ChiselAnnotation] = dynamicContext.annotations
  def namingStack: NamingStack = dynamicContext.namingStack

  def currentModule: Option[BaseModule] = dynamicContextVar.value match {
    case Some(dyanmicContext) => dynamicContext.currentModule
    case _ => None
  }
  def currentModule_=(target: Option[BaseModule]): Unit = {
    dynamicContext.currentModule = target
  }
  def aspectModule(module: BaseModule): Option[BaseModule] = dynamicContextVar.value match {
    case Some(dynamicContext) => dynamicContext.aspectModule.get(module)
    case _ => None
  }
  def addAspect(module: BaseModule, aspect: BaseModule): Unit = {
    dynamicContext.aspectModule += ((module, aspect))
  }
  def forcedModule: BaseModule = currentModule match {
    case Some(module) => module
    case None => throwException(
      "Error: Not in a Module. Likely cause: Missed Module() wrap or bare chisel API call."
      // A bare api call is, e.g. calling Wire() from the scala console).
    )
  }
  def referenceUserModule: RawModule = {
    currentModule match {
      case Some(module: RawModule) =>
        aspectModule(module) match {
          case Some(aspect: RawModule) => aspect
          case other => module
        }
      case _ => throwException(
        "Error: Not in a RawModule. Likely cause: Missed Module() wrap, bare chisel API call, or attempting to construct hardware inside a BlackBox." // scalastyle:ignore line.size.limit
        // A bare api call is, e.g. calling Wire() from the scala console).
      )
    }
  }
  def forcedUserModule: RawModule = currentModule match {
    case Some(module: RawModule) => module
    case _ => throwException(
      "Error: Not in a UserModule. Likely cause: Missed Module() wrap, bare chisel API call, or attempting to construct hardware inside a BlackBox." // scalastyle:ignore line.size.limit
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
  def currentClock: Option[Clock] = dynamicContext.currentClock
  def currentClock_=(newClock: Option[Clock]): Unit = {
    dynamicContext.currentClock = newClock
  }

  def currentReset: Option[Reset] = dynamicContext.currentReset
  def currentReset_=(newReset: Option[Reset]): Unit = {
    dynamicContext.currentReset = newReset
  }

  def forcedClock: Clock = currentClock.getOrElse(
    throwException("Error: No implicit clock.")
  )
  def forcedReset: Reset = currentReset.getOrElse(
    throwException("Error: No implicit reset.")
  )

  // TODO(twigg): Ideally, binding checks and new bindings would all occur here
  // However, rest of frontend can't support this yet.
  def pushCommand[T <: Command](c: T): T = {
    forcedUserModule.addCommand(c)
    c
  }
  def pushOp[T <: Data](cmd: DefPrim[T]): T = {
    def trulyPushOp[T <: Data](cmd: DefPrim[T]): T = {
      // Bind each element of the returned Data to being a Op
      cmd.id.bind(OpBinding(forcedUserModule))
      pushCommand(cmd).id
    }

    val memoizableCmd = new MemoizableDefPrim(cmd)
    forcedUserModule.getMemoizedCommand(memoizableCmd) match {
      case Some(memoized: T @unchecked) => memoized
      case _ =>
        val res = trulyPushOp(cmd)
        if (whenDepth == 0)
          forcedUserModule.addMemoizableCommand(memoizableCmd, res)
        res
    }
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
    val pruneLength = chiselContext.get.bundleStack.reverse.prefixLength { case (_, cname, mname, pos) =>
      pos >= stackEltsTop || stackElts(pos).getClassName != cname || stackElts(pos).getMethodName != mname
    }
    chiselContext.get.bundleStack.trimEnd(pruneLength)

    // Return the stack state before adding the most recent bundle
    val lastStack = chiselContext.get.bundleStack.map(_._1).toSeq

    // Append the current Bundle to the stack, if it's on the stack trace
    if (eltStackPos >= 0) {
      val stackElt = stackElts(eltStackPos)
      chiselContext.get.bundleStack.append((elt, eltClassName, stackElt.getMethodName, eltStackPos))
    }
    // Otherwise discard the stack frame, this shouldn't fail noisily

    lastStack
  }

  /** Recursively suggests names to supported "container" classes
    * Arbitrary nestings of supported classes are allowed so long as the
    * innermost element is of type HasId
    * (Note: Map is Iterable[Tuple2[_,_]] and thus excluded)
    */
  def nameRecursively(prefix: String, nameMe: Any, namer: (HasId, String) => Unit): Unit = nameMe match {
    case (id: HasId) => namer(id, prefix)
    case Some(elt) => nameRecursively(prefix, elt, namer)
    case (iter: Iterable[_]) if iter.hasDefiniteSize =>
      for ((elt, i) <- iter.zipWithIndex) {
        nameRecursively(s"${prefix}_${i}", elt, namer)
      }
    case _ => // Do nothing
  }

  def errors: ErrorLog = dynamicContext.errors
  def error(m: => String): Unit = {
    if (dynamicContextVar.value.isDefined) {
      errors.error(m)
    } else {
      throwException(m)
    }
  }
  def warning(m: => String): Unit = if (dynamicContextVar.value.isDefined) errors.warning(m)
  def deprecated(m: => String, location: Option[String] = None): Unit =
    if (dynamicContextVar.value.isDefined) errors.deprecated(m, location)

  /** Record an exception as an error, and throw it.
    *
    * @param m exception message
    */
  @throws(classOf[ChiselException])
  def exception(m: => String): Unit = {
    error(m)
    throwException(m)
  }

  def build[T <: RawModule](f: => T): (Circuit, T) = {
    dynamicContextVar.withValue(Some(new DynamicContext())) {
      errors.info("Elaborating design...")
      val mod = f
      mod.forceName(mod.name, globalNamespace)
      errors.checkpoint()
      errors.info("Done elaborating.")

      (Circuit(components.last.name, components, annotations), mod)
    }
  }
  initializeSingletons()
}

/** Allows public access to the naming stack in Builder / DynamicContext, and handles invocations
  * outside a Builder context.
  * Necessary because naming macros expand in user code and don't have access into private[chisel3]
  * objects.
  */
object DynamicNamingStack {
  def pushContext(): NamingContextInterface = {
    Builder.namingStackOption match {
      case Some(namingStack) => namingStack.pushContext()
      case None => DummyNamer
    }
  }

  def popReturnContext[T <: Any](prefixRef: T, until: NamingContextInterface): T = {
    until match {
      case DummyNamer =>
        require(Builder.namingStackOption.isEmpty,
          "Builder context must remain stable throughout a chiselName-annotated function invocation")
      case context: NamingContext =>
        require(Builder.namingStackOption.isDefined,
          "Builder context must remain stable throughout a chiselName-annotated function invocation")
        Builder.namingStackOption.get.popContext(prefixRef, context)
    }
    prefixRef
  }
  
  def length() : Int = Builder.namingStackOption.get.length
}

/** Casts BigInt to Int, issuing an error when the input isn't representable. */
private[chisel3] object castToInt {
  def apply(x: BigInt, msg: String): Int = {
    val res = x.toInt
    require(x == res, s"$msg $x is too large to be represented as Int")
    res
  }
}
