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

private[chisel3] trait HasId {
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
}

private[chisel3] class DynamicContext(moduleCompileOptions: Option[ExplicitCompileOptions] = None) {
  val idGen = new IdGen
  val globalNamespace = new Namespace(None, Set())
  val components = ArrayBuffer[Component]()
  var currentModule: Option[Module] = None
  val errors = new ErrorLog
  val compileOptions = moduleCompileOptions match {
    case Some(options: ExplicitCompileOptions) => options
    case None => chisel3.NotStrict.NotStrictCompileOptions
  }
}

private[chisel3] object Builder {
  // All global mutable state must be referenced via dynamicContextVar!!
  private val dynamicContextVar = new DynamicVariable[Option[DynamicContext]](None)
  private def dynamicContext: DynamicContext =
    dynamicContextVar.value.getOrElse(new DynamicContext)

  def idGen: IdGen = dynamicContext.idGen
  def globalNamespace: Namespace = dynamicContext.globalNamespace
  def components: ArrayBuffer[Component] = dynamicContext.components
  def compileOptions = dynamicContext.compileOptions

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

  // TODO(twigg): Ideally, binding checks and new bindings would all occur here
  // However, rest of frontend can't support this yet.
  def pushCommand[T <: Command](c: T): T = {
    forcedModule._commands += c
    c
  }
  def pushOp[T <: Data](cmd: DefPrim[T]): T = {
    // Bind each element of the returned Data to being a Op
    Binding.bind(cmd.id, OpBinder(forcedModule), "Error: During op creation, fresh result")
    pushCommand(cmd).id
  }

  def errors: ErrorLog = dynamicContext.errors
  def error(m: => String): Unit = errors.error(m)

  def build[T <: Module](f: => T, moduleCompileOptions: Option[ExplicitCompileOptions] = None): Circuit = {
    dynamicContextVar.withValue(Some(new DynamicContext(moduleCompileOptions))) {
      errors.info("Elaborating design...")
      val mod = f
      mod.forceName(mod.name, globalNamespace)
      errors.checkpoint()
      errors.info("Done elaborating.")

      Circuit(components.last.name, components)
    }
  }
}
