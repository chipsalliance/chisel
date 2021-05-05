// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.util.Try
import scala.language.experimental.macros

import chisel3.experimental.BaseModule
import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.UnlocatableSourceInfo

/** Abstract base class for Modules that contain Chisel RTL.
  * This abstract base class is a user-defined module which does not include implicit clock and reset and supports
  * multiple IO() declarations.
  */
abstract class RawModule(implicit moduleCompileOptions: CompileOptions)
    extends BaseModule {
  //
  // RTL construction internals
  //
  private val _commands = ArrayBuffer[Command]()
  private[chisel3] def addCommand(c: Command) {
    require(!_closed, "Can't write to module after module close")
    _commands += c
  }
  protected def getCommands = {
    require(_closed, "Can't get commands before module close")
    _commands.toSeq
  }

  //
  // Other Internal Functions
  //
  // For debuggers/testers, TODO: refactor out into proper public API
  private var _firrtlPorts: Option[Seq[firrtl.Port]] = None
  lazy val getPorts = _firrtlPorts.get

  val compileOptions = moduleCompileOptions

  private[chisel3] def namePorts(names: HashMap[HasId, String]): Unit = {
    for (port <- getModulePorts) {
      port.computeName(None, None).orElse(names.get(port)) match {
        case Some(name) =>
          if (_namespace.contains(name)) {
            Builder.error(s"""Unable to name port $port to "$name" in $this,""" +
              " name is already taken by another port!")
          }
          port.setRef(ModuleIO(this, _namespace.name(name)))
        case None =>
          Builder.error(s"Unable to name port $port in $this, " +
            "try making it a public field of the Module")
          port.setRef(ModuleIO(this, "<UNNAMED>"))
      }
    }
  }


  private[chisel3] override def generateComponent(): Component = {
    require(!_closed, "Can't generate module more than once")
    _closed = true

    val names = nameIds(classOf[RawModule])

    // Ports get first naming priority, since they are part of a Module's IO spec
    namePorts(names)

    // Then everything else gets named
    for ((node, name) <- names) {
      node.suggestName(name)
    }

    // All suggestions are in, force names to every node.
    for (id <- getIds) {
      id match {
        case id: BaseModule => id.forceName(None, default=id.desiredName, _namespace)
        case id: MemBase[_] => id.forceName(None, default="MEM", _namespace)
        case id: Data  =>
          if (id.isSynthesizable) {
            id.topBinding match {
              case OpBinding(_, _) =>
                id.forceName(Some(""), default="T", _namespace)
              case MemoryPortBinding(_, _) =>
                id.forceName(None, default="MPORT", _namespace)
              case PortBinding(_) =>
                id.forceName(None, default="PORT", _namespace)
              case RegBinding(_, _) =>
                id.forceName(None, default="REG", _namespace)
              case WireBinding(_, _) =>
                id.forceName(Some(""), default="WIRE", _namespace)
              case _ =>  // don't name literals
            }
          } // else, don't name unbound types
      }
      id._onModuleClose
    }

    closeUnboundIds(names)

    val firrtlPorts = getModulePorts map { port: Data =>
      // Special case Vec to make FIRRTL emit the direction of its
      // element.
      // Just taking the Vec's specifiedDirection is a bug in cases like
      // Vec(Flipped()), since the Vec's specifiedDirection is
      // Unspecified.
      val direction = port match {
        case v: Vec[_] => v.specifiedDirection match {
          case SpecifiedDirection.Input => SpecifiedDirection.Input
          case SpecifiedDirection.Output => SpecifiedDirection.Output
          case SpecifiedDirection.Flip => SpecifiedDirection.flip(v.sample_element.specifiedDirection)
          case SpecifiedDirection.Unspecified => v.sample_element.specifiedDirection
        }
        case _ => port.specifiedDirection
      }

      Port(port, direction)
    }
    _firrtlPorts = Some(firrtlPorts)

    // Generate IO invalidation commands to initialize outputs as unused,
    //  unless the client wants explicit control over their generation.
    val invalidateCommands = {
      if (!compileOptions.explicitInvalidate) {
        getModulePorts map { port => DefInvalid(UnlocatableSourceInfo, port.ref) }
      } else {
        Seq()
      }
    }
    val component = DefModule(this, name, firrtlPorts, invalidateCommands ++ getCommands)
    _component = Some(component)
    component
  }

  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo

    if (!parentCompileOptions.explicitInvalidate) {
      for (port <- getModulePorts) {
        pushCommand(DefInvalid(sourceInfo, port.ref))
      }
    }
  }
}

trait RequireAsyncReset extends Module {
  override private[chisel3] def mkReset: AsyncReset = AsyncReset()
}

trait RequireSyncReset extends Module {
  override private[chisel3] def mkReset: Bool = Bool()
}

package object internal {

  // Private reflective version of "val io" to maintain Chisel.Module semantics without having
  // io as a virtual method. See https://github.com/freechipsproject/chisel3/pull/1550 for more
  // information about the removal of "val io"
  private def reflectivelyFindValIO(self: BaseModule): Record = {
    // Java reflection is faster and works for the common case
    def tryJavaReflect: Option[Record] = Try {
      self.getClass.getMethod("io").invoke(self).asInstanceOf[Record]
    }.toOption
    // Anonymous subclasses don't work with Java reflection, so try slower, Scala reflection
    def tryScalaReflect: Option[Record] = {
      val ru = scala.reflect.runtime.universe
      import ru.{Try => _, _}
      val m = ru.runtimeMirror(self.getClass.getClassLoader)
      val im = m.reflect(self)
      val tpe = im.symbol.toType
      // For some reason, in anonymous subclasses, looking up the Term by name (TermName("io"))
      // hits an internal exception. Searching for the term seems to work though so we use that.
      val ioTerm: Option[TermSymbol] = tpe.decls.collectFirst {
        case d if d.name.toString == "io" && d.isTerm => d.asTerm
      }
      ioTerm.flatMap { term =>
        Try {
          im.reflectField(term).get.asInstanceOf[Record]
        }.toOption
      }
    }

    tryJavaReflect
      .orElse(tryScalaReflect)
      .map(_.autoSeed("io"))
      .orElse {
        // Fallback if reflection fails, user can wrap in IO(...)
        self.findPort("io")
          .collect { case r: Record => r }
      }.getOrElse(throwException(
        s"Compatibility mode Module '$this' must have a 'val io' Bundle. " +
        "If there is such a field and you still see this error, autowrapping has failed (sorry!). " +
        "Please wrap the Bundle declaration in IO(...)."
      ))
  }

  /** Legacy Module class that restricts IOs to just io, clock, and reset, and provides a constructor
    * for threading through explicit clock and reset.
    *
    * '''Do not use this class in user code'''. Use whichever `Module` is imported by your wildcard
    * import (preferably `import chisel3._`).
    */
  abstract class LegacyModule(implicit moduleCompileOptions: CompileOptions) extends Module {
    // Provide a non-deprecated constructor
    def this(override_clock: Option[Clock] = None, override_reset: Option[Bool]=None)
      (implicit moduleCompileOptions: CompileOptions) = {
      this()
      this.override_clock = override_clock
      this.override_reset = override_reset
    }
    def this(_clock: Clock)(implicit moduleCompileOptions: CompileOptions) =
      this(Option(_clock), None)(moduleCompileOptions)
    def this(_reset: Bool)(implicit moduleCompileOptions: CompileOptions)  =
      this(None, Option(_reset))(moduleCompileOptions)
    def this(_clock: Clock, _reset: Bool)(implicit moduleCompileOptions: CompileOptions) =
      this(Option(_clock), Option(_reset))(moduleCompileOptions)

    private lazy val _io: Record = reflectivelyFindValIO(this)

    // Allow access to bindings from the compatibility package
    protected def _compatIoPortBound() = portsContains(_io)

    private[chisel3] override def generateComponent(): Component = {
      _compatAutoWrapPorts()  // pre-IO(...) compatibility hack

      // Restrict IO to just io, clock, and reset
      require(_io != null, "Module must have io")
      require(portsContains(_io), "Module must have io wrapped in IO(...)")
      require((portsContains(clock)) && (portsContains(reset)), "Internal error, module did not have clock or reset as IO")
      require(portsSize == 3, "Module must only have io, clock, and reset as IO")

      super.generateComponent()
    }

    override def _compatAutoWrapPorts(): Unit = {
      if (!_compatIoPortBound() && _io != null) {
        _bindIoInPlace(_io)
      }
    }
  }

  import chisel3.experimental.Param

  /** Legacy BlackBox class will reflectively autowrap val io
    *
    * '''Do not use this class in user code'''. Use whichever `BlackBox` is imported by your wildcard
    * import (preferably `import chisel3._`).
    */
  abstract class LegacyBlackBox(params: Map[String, Param] = Map.empty[String, Param])
                               (implicit moduleCompileOptions: CompileOptions)
      extends chisel3.BlackBox(params) {

    override private[chisel3] lazy val _io: Record = reflectivelyFindValIO(this)

    // This class auto-wraps the BlackBox with IO(...), allowing legacy code (where IO(...) wasn't
    // required) to build.
    override def _compatAutoWrapPorts(): Unit = {
      if (!_compatIoPortBound()) {
        _bindIoInPlace(_io)
      }
    }
  }
}
