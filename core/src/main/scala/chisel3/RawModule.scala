// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.util.Try
import scala.language.experimental.macros
import scala.annotation.nowarn
import chisel3.experimental.BaseModule
import chisel3.internal._
import chisel3.experimental.hierarchy.{InstanceClone, ModuleClone}
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.UnlocatableSourceInfo
import _root_.firrtl.annotations.{IsModule, ModuleTarget}

/** Abstract base class for Modules that contain Chisel RTL.
  * This abstract base class is a user-defined module which does not include implicit clock and reset and supports
  * multiple IO() declarations.
  */
@nowarn("msg=class Port") // delete when Port becomes private
abstract class RawModule(implicit moduleCompileOptions: CompileOptions) extends BaseModule {
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
  private var _firrtlPorts: Option[Seq[firrtl.Port]] = None

  @deprecated("Use DataMirror.modulePorts instead. this API will be removed in Chisel 3.6", "Chisel 3.5")
  lazy val getPorts: Seq[Port] = _firrtlPorts.get

  val compileOptions = moduleCompileOptions

  private[chisel3] def checkPorts(): Unit = {
    for ((port, _) <- getModulePorts) {
      if (port._computeName(None).isEmpty) {
        Builder.error(
          s"Unable to name port $port in $this, " +
            "try making it a public field of the Module"
        )
      }
    }
  }

  private[chisel3] override def generateComponent(): Option[Component] = {
    require(!_closed, "Can't generate module more than once")
    _closed = true

    // Check to make sure that all ports can be named
    checkPorts()

    // Now that elaboration is complete for this Module, we can finalize names
    for (id <- getIds) {
      id match {
        case id: ModuleClone[_]   => id.setRefAndPortsRef(_namespace) // special handling
        case id: InstanceClone[_] => id.setAsInstanceRef()
        case id: BaseModule       => id.forceName(default = id.desiredName, _namespace)
        case id: MemBase[_]       => id.forceName(default = "MEM", _namespace)
        case id: stop.Stop        => id.forceName(default = "stop", _namespace)
        case id: assert.Assert    => id.forceName(default = "assert", _namespace)
        case id: assume.Assume    => id.forceName(default = "assume", _namespace)
        case id: cover.Cover      => id.forceName(default = "cover", _namespace)
        case id: printf.Printf => id.forceName(default = "printf", _namespace)
        case id: Data =>
          if (id.isSynthesizable) {
            id.topBinding match {
              case OpBinding(_, _) =>
                id.forceName(default = "_T", _namespace)
              case MemoryPortBinding(_, _) =>
                id.forceName(default = "MPORT", _namespace)
              case PortBinding(_) =>
                id.forceName(default = "PORT", _namespace, true, x => ModuleIO(this, x))
              case RegBinding(_, _) =>
                id.forceName(default = "REG", _namespace)
              case WireBinding(_, _) =>
                id.forceName(default = "_WIRE", _namespace)
              case _ => // don't name literals
            }
          } // else, don't name unbound types
      }
    }

    val firrtlPorts = getModulePorts.map {
      case (port, sourceInfo) =>
        // Special case Vec to make FIRRTL emit the direction of its
        // element.
        // Just taking the Vec's specifiedDirection is a bug in cases like
        // Vec(Flipped()), since the Vec's specifiedDirection is
        // Unspecified.
        val direction = port match {
          case v: Vec[_] =>
            v.specifiedDirection match {
              case SpecifiedDirection.Input       => SpecifiedDirection.Input
              case SpecifiedDirection.Output      => SpecifiedDirection.Output
              case SpecifiedDirection.Flip        => SpecifiedDirection.flip(v.sample_element.specifiedDirection)
              case SpecifiedDirection.Unspecified => v.sample_element.specifiedDirection
            }
          case _ => port.specifiedDirection
        }

        Port(port, direction, sourceInfo)
    }
    _firrtlPorts = Some(firrtlPorts)

    // Generate IO invalidation commands to initialize outputs as unused,
    //  unless the client wants explicit control over their generation.
    val invalidateCommands = {
      if (!compileOptions.explicitInvalidate || this.isInstanceOf[ImplicitInvalidate]) {
        getModulePorts.map { case (port, sourceInfo) => DefInvalid(sourceInfo, port.ref) }
      } else {
        Seq()
      }
    }
    val component = DefModule(this, name, firrtlPorts, invalidateCommands ++ getCommands)
    _component = Some(component)
    _component
  }

  private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = {
    implicit val sourceInfo = UnlocatableSourceInfo

    if (!parentCompileOptions.explicitInvalidate || Builder.currentModule.get.isInstanceOf[ImplicitInvalidate]) {
      for ((port, sourceInfo) <- getModulePorts) {
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

/** Mix with a [[RawModule]] to automatically connect DontCare to the module's ports, wires, and children instance IOs. */
trait ImplicitInvalidate { self: RawModule => }

package object internal {

  import scala.annotation.implicitNotFound
  @implicitNotFound("You are trying to access a macro-only API. Please use the @public annotation instead.")
  trait MacroGenerated

  /** Marker trait for modules that are not true modules */
  private[chisel3] trait PseudoModule extends BaseModule

  /* Check if a String name is a temporary name */
  def isTemp(name: String): Boolean = name.nonEmpty && name.head == '_'

  /** Creates a name String from a prefix and a seed
    * @param prefix The prefix associated with the seed (must be in correct order, *not* reversed)
    * @param seed The seed for computing the name (if available)
    */
  def buildName(seed: String, prefix: Prefix): String = {
    // Don't bother copying the String if there's no prefix
    if (prefix.isEmpty) {
      seed
    } else {
      // Using Java's String builder to micro-optimize appending a String excluding 1st character
      // for temporaries
      val builder = new java.lang.StringBuilder()
      // Starting with _ is the indicator of a temporary
      val temp = isTemp(seed)
      // Make sure the final result is also a temporary if this is a temporary
      if (temp) {
        builder.append('_')
      }
      prefix.foreach { p =>
        builder.append(p)
        builder.append('_')
      }
      if (temp) {
        // We've moved the leading _ to the front, drop it here
        builder.append(seed, 1, seed.length)
      } else {
        builder.append(seed)
      }
      builder.toString
    }
  }

  // Private reflective version of "val io" to maintain Chisel.Module semantics without having
  // io as a virtual method. See https://github.com/freechipsproject/chisel3/pull/1550 for more
  // information about the removal of "val io"
  private def reflectivelyFindValIO(self: BaseModule): Option[Record] = {
    // Java reflection is faster and works for the common case
    def tryJavaReflect: Option[Record] = Try {
      self.getClass.getMethod("io").invoke(self).asInstanceOf[Record]
    }.toOption
      .filter(_ != null)
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
          .filter(_ != null)
      }
    }

    tryJavaReflect
      .orElse(tryScalaReflect)
      .map(_.forceFinalName("io"))
      .orElse {
        // Fallback if reflection fails, user can wrap in IO(...)
        self.findPort("io").collect { case r: Record => r }
      }
  }

  /** Legacy Module class that restricts IOs to just io, clock, and reset, and provides a constructor
    * for threading through explicit clock and reset.
    *
    * '''Do not use this class in user code'''. Use whichever `Module` is imported by your wildcard
    * import (preferably `import chisel3._`).
    */
  abstract class LegacyModule(implicit moduleCompileOptions: CompileOptions) extends Module {
    // Provide a non-deprecated constructor
    def this(
      override_clock: Option[Clock] = None,
      override_reset: Option[Bool] = None
    )(
      implicit moduleCompileOptions: CompileOptions
    ) = {
      this()
      this.override_clock = override_clock
      this.override_reset = override_reset
    }
    def this(_clock: Clock)(implicit moduleCompileOptions: CompileOptions) =
      this(Option(_clock), None)(moduleCompileOptions)
    def this(_reset: Bool)(implicit moduleCompileOptions: CompileOptions) =
      this(None, Option(_reset))(moduleCompileOptions)
    def this(_clock: Clock, _reset: Bool)(implicit moduleCompileOptions: CompileOptions) =
      this(Option(_clock), Option(_reset))(moduleCompileOptions)

    // Sort of a DIY lazy val because if the user tries to construct hardware before val io is
    // constructed, _compatAutoWrapPorts will try to access it but it will be null
    // In that case, we basically need to delay setting this var until later
    private var _ioValue: Option[Record] = None
    private def _io: Option[Record] = _ioValue.orElse {
      _ioValue = reflectivelyFindValIO(this)
      _ioValue
    }

    // Allow access to bindings from the compatibility package
    protected def _compatIoPortBound() = _io.exists(portsContains(_))

    private[chisel3] override def generateComponent(): Option[Component] = {
      _compatAutoWrapPorts() // pre-IO(...) compatibility hack

      // Restrict IO to just io, clock, and reset
      if (_io.isEmpty || !_compatIoPortBound) {
        throwException(
          s"Compatibility mode Module '$this' must have a 'val io' Bundle. " +
            "If there is such a field and you still see this error, autowrapping has failed (sorry!). " +
            "Please wrap the Bundle declaration in IO(...)."
        )
      }
      require(
        (portsContains(clock)) && (portsContains(reset)),
        "Internal error, module did not have clock or reset as IO"
      )
      require(portsSize == 3, "Module must only have io, clock, and reset as IO")

      super.generateComponent()
    }

    override def _compatAutoWrapPorts(): Unit = {
      if (!_compatIoPortBound()) {
        _io.foreach(_bindIoInPlace(_)(UnlocatableSourceInfo, moduleCompileOptions))
      }
    }
  }

  import chisel3.experimental.Param

  /** Legacy BlackBox class will reflectively autowrap val io
    *
    * '''Do not use this class in user code'''. Use whichever `BlackBox` is imported by your wildcard
    * import (preferably `import chisel3._`).
    */
  abstract class LegacyBlackBox(
    params: Map[String, Param] = Map.empty[String, Param]
  )(
    implicit moduleCompileOptions: CompileOptions)
      extends chisel3.BlackBox(params) {

    override private[chisel3] lazy val _io: Option[Record] = reflectivelyFindValIO(this)

    // This class auto-wraps the BlackBox with IO(...), allowing legacy code (where IO(...) wasn't
    // required) to build.
    override def _compatAutoWrapPorts(): Unit = {
      if (!_compatIoPortBound()) {
        _io.foreach(_bindIoInPlace(_)(UnlocatableSourceInfo, moduleCompileOptions))
      }
    }
  }

  /** Internal API for [[ViewParent]] */
  sealed private[chisel3] class ViewParentAPI extends RawModule()(ExplicitCompileOptions.Strict) with PseudoModule {
    // We must provide `absoluteTarget` but not `toTarget` because otherwise they would be exactly
    // the same and we'd have no way to distinguish the kind of target when renaming view targets in
    // the Converter
    // Note that this is not overriding .toAbsoluteTarget, that is a final def in BaseModule that delegates
    // to this method
    private[chisel3] val absoluteTarget: IsModule = ModuleTarget(this.circuitName, "_$$AbsoluteView$$_")

    // This module is not instantiable
    override private[chisel3] def generateComponent(): Option[Component] = None
    override private[chisel3] def initializeInParent(parentCompileOptions: CompileOptions): Unit = ()
    // This module is not really part of the circuit
    _parent = None

    // Sigil to mark views, starts with '_' to make it a legal FIRRTL target
    override def desiredName = "_$$View$$_"

    private[chisel3] val fakeComponent: Component = DefModule(this, desiredName, Nil, Nil)
  }

  /** Special internal object representing the parent of all views
    *
    * @note this is a val instead of an object because of the need to wrap in Module(...)
    */
  private[chisel3] val ViewParent =
    Module.do_apply(new ViewParentAPI)(UnlocatableSourceInfo, ExplicitCompileOptions.Strict)
}
