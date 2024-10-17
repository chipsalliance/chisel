// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.util.Try
import chisel3.experimental.{BaseModule, OpaqueType, SourceInfo, UnlocatableSourceInfo}
import chisel3.internal._
import chisel3.internal.binding._
import chisel3.experimental.hierarchy.{InstanceClone, ModuleClone}
import chisel3.properties.{DynamicObject, Property, StaticObject}
import chisel3.internal.Builder._
import chisel3.internal.firrtl.ir._
import chisel3.reflect.DataMirror
import chisel3.Module.ResetType._
import _root_.firrtl.annotations.{IsModule, ModuleTarget}
import scala.collection.immutable.VectorBuilder
import scala.collection.mutable.ArrayBuffer

/** Abstract base class for Modules that contain Chisel RTL.
  * This abstract base class is a user-defined module which does not include implicit clock and reset and supports
  * multiple IO() declarations.
  */
abstract class RawModule extends BaseModule {

  /** Hook to invoke hardware generators after the rest of the Module is constructed.
    *
    * This is a power-user API, and should not normally be needed.
    *
    * In rare cases, it is necessary to run hardware generators at a late stage, but still within the scope of the
    * Module. In these situations, atModuleBodyEnd may be used to register such generators. For example:
    *
    *  {{{
    *    class Example extends RawModule {
    *      atModuleBodyEnd {
    *        val extraPort0 = IO(Output(Bool()))
    *        extraPort0 := 0.B
    *      }
    *    }
    *  }}}
    *
    * Any generators registered with atModuleBodyEnd are the last code to execute when the Module is constructed. The
    * execution order is:
    *
    *   - The constructors of any super classes or traits the Module extends
    *   - The constructor of the Module itself
    *   - The atModuleBodyEnd generators
    *
    * The atModuleBodyEnd generators execute in the lexical order they appear in the Module constructor.
    *
    * For example:
    *
    *  {{{
    *    trait Parent {
    *      // Executes first.
    *      val foo = ...
    *    }
    *
    *    class Example extends Parent {
    *      // Executes second.
    *      val bar = ...
    *
    *      atModuleBodyEnd {
    *        // Executes fourth.
    *        val qux = ...
    *      }
    *
    *      atModuleBodyEnd {
    *        // Executes fifth.
    *        val quux = ...
    *      }
    *
    *      // Executes third..
    *      val baz = ...
    *    }
    *  }}}
    *
    * If atModuleBodyEnd is used in a Definition, any generated hardware will be included in the Definition. However, it
    * is currently not possible to annotate any val within atModuleBodyEnd as @public.
    */
  protected def atModuleBodyEnd(gen: => Unit): Unit = {
    _atModuleBodyEnd += { () => gen }
  }
  private val _atModuleBodyEnd = new ArrayBuffer[() => Unit]

  /** Hook to invoke hardware generators after a Module has been constructed and closed.
    *
    * This is useful for running hardware generators after a Module's constructor has run and its Definition is available, while still having access to arguments and definitions in the constructor. The Module itself can no longer be modified at this point.
    *
    * An interesting application of this is the generation of unit tests whenever a module is instantiated. For example:
    *
    * {{{
    *   class Example(N: int) extends RawModule {
    *     private val someSecret: Int = ...
    *
    *     afterModuleBuilt {
    *       // Executes once the surrounding module is closed.
    *       // We can get its definition at this point and pass it to another module.
    *       Definition(ExampleTest(this.definition, someSecret))
    *     }
    *   }
    *
    *   class ExampleTest(unitDef: Definition[Example], someSecret: Int) extends RawModule {
    *     // Instantiate the generated module and test it.
    *     val unit = Instance(unitDef)
    *     ...
    *   }
    *
    *   class Parent extends RawModule {
    *     Instantiate(Example(42))
    *   }
    *
    *   // Resulting modules:
    *   // - Parent (top-level)
    *   //   - instantiates Example
    *   // - ExampleTest (top-level)
    *   //   - instantiates Example
    *   // - Example
    * }}}
    */
  protected def afterModuleBuilt(gen: => Unit): Unit = {
    _afterModuleBuilt += { () => gen }
  }
  private val _afterModuleBuilt = new ArrayBuffer[() => Unit]

  //
  // RTL construction internals
  //
  protected override def hasBody: Boolean = true

  private[chisel3] def withRegion[A](newRegion: Block)(thunk: => A): A = {
    Builder.pushBlock(newRegion)
    val result = thunk
    require(Builder.currentBlock == Some(newRegion), "didn't return to same region")
    Builder.popBlock()
    result
  }

  private[chisel3] def addCommand(c: Command): Unit = {
    require(!_closed, "Can't write to module after module close")
    require(Builder.currentBlock.isDefined, "must have block set")
    Builder.currentBlock.get.addCommand(c)
  }
  protected def getCommands: Seq[Command] = {
    require(_closed, "Can't get commands before module close")
    _body.getCommands()
  }

  //
  // Other Internal Functions
  //
  private var _firrtlPorts: Option[Seq[Port]] = None

  private[chisel3] def checkPorts(): Unit = {
    for ((port, source) <- getModulePortsAndLocators) {
      if (port._computeName(None).isEmpty) {
        Builder.error(
          s"Unable to name port $port in $this, " +
            s"try making it a public field of the Module ${source.makeMessage()}"
        )(UnlocatableSourceInfo)
      }
    }
  }

  /** Private variable that tracks if a module is public. */
  private[chisel3] def _isPublic: Boolean = false

  /** Finalize name for an id created during this RawModule's constructor.
    *
    * @param id The id to finalize.
    */
  private def nameId(id: HasId) = id match {
    case id: ModuleClone[_]   => id.setRefAndPortsRef(_namespace) // special handling
    case id: InstanceClone[_] => id.setAsInstanceRef()
    case id: BaseModule       => id.forceName(default = id.desiredName, _namespace)
    case id: MemBase[_]       => id.forceName(default = "MEM", _namespace)
    case id: stop.Stop        => id.forceName(default = "stop", _namespace)
    case id: assert.Assert    => id.forceName(default = "assert", _namespace)
    case id: assume.Assume    => id.forceName(default = "assume", _namespace)
    case id: cover.Cover      => id.forceName(default = "cover", _namespace)
    case id: printf.Printf => id.forceName(default = "printf", _namespace)
    case id: DynamicObject => {
      // Force name of the DynamicObject, and set its Property[ClassType] type's ref to the DynamicObject.
      // The type's ref can't be set upon instantiation, because the DynamicObject hasn't been named yet.
      // This also updates the source Class ref to the DynamicObject ref now that it's named.
      id.forceName(default = "_object", _namespace)
      id.getReference.setRef(id.getRef)
      id.setSourceClassRef()
    }
    case id: StaticObject => {
      // Set the StaticObject's ref and Property[ClassType] type's ref to the BaseModule for the Class.
      // These refs can't be set upon instantiation, because the ModuleClone hasn't been named yet.
      id.setRef(id.getInstanceModule.getRef)
      id.getReference.setRef(id.getInstanceModule.getRef)
    }
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
          // probes have their refs set eagerly
          case _ => // don't name literals
        }
      }
    case m: SramTarget =>
      id.forceName(default = "MEM", _namespace)
  }

  private[chisel3] override def generateComponent(): Option[Component] = {
    require(!_closed, "Can't generate module more than once")

    // Now that elaboration is complete for this Module, we can finalize names that have been generated thus far.
    val numInitialIds = _ids.size
    for (id <- _ids) {
      nameId(id)
    }

    // Evaluate any atModuleBodyEnd generators.
    _atModuleBodyEnd.foreach { gen =>
      gen()
    }

    _closed = true

    // Check to make sure that all ports can be named
    checkPorts()

    // Take a second pass through any ids generated during atModuleBodyEnd blocks to finalize names for them.
    for (id <- _ids.view.drop(numInitialIds)) {
      nameId(id)
    }

    val firrtlPorts = getModulePortsAndLocators.map {
      case (port, sourceInfo) =>
        Port(port, port.specifiedDirection, sourceInfo)
    }
    _firrtlPorts = Some(firrtlPorts)

    // No more commands.
    _body.close()

    // Generate IO invalidation commands to initialize outputs as unused,
    //  unless the client wants explicit control over their generation.
    val component =
      DefModule(this, name, _isPublic, Builder.enabledLayers.toSeq, firrtlPorts, _body)

    // Secret connections can be staged if user bored into children modules
    _component = Some(component)
    _component
  }

  private[chisel3] def secretConnection(left: Data, _right: Data)(implicit si: SourceInfo): Unit = {
    val (right: Data, _) = chisel3.experimental.dataview
      .reifyIdentityView(_right)
      .getOrElse(
        throwException(s"BoringUtils currently only support identity views, ${_right} has multiple targets.")
      )

    def computeConnection(left: Data, right: Data): Command = {
      (left.probeInfo.nonEmpty, right.probeInfo.nonEmpty) match {
        case (true, true)                                 => ProbeDefine(si, left.lref, Node(right))
        case (true, false) if left.probeInfo.get.writable => ProbeDefine(si, left.lref, RWProbeExpr(Node(right)))
        case (true, false)                                => ProbeDefine(si, left.lref, ProbeExpr(Node(right)))
        case (false, true)                                => Connect(si, left.lref, ProbeRead(Node(right)))
        case (false, false)                               =>
          // For non-probe, directly create Nodes for lhs, skipping visibility check to support BoringUtils.drive.
          (left, right) match {
            case (lhsOpaque: Record, rhsOpaque: Record)
                if lhsOpaque._isOpaqueType && rhsOpaque._isOpaqueType && DataMirror.isProperty(
                  lhsOpaque.allElements.head
                ) && DataMirror.isProperty(rhsOpaque.allElements.head) =>
              computeConnection(lhsOpaque.allElements.head, rhsOpaque.allElements.head)
            case (_: Property[_], _: Property[_]) => PropAssign(si, Node(left), Node(right))
            // Use `connect lhs, read(probe(rhs))` if lhs is passive version of rhs.
            // This provides solution for this: https://github.com/chipsalliance/chisel/issues/3557
            case (_, _)
                if !DataMirror.checkAlignmentTypeEquivalence(left, right) &&
                  DataMirror.checkAlignmentTypeEquivalence(left, Output(chiselTypeOf(right))) =>
              Connect(si, Node(left), ProbeRead(ProbeExpr(Node(right))))
            case (_, _) => Connect(si, Node(left), Node(right))
          }
      }
    }

    val rhs = computeConnection(left, right)
    Builder.currentBlock.get.addSecretCommand(rhs)
  }

  protected[chisel3] override def moduleBuilt(): Unit = {
    // Evaluate any afterModuleBuilt generators.
    _afterModuleBuilt.foreach { gen => gen() }
  }

  private[chisel3] def initializeInParent(): Unit = {}
}

/** Enforce that the Module.reset be Asynchronous (AsyncReset) */
trait RequireAsyncReset extends Module {
  override final def resetType: Module.ResetType.Asynchronous.type = Module.ResetType.Asynchronous
}

/** Enforce that the Module.reset be Synchronous (Bool) */
trait RequireSyncReset extends Module {
  override final def resetType: Module.ResetType.Synchronous.type = Module.ResetType.Synchronous
}
