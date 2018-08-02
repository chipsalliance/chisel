// See LICENSE for license details.

package chisel3.util.experimental

import chisel3._
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform, annotate}
import chisel3.internal.{InstanceId, NamedComponent}
import firrtl.transforms.{DontTouchAnnotation, NoDedupAnnotation}
import firrtl.passes.wiring.{WiringTransform, SourceAnnotation, SinkAnnotation}
import firrtl.annotations.{ModuleName, ComponentName}

import scala.concurrent.SyncVar

/** Utilities for generating synthesizeable cross module references
  * ("boring" through a hierarchy).
  *
  * @example {{{
  * import chisel3.util.experimental.BoringUtils
  * class ModuleA extends Module {
  *   val a = Reg(Bool())
  *   BoringUtils.addSource(a, "unique_identifier")
  * }
  * class ModuleB extends Module {
  *   val b = Wire(Bool())
  *   BoringUtils.addSink(b, "unique_identifier")
  * }
  * class ModuleC extends Module {
  *   val c = Wire(Bool())
  *   BoringUtils.addSink(c, "unique_identifier")
  * }
  * }}}
  */
object BoringUtils {
  /** A representation of a FIRRTL-like Namespace
    *
    * @param names used names
    * @param indices the last numerical suffix used to mangle a given name
    */
  private case class Namespace(names: Set[String], indices: Map[String, BigInt])

  /** A global, mutable store of the Namespace */
  private val namespace: SyncVar[Namespace] = new SyncVar

  /* Initialize the namespace */
  namespace.put(Namespace(Set.empty, Map.empty.withDefaultValue(0)))

  /** Add a named source cross module reference
    *
    * @param component source circuit component
    * @param name unique identifier for this source
    */
  def addSource(component: NamedComponent, name: String): Unit = {
    Seq(new ChiselAnnotation with RunFirrtlTransform {
          def toFirrtl = SourceAnnotation(component.toNamed, name)
          def transformClass = classOf[WiringTransform] },
        new ChiselAnnotation {
          def toFirrtl = DontTouchAnnotation(component.toNamed) },
        new ChiselAnnotation {
          def toFirrtl = NoDedupAnnotation(component.toNamed.module) })
      .map(annotate(_))
  }

  /** Add a named sink cross module reference. Multiple sinks may map to
    * the same source.
    *
    * @param component sink circuit component
    * @param name unique identifier for this sink that must resolve to
    * a source identifier
    */
  def addSink(component: InstanceId, name: String): Unit = {
    def moduleName = component.toNamed match {
      case c: ModuleName => c
      case c: ComponentName => c.module
      case _ => throw new ChiselException("Can only add a Module or Component sink", null)
    }
    Seq(
      new ChiselAnnotation with RunFirrtlTransform {
        def toFirrtl = SinkAnnotation(component.toNamed, name)
        def transformClass = classOf[WiringTransform] },
      new ChiselAnnotation {
        def toFirrtl = NoDedupAnnotation(moduleName) } )
      .map(annotate(_))
  }

  /** Get a new name from the global namespace
    *
    * @param value the name you'd like to get
    * @return the name safe in the global namespace
    */
  private def newName(value: String): String = {
    val ns = namespace.take()

    var valuex = value
    var idx = ns.indices(value)
    while (ns.names.contains(valuex)) {
      valuex = s"${value}_$idx"
      idx += 1
    }
    val nsx = ns.copy(names = ns.names + valuex, indices = ns.indices ++ Map(value -> idx))

    namespace.put(nsx)
    valuex
  }

  /** Connect a source to one or more sinks
    *
    * @param source a source component
    * @param sinks one or more sink components
    */
  def bore(source: Data, sinks: Seq[Data]): Unit = {
    lazy val genName: String = newName(source.instanceName)
    addSource(source, genName)
    sinks.map(addSink(_, genName))
  }
}
