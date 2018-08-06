// See LICENSE for license details.

package chisel3.util.experimental

import chisel3._
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform, annotate}
import chisel3.internal.{InstanceId, NamedComponent}
import firrtl.transforms.{DontTouchAnnotation, NoDedupAnnotation}
import firrtl.passes.wiring.{WiringTransform, SourceAnnotation, SinkAnnotation}
import firrtl.annotations.{ModuleName, ComponentName}

import scala.concurrent.SyncVar
import chisel3.internal.Namespace

/** An exception related to BoringUtils
  * @param message the exception message
  */
class BoringUtilsException(message: String) extends Exception(message)

/** Utilities for generating synthesizeable cross module references
  * ("boring" through a hierarchy) from one source module to one or more
  * sink modules.
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
  /** A global namespace for boring ids */
  private val namespace: SyncVar[Namespace] = new SyncVar
  namespace.put(Namespace.empty)

  /** Get a new name from the global namespace
    *
    * @param value the name you'd like to get
    * @return the name safe in the global namespace
    */
  private def newName(value: String): String = {
    val ns = namespace.take()
    val valuex = ns.name(value)
    namespace.put(ns)
    valuex
  }

  /** Check if a name exists in the namespace */
  private def checkName(value: String): Boolean = namespace.get.contains(value)

  /** Add a named source cross module reference
    *
    * @param component source circuit component
    * @param name unique identifier for this source
    * @param dedup enable dedupblication of modules
    * @param forceUnique fail if a non-unique name parameter is used
    * @return the mangled name used
    */
  def addSource(component: NamedComponent, name: String, dedup: Boolean = false, uniqueName: Boolean = false): String = {
    val id = if (uniqueName) { newName(name) } else { name }
    val maybeDedup =
      if (dedup) { Seq(new ChiselAnnotation { def toFirrtl = NoDedupAnnotation(component.toNamed.module) }) }
      else       { Seq[ChiselAnnotation]()                                                                  }
    val annotations =
      Seq(new ChiselAnnotation with RunFirrtlTransform {
            def toFirrtl = SourceAnnotation(component.toNamed, id)
            def transformClass = classOf[WiringTransform] },
          new ChiselAnnotation { def toFirrtl = DontTouchAnnotation(component.toNamed) } ) ++ maybeDedup

    annotations.map(annotate(_))
    id
  }

  /** Add a named sink cross module reference. Multiple sinks may map to
    * the same source.
    *
    * @param component sink circuit component
    * @param name unique identifier for this sink that must resolve to
    * a source identifier
    */
  def addSink(component: InstanceId, name: String, dedup: Boolean = false, forceExists: Boolean = false): Unit = {
    if (forceExists && !checkName(name)) {
      throw new BoringUtilsException(s"Sink ID '$name' not found in BoringUtils ID namespace") }
    def moduleName = component.toNamed match {
      case c: ModuleName => c
      case c: ComponentName => c.module
      case _ => throw new ChiselException("Can only add a Module or Component sink", null)
    }
    val maybeDedup =
      if (dedup) { Seq(new ChiselAnnotation { def toFirrtl = NoDedupAnnotation(moduleName) }) }
      else       { Seq[ChiselAnnotation]()                                                    }
    val annotations =
      Seq(new ChiselAnnotation with RunFirrtlTransform {
            def toFirrtl = SinkAnnotation(component.toNamed, name)
            def transformClass = classOf[WiringTransform] })
    annotations.map(annotate(_))
  }

  /** Connect a source to one or more sinks
    *
    * @param source a source component
    * @param sinks one or more sink components
    */
  def bore(source: Data, sinks: Seq[Data]): String = {
    lazy val genName = addSource(source, source.instanceName, true, true)
    sinks.map(addSink(_, genName, true, true))
    genName
  }
}
