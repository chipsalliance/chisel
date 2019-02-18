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

/** Utilities for generating synthesizable cross module references that "bore" through the hierarchy. The underlying
  * cross module connects are handled by FIRRTL's Wiring Transform.
  *
  * Consider the following exmple where you want to connect a component in one module to a component in another. Module
  * `Constant` has a wire tied to `42` and `Expect` will assert unless connected to `42`:
  * {{{
  * class Constant extends Module {
  *   val io = IO(new Bundle{})
  *   val x = Wire(UInt(6.W))
  *   x := 42.U
  * }
  * class Expect extends Module {
  *   val io = IO(new Bundle{})
  *   val y = Wire(UInt(6.W))
  *   y := 0.U
  *   // This assertion will fail unless we bore!
  *   chisel3.assert(y === 42.U, "y should be 42 in module Expect")
  * }
  * }}}
  *
  * We can then connect `x` to `y` using [[BoringUtils]] without modifiying the Chisel IO of `Constant`, `Expect`, or
  * modules that may instantiate them. There are two approaches to do this:
  *
  * 1. Hierarchical boring using [[BoringUtils.bore]]
  *
  * 2. Non-hierarchical boring using [[BoringUtils.addSink]]/[[BoringUtils.addSource]]
  *
  * ===Hierarchical Boring===
  *
  * Hierarchcical boring involves connecting one sink instance to another source instance in a parent module. Below,
  * module `Top` contains an instance of `Cosntant` and `Expect`. Using [[BoringUtils.bore]], we can connect
  * `constant.x` to `expect.y`.
  *
  * {{{
  * class Top extends Module {
  *   val io = IO(new Bundle{})
  *   val constant = Module(new Constant)
  *   val expect = Module(new Expect)
  *   BoringUtils.bore(constant.x, Seq(expect.y))
  * }
  * }}}
  *
  * ===Non-hierarchical Boring===
  *
  * Non-hierarchical boring involves connections from sources to sinks that cannot see each other. Here, `x` is
  * described as a source and given a name, `uniqueId`, and `y` is described as a sink with the same name. This is
  * equivalent to the hierarchical boring example above, but requires no modifications to `Top`.
  *
  * {{{
  * class Constant extends Module {
  *   val io = IO(new Bundle{})
  *   val x = Wire(UInt(6.W))
  *   x := 42.U
  *   BoringUtils.addSource(x, "uniqueId")
  * }
  * class Expect extends Module {
  *   val io = IO(new Bundle{})
  *   val y = Wire(UInt(6.W))
  *   y := 0.U
  *   // This assertion will fail unless we bore!
  *   chisel3.assert(y === 42.U, "y should be 42 in module Expect")
  *   BoringUtils.addSink(y, "uniqueId")
  * }
  * class Top extends Module {
  *   val io = IO(new Bundle{})
  *   val constant = Module(new Constant)
  *   val expect = Module(new Expect)
  * }
  * }}}
  *
  * ==Comments==
  *
  * Both hierarchical and non-hierarchical boring emit FIRRTL annotations that describe sources and sinks. These are
  * matched by a `name` key that indicates they should be wired together. Hierarhical boring safely generates this name
  * automatically. Non-hierarchical boring unsafely relies on user input to generate this name. Use of non-hierarchical
  * naming may result in naming conflicts that the user must handle.
  *
  * The automatic generation of hierarchical names relies on a global, mutable namespace. This is currently persistent
  * across circuit elaborations.
  */
object BoringUtils {
  /* A global namespace for boring ids */
  private val namespace: SyncVar[Namespace] = new SyncVar
  namespace.put(Namespace.empty)

  /* Get a new name (value) from the namespace */
  private def newName(value: String): String = {
    val ns = namespace.take()
    val valuex = ns.name(value)
    namespace.put(ns)
    valuex
  }

  /* True if the requested name (value) exists in the namespace */
  private def checkName(value: String): Boolean = namespace.get.contains(value)

  /** Add a named source cross module reference
    * @param component source circuit component
    * @param name unique identifier for this source
    * @param disableDedup disable dedupblication of this source component (this should be true if you are trying to wire
    * from specific identical sources differently)
    * @param uniqueName if true, this will use a non-conflicting name from the global namespace
    * @return the name used
    * @note if a uniqueName is not specified, the returned name may differ from the user-provided name
    */
  def addSource(
    component: NamedComponent,
    name: String,
    disableDedup: Boolean = false,
    uniqueName: Boolean = false): String = {

    val id = if (uniqueName) { newName(name) } else { name }
    val maybeDedup =
      if (disableDedup) { Seq(new ChiselAnnotation { def toFirrtl = NoDedupAnnotation(component.toNamed.module) }) }
      else              { Seq[ChiselAnnotation]()                                                                  }
    val annotations =
      Seq(new ChiselAnnotation with RunFirrtlTransform {
            def toFirrtl = SourceAnnotation(component.toNamed, id)
            def transformClass = classOf[WiringTransform] },
          new ChiselAnnotation { def toFirrtl = DontTouchAnnotation(component.toNamed) } ) ++ maybeDedup

    annotations.map(annotate(_))
    id
  }

  /** Add a named sink cross module reference. Multiple sinks may map to the same source.
    * @param component sink circuit component
    * @param name unique identifier for this sink that must resolve to
    * @param disableDedup disable deduplication of this sink component (this should be true if you are trying to wire
    * specific, identical sinks differently)
    * @param forceExists if true, require that the provided `name` paramater already exists in the global namespace
    * @throws BoringUtilsException if name is expected to exist and itdoesn't
    */
  def addSink(
    component: InstanceId,
    name: String,
    disableDedup: Boolean = false,
    forceExists: Boolean = false): Unit = {

    if (forceExists && !checkName(name)) {
      throw new BoringUtilsException(s"Sink ID '$name' not found in BoringUtils ID namespace") }
    def moduleName = component.toNamed match {
      case c: ModuleName => c
      case c: ComponentName => c.module
      case _ => throw new ChiselException("Can only add a Module or Component sink", null)
    }
    val maybeDedup =
      if (disableDedup) { Seq(new ChiselAnnotation { def toFirrtl = NoDedupAnnotation(moduleName) }) }
      else              { Seq[ChiselAnnotation]()                                                    }
    val annotations =
      Seq(new ChiselAnnotation with RunFirrtlTransform {
            def toFirrtl = SinkAnnotation(component.toNamed, name)
            def transformClass = classOf[WiringTransform] }) ++ maybeDedup
    annotations.map(annotate(_))
  }

  /** Connect a source to one or more sinks
    * @param source a source component
    * @param sinks one or more sink components
    * @return the name of the signal used to connect the source to the
    * sinks
    * @note the returned name will be based on the name of the source
    * component
    */
  def bore(source: Data, sinks: Seq[Data]): String = {
    lazy val genName = addSource(source, source.instanceName, true, true)
    sinks.map(addSink(_, genName, true, true))
    genName
  }
}
