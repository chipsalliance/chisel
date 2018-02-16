// See LICENSE for license details.

package chisel3.util

import chisel3._
import chisel3.experimental._
import chisel3.internal.InstanceId
import _root_.firrtl.passes.wiring.WiringTransform

/** Utilities for generating synthesizeable cross module references.
  *
  * @example {{{
  * class ModuleA extends Module with BoringUtils {
  *   val a = Reg(Bool())
  *   addSource(a, "unique_identifier")
  * }
  * class ModuleB extends Module with BoringUtils {
  *   val b = Wire(Bool())
  *   addSink(b, "unique_identifier")
  * }
  * class ModuleC extends Module with BoringUtils {
  *   val c = Wire(Bool())
  *   addSink(c, "unique_identifier")
  * }
  * }}}
  */
trait BoringUtils extends BaseModule {
  self: BaseModule =>

  /** Add a named source cross module reference
    *
    * @param component source circuit component
    * @param name unique identifier for this source
    */
  def addSource(component: InstanceId, name: String): Unit = {
    annotate(
      ChiselAnnotation(component, classOf[WiringTransform], s"source $name"))
  }

  /** Add a named sink cross module reference. Multiple sinks may map to
    * the same source.
    *
    * @param component sink circuit component
    * @param name unique identifier for this sink that must resolve to
    * a source identifier
    */
  def addSink(component: InstanceId, name: String): Unit = {
    annotate(
      ChiselAnnotation(component, classOf[WiringTransform], s"sink $name"))
  }
}
