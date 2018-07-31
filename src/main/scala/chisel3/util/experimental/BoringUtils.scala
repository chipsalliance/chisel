// See LICENSE for license details.

package chisel3.util.experimental

import chisel3._
import chisel3.experimental.{BaseModule, ChiselAnnotation, RunFirrtlTransform}
import chisel3.internal.{InstanceId, NamedComponent}
import firrtl.transforms.DontTouchAnnotation
import firrtl.passes.wiring.{WiringTransform, SourceAnnotation, SinkAnnotation}

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
  def addSource(component: NamedComponent, name: String): Unit = {
    Seq(new ChiselAnnotation with RunFirrtlTransform {
          def toFirrtl = SourceAnnotation(component.toNamed, name)
          def transformClass = classOf[WiringTransform] },
        new ChiselAnnotation {
          def toFirrtl = DontTouchAnnotation(component.toNamed) })
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
    val anno = new ChiselAnnotation with RunFirrtlTransform {
      def toFirrtl = SinkAnnotation(component.toNamed, name)
      def transformClass = classOf[WiringTransform]
    }
    annotate(anno)
  }
}
