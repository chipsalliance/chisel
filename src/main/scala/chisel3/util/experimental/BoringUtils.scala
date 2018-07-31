// See LICENSE for license details.

package chisel3.util.experimental

import chisel3._
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform, annotate}
import chisel3.internal.{InstanceId, NamedComponent}
import firrtl.transforms.DontTouchAnnotation
import firrtl.passes.wiring.{WiringTransform, SourceAnnotation, SinkAnnotation}

/** Utilities for generating synthesizeable cross module references.
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
