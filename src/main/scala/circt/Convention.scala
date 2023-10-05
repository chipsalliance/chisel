// SPDX-License-Identifier: Apache-2.0

package circt

import chisel3.experimental.{annotate, BaseModule, ChiselAnnotation}
import firrtl.annotations.{ModuleTarget, SingleTargetAnnotation}
import firrtl.FirrtlUserException

/** Annotation to specify a module's port convention.
  * The port convention defines how the ports of a module are transformed while
  * lowering to verilog.
  */
case class ConventionAnnotation(target: ModuleTarget, convention: String) extends SingleTargetAnnotation[ModuleTarget] {
  override def duplicate(target: ModuleTarget): ConventionAnnotation =
    ConventionAnnotation(target, convention)
}

/** Utilities for annotating modules with a port convention.
  * The port convention defines how the ports of a module are transformed while
  * lowering to verilog.
  *
  * @example {{{
  * class Inner extends Module {
  *   val io = IO(new Bundle{})
  * }
  *
  * class Top extends Module {
  *   val inner = Module(new Inner)
  *   convention.scalarized(inner)
  * }
  * }}}
  */
object convention {
  private def apply[T <: BaseModule](data: T, convention: String): Unit =
    annotate(new ChiselAnnotation {
      def toFirrtl = ConventionAnnotation(data.toTarget, convention)
    })

  /** Annotate a module as having the "scalarized" port convention.
    * With this port convention, the aggregate ports of the module will be
    * fully scalarized.
    */
  def scalarized[T <: BaseModule](data: T): Unit =
    apply(data, "scalarized")
}
