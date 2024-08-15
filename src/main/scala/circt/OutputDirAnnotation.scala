// SPDX-License-Identifier: Apache-2.0

package circt

import chisel3.experimental.{annotate, BaseModule, ChiselAnnotation}
import firrtl.annotations.{ModuleTarget, SingleTargetAnnotation}
import firrtl.FirrtlUserException

/** Annotation to specify a module's output directory.
  */
case class OutputDirAnnotation(target: ModuleTarget, dirname: String) extends SingleTargetAnnotation[ModuleTarget] {
  override def duplicate(target: ModuleTarget): OutputDirAnnotation =
    OutputDirAnnotation(target, dirname)
}

/** Utilities for specifying the output directory for a public module.
  * @example {{{
  * class Inner extends Module with Public {
  *   val io = IO(new Bundle{})
  * }
  *
  * class Top extends Module {
  *   val inner = outputDir(Module(new Inner), "foo")
  * }
  * }}}
  */
object outputDir {
  def apply[T <: BaseModule](data: T, dirname: String): T = {
    annotate(new ChiselAnnotation {
      def toFirrtl = OutputDirAnnotation(data.toTarget, dirname)
    })
    data
  }
}
