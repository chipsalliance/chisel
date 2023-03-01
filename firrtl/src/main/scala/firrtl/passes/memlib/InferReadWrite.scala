// SPDX-License-Identifier: Apache-2.0

package firrtl.passes.memlib

import firrtl.options.{HasShellOptions, ShellOption}
import firrtl.annotations.NoTargetAnnotation

case object InferReadWriteAnnotation extends NoTargetAnnotation with HasShellOptions {
  val options = Seq(
    new ShellOption[Unit](
      longOption = "infer-rw",
      toAnnotationSeq = (_: Unit) => Seq(InferReadWriteAnnotation),
      helpText = "Enable read/write port inference for memories",
      shortOption = Some("firw")
    )
  )
}
