// SPDX-License-Identifier: Apache-2.0

package firrtl.stage

import firrtl.options.Shell
import firrtl.passes.CommonSubexpressionElimination
import firrtl.transforms.{CustomRadixTransform, NoCircuitDedupAnnotation}

/** [[firrtl.options.Shell Shell]] mixin that provides command line options for FIRRTL. This does not include any
  * [[firrtl.options.RegisteredLibrary RegisteredLibrary]] or [[firrtl.options.RegisteredTransform RegisteredTransform]]
  * as those are automatically loaded by the [[firrtl.options.Stage Stage]] using this [[firrtl.options.Shell Shell]].
  */
trait FirrtlCli { this: Shell =>
  parser.note("FIRRTL Compiler Options")
  Seq(
    FirrtlFileAnnotation,
    FirrtlDirectoryAnnotation,
    OutputFileAnnotation,
    InfoModeAnnotation,
    FirrtlSourceAnnotation,
    RunFirrtlTransformAnnotation,
    firrtl.EmitCircuitAnnotation,
    firrtl.EmitAllModulesAnnotation,
    NoCircuitDedupAnnotation,
    WarnNoScalaVersionDeprecation,
    PrettyNoExprInlining,
    DisableFold,
    OptimizeForFPGA,
    CurrentFirrtlStateAnnotation,
    CommonSubexpressionElimination,
    AllowUnrecognizedAnnotations,
    CustomRadixTransform
  )
    .map(_.addOptions(parser))

  phases.DriverCompatibility.TopNameAnnotation.addOptions(parser)
  phases.DriverCompatibility.EmitOneFilePerModuleAnnotation.addOptions(parser)
}
