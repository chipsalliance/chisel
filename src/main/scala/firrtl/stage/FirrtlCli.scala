// See LICENSE for license details.

package firrtl.stage

import firrtl.options.Shell

/** [[firrtl.options.Shell Shell]] mixin that provides command line options for FIRRTL. This does not include any
  * [[firrtl.options.RegisteredLibrary RegisteredLibrary]] or [[firrtl.options.RegisteredTransform RegisteredTransform]]
  * as those are automatically loaded by the [[firrtl.options.Stage Stage]] using this [[firrtl.options.Shell Shell]].
  */
trait FirrtlCli { this: Shell =>
  parser.note("FIRRTL Compiler Options")
  Seq( FirrtlFileAnnotation,
       OutputFileAnnotation,
       InfoModeAnnotation,
       FirrtlSourceAnnotation,
       CompilerAnnotation,
       RunFirrtlTransformAnnotation,
       firrtl.EmitCircuitAnnotation,
       firrtl.EmitAllModulesAnnotation )
    .map(_.addOptions(parser))

  phases.DriverCompatibility.TopNameAnnotation.addOptions(parser)
  phases.DriverCompatibility.EmitOneFilePerModuleAnnotation.addOptions(parser)
}
