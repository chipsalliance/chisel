// SPDX-License-Identifier: Apache-2.0

package chisel3.stage.phases

import firrtl.{AnnotationSeq, ExecutionOptionsManager, HasFirrtlOptions}
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{Dependency, OptionsException, OutputAnnotationFileAnnotation, Phase, Unserializable}
import firrtl.stage.{FirrtlCircuitAnnotation, RunFirrtlTransformAnnotation}
import firrtl.stage.phases.DriverCompatibility.TopNameAnnotation

import chisel3.HasChiselExecutionOptions
import chisel3.stage.{ChiselStage, NoRunFirrtlCompilerAnnotation, ChiselOutputFileAnnotation}

/** This provides components of a compatibility wrapper around Chisel's deprecated [[chisel3.Driver]].
  *
  * Primarily, this object includes [[firrtl.options.Phase Phase]]s that generate [[firrtl.annotations.Annotation]]s
  * derived from the deprecated [[firrtl.stage.phases.DriverCompatibility.TopNameAnnotation]].
  */
@deprecated("This object contains no public members. This will be removed in Chisel 3.6.", "Chisel 3.5")
object DriverCompatibility
