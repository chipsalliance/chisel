// SPDX-License-Identifier: Apache-2.0

package firrtl.stage

import firrtl._
import firrtl.options.Dependency
import firrtl.stage.TransformManager.TransformDependency

/*
 * - InferWidths should have InferTypes split out
 * - ConvertFixedToSInt should have InferTypes split out
 * - Move InferTypes out of ZeroWidth
 */

object Forms {

  val ChirrtlForm: Seq[TransformDependency] = Seq.empty

  val MinimalHighForm: Seq[TransformDependency] = Seq.empty

  @deprecated("Use firrtl.stage.forms.MinimalHighForm", "FIRRTL 1.4.2")
  val WorkingIR: Seq[TransformDependency] = MinimalHighForm

  val Checks: Seq[TransformDependency] = Seq.empty

  val Resolved: Seq[TransformDependency] = MinimalHighForm ++ Checks

  val Deduped: Seq[TransformDependency] = Resolved

  val HighForm: Seq[TransformDependency] = ChirrtlForm ++
    MinimalHighForm ++
    Resolved ++
    Deduped

  val MidForm: Seq[TransformDependency] = HighForm

  val LowForm: Seq[TransformDependency] = MidForm

  val LowFormMinimumOptimized: Seq[TransformDependency] = LowForm

  val LowFormOptimized: Seq[TransformDependency] = LowFormMinimumOptimized

  val VerilogMinimumOptimized: Seq[TransformDependency] = LowFormMinimumOptimized

  val VerilogOptimized: Seq[TransformDependency] = LowFormOptimized

  val AssertsRemoved: Seq[TransformDependency] = Seq.empty

  val BackendEmitters =
    Seq(
      Dependency[VerilogEmitter],
      Dependency[MinimumVerilogEmitter],
      Dependency[SystemVerilogEmitter]
    )

  val LowEmitters = Dependency[LowFirrtlEmitter] +: BackendEmitters

  val MidEmitters = Dependency[MiddleFirrtlEmitter] +: LowEmitters

  val HighEmitters = Dependency[HighFirrtlEmitter] +: MidEmitters

  val ChirrtlEmitters = Dependency[ChirrtlEmitter] +: HighEmitters

}
