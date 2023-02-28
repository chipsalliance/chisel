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

  val MinimalHighForm: Seq[TransformDependency] = ChirrtlForm ++
    Seq(
      Dependency(passes.CInferTypes),
      Dependency(passes.CInferMDir),
      Dependency(passes.RemoveCHIRRTL),
      Dependency[annotations.transforms.CleanupNamedTargets]
    )

  @deprecated("Use firrtl.stage.forms.MinimalHighForm", "FIRRTL 1.4.2")
  val WorkingIR: Seq[TransformDependency] = MinimalHighForm

  val Checks: Seq[TransformDependency] =
    Seq(
      Dependency(passes.CheckTypes),
      Dependency(passes.CheckWidths)
    )

  val Resolved: Seq[TransformDependency] = MinimalHighForm ++ Checks ++
    Seq(
      Dependency(passes.ResolveKinds),
      Dependency(passes.InferTypes),
      Dependency(passes.ResolveFlows),
      Dependency[passes.InferWidths],
      Dependency[firrtl.transforms.InferResets]
    )

  val Deduped: Seq[TransformDependency] = Resolved ++
    Seq(
      Dependency[firrtl.transforms.DedupModules],
      Dependency[firrtl.transforms.DedupAnnotationsTransform]
    )

  val HighForm: Seq[TransformDependency] = ChirrtlForm ++
    MinimalHighForm ++
    Resolved ++
    Deduped

  val MidForm: Seq[TransformDependency] = HighForm ++
    Seq(
      Dependency(passes.PullMuxes),
      Dependency(passes.ReplaceAccesses),
      Dependency(passes.ExpandConnects),
      Dependency(passes.RemoveAccesses),
      Dependency(passes.ZeroLengthVecs),
      Dependency[passes.ExpandWhensAndCheck],
      Dependency(passes.ZeroWidth),
      Dependency[firrtl.transforms.formal.AssertSubmoduleAssumptions]
    )

  val LowForm: Seq[TransformDependency] = MidForm ++
    Seq(
      Dependency(passes.LowerTypes),
      Dependency(passes.LegalizeConnects),
      Dependency(firrtl.transforms.RemoveReset),
      Dependency[firrtl.transforms.CheckCombLoops],
      Dependency[checks.CheckResets],
      Dependency[firrtl.transforms.RemoveWires]
    )

  val LowFormMinimumOptimized: Seq[TransformDependency] = LowForm ++
    Seq(
      Dependency(passes.RemoveValidIf),
      Dependency(passes.PadWidths),
      Dependency(passes.SplitExpressions)
    )

  val LowFormOptimized: Seq[TransformDependency] = LowFormMinimumOptimized ++
    Seq(
      Dependency[firrtl.transforms.ConstantPropagation],
      Dependency(passes.CommonSubexpressionElimination),
      Dependency[firrtl.transforms.DeadCodeElimination]
    )

  private def VerilogLowerings(optimize: Boolean): Seq[TransformDependency] = {
    Seq(
      Dependency(firrtl.backends.verilog.LegalizeVerilog),
      Dependency(passes.memlib.VerilogMemDelays),
      Dependency[firrtl.transforms.CombineCats]
    ) ++
      (if (optimize) Seq(Dependency[firrtl.transforms.InlineBooleanExpressions]) else Seq()) ++
      Seq(
        Dependency[firrtl.transforms.LegalizeAndReductionsTransform],
        Dependency[firrtl.transforms.FixAddingNegativeLiterals],
        Dependency[firrtl.transforms.ReplaceTruncatingArithmetic],
        Dependency[firrtl.transforms.InlineBitExtractionsTransform],
        Dependency[firrtl.transforms.InlineAcrossCastsTransform],
        Dependency[firrtl.transforms.LegalizeClocksAndAsyncResetsTransform],
        Dependency[firrtl.transforms.FlattenRegUpdate],
        Dependency(passes.VerilogModulusCleanup),
        Dependency[firrtl.transforms.VerilogRename],
        Dependency(passes.VerilogPrep),
        Dependency[firrtl.AddDescriptionNodes]
      )
  }

  val VerilogMinimumOptimized: Seq[TransformDependency] = LowFormMinimumOptimized ++ VerilogLowerings(optimize = false)

  val VerilogOptimized: Seq[TransformDependency] = LowFormOptimized ++ VerilogLowerings(optimize = true)

  val AssertsRemoved: Seq[TransformDependency] =
    Seq(
      Dependency(firrtl.transforms.formal.ConvertAsserts),
      Dependency[firrtl.transforms.formal.RemoveVerificationStatements]
    )

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
