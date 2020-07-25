// See LICENSE for license details.

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
    Seq( Dependency(passes.CheckChirrtl),
         Dependency(passes.CInferTypes),
         Dependency(passes.CInferMDir),
         Dependency(passes.RemoveCHIRRTL) )

  val WorkingIR: Seq[TransformDependency] = MinimalHighForm :+ Dependency(passes.ToWorkingIR)

  val Checks: Seq[TransformDependency] =
    Seq( Dependency(passes.CheckHighForm),
         Dependency(passes.CheckTypes),
         Dependency(passes.CheckFlows),
         Dependency(passes.CheckWidths) )

  val Resolved: Seq[TransformDependency] = WorkingIR ++ Checks ++
    Seq( Dependency(passes.ResolveKinds),
         Dependency(passes.InferTypes),
         Dependency(passes.Uniquify),
         Dependency(passes.ResolveFlows),
         Dependency[passes.InferBinaryPoints],
         Dependency[passes.TrimIntervals],
         Dependency[passes.InferWidths],
         Dependency[firrtl.transforms.InferResets] )

  val Deduped: Seq[TransformDependency] = Resolved :+ Dependency[firrtl.transforms.DedupModules]

  val HighForm: Seq[TransformDependency] = ChirrtlForm ++
    MinimalHighForm ++
    WorkingIR ++
    Resolved ++
    Deduped

  val MidForm: Seq[TransformDependency] = HighForm ++
    Seq( Dependency(passes.PullMuxes),
         Dependency(passes.ReplaceAccesses),
         Dependency(passes.ExpandConnects),
         Dependency(passes.RemoveAccesses),
         Dependency(passes.ZeroLengthVecs),
         Dependency[passes.ExpandWhensAndCheck],
         Dependency[passes.RemoveIntervals],
         Dependency(passes.ConvertFixedToSInt),
         Dependency(passes.ZeroWidth),
         Dependency[firrtl.transforms.formal.AssertSubmoduleAssumptions] )

  val LowForm: Seq[TransformDependency] = MidForm ++
    Seq( Dependency(passes.LowerTypes),
         Dependency(passes.Legalize),
         Dependency(firrtl.transforms.RemoveReset),
         Dependency[firrtl.transforms.CheckCombLoops],
         Dependency[checks.CheckResets],
         Dependency[firrtl.transforms.RemoveWires] )

  val LowFormMinimumOptimized: Seq[TransformDependency] = LowForm ++
    Seq( Dependency(passes.RemoveValidIf),
         Dependency(passes.PadWidths),
         Dependency(passes.memlib.VerilogMemDelays),
         Dependency(passes.SplitExpressions),
         Dependency[firrtl.transforms.LegalizeAndReductionsTransform] )

  val LowFormOptimized: Seq[TransformDependency] = LowFormMinimumOptimized ++
    Seq( Dependency[firrtl.transforms.ConstantPropagation],
         Dependency[firrtl.transforms.CombineCats],
         Dependency(passes.CommonSubexpressionElimination),
         Dependency[firrtl.transforms.DeadCodeElimination] )

  val VerilogMinimumOptimized: Seq[TransformDependency] = LowFormMinimumOptimized ++
    Seq( Dependency[firrtl.transforms.BlackBoxSourceHelper],
         Dependency[firrtl.transforms.FixAddingNegativeLiterals],
         Dependency[firrtl.transforms.ReplaceTruncatingArithmetic],
         Dependency[firrtl.transforms.InlineBitExtractionsTransform],
         Dependency[firrtl.transforms.InlineCastsTransform],
         Dependency[firrtl.transforms.LegalizeClocksTransform],
         Dependency[firrtl.transforms.FlattenRegUpdate],
         Dependency(passes.VerilogModulusCleanup),
         Dependency[firrtl.transforms.VerilogRename],
         Dependency(passes.VerilogPrep),
         Dependency[firrtl.AddDescriptionNodes] )

  val VerilogOptimized: Seq[TransformDependency] = LowFormOptimized ++ VerilogMinimumOptimized

  val AssertsRemoved: Seq[TransformDependency] =
    Seq( Dependency(firrtl.transforms.formal.ConvertAsserts),
         Dependency[firrtl.transforms.formal.RemoveVerificationStatements] )

  val BackendEmitters =
    Seq( Dependency[VerilogEmitter],
         Dependency[MinimumVerilogEmitter],
         Dependency[SystemVerilogEmitter] )

  val LowEmitters = Dependency[LowFirrtlEmitter] +: BackendEmitters

  val MidEmitters = Dependency[MiddleFirrtlEmitter] +: LowEmitters

  val HighEmitters = Dependency[HighFirrtlEmitter] +: MidEmitters

  val ChirrtlEmitters = Dependency[ChirrtlEmitter] +: HighEmitters

}
