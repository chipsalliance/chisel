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

  lazy val ChirrtlForm: Seq[TransformDependency] = Seq.empty

  lazy val MinimalHighForm: Seq[TransformDependency] = ChirrtlForm ++
    Seq( Dependency(passes.CheckChirrtl),
         Dependency(passes.CInferTypes),
         Dependency(passes.CInferMDir),
         Dependency(passes.RemoveCHIRRTL) )

  lazy val WorkingIR: Seq[TransformDependency] = MinimalHighForm :+ Dependency(passes.ToWorkingIR)

  lazy val Resolved: Seq[TransformDependency] = WorkingIR ++
    Seq( Dependency(passes.CheckHighForm),
         Dependency(passes.ResolveKinds),
         Dependency(passes.InferTypes),
         Dependency(passes.CheckTypes),
         Dependency(passes.Uniquify),
         Dependency(passes.ResolveFlows),
         Dependency(passes.CheckFlows),
         Dependency[passes.InferBinaryPoints],
         Dependency[passes.TrimIntervals],
         Dependency[passes.InferWidths],
         Dependency(passes.CheckWidths),
         Dependency[firrtl.transforms.InferResets] )

  lazy val Deduped: Seq[TransformDependency] = Resolved :+ Dependency[firrtl.transforms.DedupModules]

  lazy val HighForm: Seq[TransformDependency] = ChirrtlForm ++
    MinimalHighForm ++
    WorkingIR ++
    Resolved ++
    Deduped

  lazy val MidForm: Seq[TransformDependency] = HighForm ++
    Seq( Dependency(passes.PullMuxes),
         Dependency(passes.ReplaceAccesses),
         Dependency(passes.ExpandConnects),
         Dependency(passes.RemoveAccesses),
         Dependency[passes.ExpandWhensAndCheck],
         Dependency[passes.RemoveIntervals],
         Dependency(passes.ConvertFixedToSInt),
         Dependency(passes.ZeroWidth) )

  lazy val LowForm: Seq[TransformDependency] = MidForm ++
    Seq( Dependency(passes.LowerTypes),
         Dependency(passes.Legalize),
         Dependency(firrtl.transforms.RemoveReset),
         Dependency[firrtl.transforms.CheckCombLoops],
         Dependency[checks.CheckResets],
         Dependency[firrtl.transforms.RemoveWires] )

  lazy val LowFormMinimumOptimized: Seq[TransformDependency] = LowForm ++
    Seq( Dependency(passes.RemoveValidIf),
         Dependency(passes.memlib.VerilogMemDelays),
         Dependency(passes.SplitExpressions) )

  lazy val LowFormOptimized: Seq[TransformDependency] = LowFormMinimumOptimized ++
    Seq( Dependency[firrtl.transforms.ConstantPropagation],
         Dependency(passes.PadWidths),
         Dependency[firrtl.transforms.CombineCats],
         Dependency(passes.CommonSubexpressionElimination),
         Dependency[firrtl.transforms.DeadCodeElimination] )

  lazy val VerilogMinimumOptimized: Seq[TransformDependency] = LowFormMinimumOptimized ++
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

  lazy val VerilogOptimized: Seq[TransformDependency] = LowFormOptimized ++ VerilogMinimumOptimized

}
