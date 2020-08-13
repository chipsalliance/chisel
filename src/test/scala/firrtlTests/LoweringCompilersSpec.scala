// See LICENSE for license details.

package firrtlTests

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should._

import firrtl._
import firrtl.options.Dependency
import firrtl.stage.{Forms, TransformManager}

sealed trait PatchAction { val line: Int }

case class Add(line: Int, transforms: Seq[Dependency[Transform]]) extends PatchAction
case class Del(line: Int) extends PatchAction

object Transforms {
  class IdentityTransformDiff(val inputForm: CircuitForm, val outputForm: CircuitForm) extends Transform {
    override def execute(state: CircuitState): CircuitState = state
    override def name: String = s">>>>> $inputForm -> $outputForm <<<<<"
  }
  import firrtl.{ChirrtlForm => C, HighForm => H, MidForm => M, LowForm => L, UnknownForm => U}
  class ChirrtlToChirrtl extends IdentityTransformDiff(C, C)
  class HighToChirrtl    extends IdentityTransformDiff(H, C)
  class HighToHigh       extends IdentityTransformDiff(H, H)
  class MidToMid         extends IdentityTransformDiff(M, M)
  class MidToChirrtl     extends IdentityTransformDiff(M, C)
  class MidToHigh        extends IdentityTransformDiff(M, H)
  class LowToChirrtl     extends IdentityTransformDiff(L, C)
  class LowToHigh        extends IdentityTransformDiff(L, H)
  class LowToMid         extends IdentityTransformDiff(L, M)
  class LowToLow         extends IdentityTransformDiff(L, L)
}

class LoweringCompilersSpec extends AnyFlatSpec with Matchers {

  def legacyTransforms(a: CoreTransform): Seq[Transform] = a match {
    case _: ChirrtlToHighFirrtl => Seq(
      firrtl.passes.CheckChirrtl,
      firrtl.passes.CInferTypes,
      firrtl.passes.CInferMDir,
      firrtl.passes.RemoveCHIRRTL)
    case _: IRToWorkingIR => Seq(firrtl.passes.ToWorkingIR)
    case _: ResolveAndCheck => Seq(
      firrtl.passes.CheckHighForm,
      firrtl.passes.ResolveKinds,
      firrtl.passes.InferTypes,
      firrtl.passes.CheckTypes,
      firrtl.passes.Uniquify,
      firrtl.passes.ResolveKinds,
      firrtl.passes.InferTypes,
      firrtl.passes.ResolveFlows,
      firrtl.passes.CheckFlows,
      new firrtl.passes.InferBinaryPoints,
      new firrtl.passes.TrimIntervals,
      new firrtl.passes.InferWidths,
      firrtl.passes.CheckWidths,
      new firrtl.transforms.InferResets)
    case _: HighFirrtlToMiddleFirrtl => Seq(
      firrtl.passes.PullMuxes,
      firrtl.passes.ReplaceAccesses,
      firrtl.passes.ExpandConnects,
      firrtl.passes.ZeroLengthVecs,
      firrtl.passes.RemoveAccesses,
      firrtl.passes.Uniquify,
      firrtl.passes.ExpandWhens,
      firrtl.passes.CheckInitialization,
      firrtl.passes.ResolveKinds,
      firrtl.passes.InferTypes,
      firrtl.passes.CheckTypes,
      firrtl.passes.ResolveFlows,
      new firrtl.passes.InferWidths,
      firrtl.passes.CheckWidths,
      new firrtl.passes.RemoveIntervals,
      firrtl.passes.ConvertFixedToSInt,
      firrtl.passes.ZeroWidth,
      firrtl.passes.InferTypes)
    case _: MiddleFirrtlToLowFirrtl => Seq(
      firrtl.passes.LowerTypes,
      firrtl.passes.ResolveKinds,
      firrtl.passes.InferTypes,
      firrtl.passes.ResolveFlows,
      new firrtl.passes.InferWidths,
      firrtl.passes.Legalize,
      firrtl.transforms.RemoveReset,
      firrtl.passes.ResolveFlows,
      new firrtl.transforms.CheckCombLoops,
      new checks.CheckResets,
      new firrtl.transforms.RemoveWires)
    case _: LowFirrtlOptimization => Seq(
      firrtl.passes.RemoveValidIf,
      new firrtl.transforms.ConstantPropagation,
      firrtl.passes.PadWidths,
      new firrtl.transforms.ConstantPropagation,
      firrtl.passes.Legalize,
      firrtl.passes.memlib.VerilogMemDelays, // TODO move to Verilog emitter
      new firrtl.transforms.ConstantPropagation,
      firrtl.passes.SplitExpressions,
      new firrtl.transforms.CombineCats,
      firrtl.passes.CommonSubexpressionElimination,
      new firrtl.transforms.DeadCodeElimination)
    case _: MinimumLowFirrtlOptimization => Seq(
      firrtl.passes.RemoveValidIf,
      firrtl.passes.PadWidths,
      firrtl.passes.Legalize,
      firrtl.passes.memlib.VerilogMemDelays, // TODO move to Verilog emitter
      firrtl.passes.SplitExpressions)
  }

  def compare(a: Seq[Transform], b: TransformManager, patches: Seq[PatchAction] = Seq.empty): Unit = {
    info(s"""Transform Order:\n${b.prettyPrint("    ")}""")

    val m = new scala.collection.mutable.HashMap[Int, Seq[Dependency[Transform]]].withDefault(_ => Seq.empty)
    a.map(Dependency.fromTransform).zipWithIndex.foreach{ case (t, idx) => m(idx) = Seq(t) }

    patches.foreach {
      case Add(line, txs) => m(line - 1) = m(line - 1) ++ txs
      case Del(line)      => m.remove(line - 1)
    }

    val patched = scala.collection.immutable.TreeMap(m.toArray:_*).values.flatten

    patched
      .zip(b.flattenedTransformOrder.map(Dependency.fromTransform))
      .foreach{ case (aa, bb) => bb should be (aa) }

    info(s"found ${b.flattenedTransformOrder.size} transforms")
    patched.size should be (b.flattenedTransformOrder.size)
  }

  behavior of "ChirrtlToHighFirrtl"

  it should "replicate the old order" in {
    val tm = new TransformManager(Forms.MinimalHighForm, Forms.ChirrtlForm)
    val patches = Seq(
      Add(5, Seq(Dependency[firrtl.annotations.transforms.CleanupNamedTargets]))
    )
    compare(legacyTransforms(new firrtl.ChirrtlToHighFirrtl), tm, patches)
  }

  behavior of "IRToWorkingIR"

  it should "replicate the old order" in {
    val tm = new TransformManager(Forms.WorkingIR, Forms.MinimalHighForm)
    compare(legacyTransforms(new firrtl.IRToWorkingIR), tm)
  }

  behavior of "ResolveAndCheck"

  it should "replicate the old order" in {
    val tm = new TransformManager(Forms.Resolved, Forms.WorkingIR)
    val patches = Seq(
      // Uniquify is now part of [[firrtl.passes.LowerTypes]]
      Del(5), Del(6), Del(7),
      Add(14, Seq(Dependency.fromTransform(firrtl.passes.CheckTypes)))
    )
    compare(legacyTransforms(new ResolveAndCheck), tm, patches)
  }

  behavior of "HighFirrtlToMiddleFirrtl"

  it should "replicate the old order" in {
    val tm = new TransformManager(Forms.MidForm, Forms.Deduped)
    val patches = Seq(
      Add(4, Seq(Dependency(firrtl.passes.ResolveFlows))),
      Add(5, Seq(Dependency(firrtl.passes.ResolveKinds))),
      // Uniquify is now part of [[firrtl.passes.LowerTypes]]
      Del(6),
      Add(6, Seq(Dependency(firrtl.passes.ResolveFlows))),
      Del(7),
      Del(8),
      Add(7, Seq(Dependency[firrtl.passes.ExpandWhensAndCheck])),
      Del(11),
      Del(12),
      Del(13),
      Add(12, Seq(Dependency(firrtl.passes.ResolveFlows),
                  Dependency[firrtl.passes.InferWidths])),
      Del(14),
      Add(15, Seq(Dependency(firrtl.passes.ResolveKinds),
                  Dependency(firrtl.passes.InferTypes))),
      // TODO
      Add(17, Seq(Dependency[firrtl.transforms.formal.AssertSubmoduleAssumptions]))
    )
    compare(legacyTransforms(new HighFirrtlToMiddleFirrtl), tm, patches)
  }

  behavior of "MiddleFirrtlToLowFirrtl"

  it should "replicate the old order" in {
    val tm = new TransformManager(Forms.LowForm, Forms.MidForm)
    val patches = Seq(
      // Uniquify is now part of [[firrtl.passes.LowerTypes]]
      Del(2), Del(3), Del(5),
      // RemoveWires now visibly invalidates ResolveKinds
      Add(11, Seq(Dependency(firrtl.passes.ResolveKinds)))
    )
    compare(legacyTransforms(new MiddleFirrtlToLowFirrtl), tm, patches)
  }

  behavior of "MinimumLowFirrtlOptimization"

  it should "replicate the old order" in {
    val tm = new TransformManager(Forms.LowFormMinimumOptimized, Forms.LowForm)
    val patches = Seq(
      Add(4, Seq(Dependency(firrtl.passes.ResolveFlows))),
      Add(6, Seq(Dependency[firrtl.transforms.LegalizeAndReductionsTransform],
                 Dependency(firrtl.passes.ResolveKinds)))
    )
    compare(legacyTransforms(new MinimumLowFirrtlOptimization), tm, patches)
  }

  behavior of "LowFirrtlOptimization"

  it should "replicate the old order" in {
    val tm = new TransformManager(Forms.LowFormOptimized, Forms.LowForm)
    val patches = Seq(
      Add(6, Seq(Dependency(firrtl.passes.ResolveFlows))),
      Add(7, Seq(Dependency(firrtl.passes.Legalize))),
      Add(8, Seq(Dependency[firrtl.transforms.LegalizeAndReductionsTransform],
                 Dependency(firrtl.passes.ResolveKinds)))
    )
    compare(legacyTransforms(new LowFirrtlOptimization), tm, patches)
  }

  behavior of "VerilogMinimumOptimized"

  it should "replicate the old order" in {
    val legacy = Seq(
      new firrtl.transforms.BlackBoxSourceHelper,
      new firrtl.transforms.FixAddingNegativeLiterals,
      new firrtl.transforms.ReplaceTruncatingArithmetic,
      new firrtl.transforms.InlineBitExtractionsTransform,
      new firrtl.transforms.PropagatePresetAnnotations,
      new firrtl.transforms.InlineCastsTransform,
      new firrtl.transforms.LegalizeClocksTransform,
      new firrtl.transforms.FlattenRegUpdate,
      firrtl.passes.VerilogModulusCleanup,
      new firrtl.transforms.VerilogRename,
      firrtl.passes.VerilogPrep,
      new firrtl.AddDescriptionNodes)
    val tm = new TransformManager(Forms.VerilogMinimumOptimized, (new firrtl.VerilogEmitter).prerequisites)
    compare(legacy, tm)
  }

  behavior of "VerilogOptimized"

  it should "replicate the old order" in {
    val legacy = Seq(
      new firrtl.transforms.BlackBoxSourceHelper,
      new firrtl.transforms.FixAddingNegativeLiterals,
      new firrtl.transforms.ReplaceTruncatingArithmetic,
      new firrtl.transforms.InlineBitExtractionsTransform,
      new firrtl.transforms.PropagatePresetAnnotations,
      new firrtl.transforms.InlineCastsTransform,
      new firrtl.transforms.LegalizeClocksTransform,
      new firrtl.transforms.FlattenRegUpdate,
      new firrtl.transforms.DeadCodeElimination,
      firrtl.passes.VerilogModulusCleanup,
      new firrtl.transforms.VerilogRename,
      firrtl.passes.VerilogPrep,
      new firrtl.AddDescriptionNodes)
    val tm = new TransformManager(Forms.VerilogOptimized, Forms.LowFormOptimized)
    compare(legacy, tm)
  }

  behavior of "Legacy Custom Transforms"

  it should "work for Chirrtl -> Chirrtl" in {
    val expected = new Transforms.ChirrtlToChirrtl :: new firrtl.ChirrtlEmitter :: Nil
    val tm = new TransformManager(Dependency[firrtl.ChirrtlEmitter] :: Dependency[Transforms.ChirrtlToChirrtl] :: Nil)
    compare(expected, tm)
  }

  it should "work for High -> High" in {
    val expected =
      new TransformManager(Forms.HighForm).flattenedTransformOrder ++
        Some(new Transforms.HighToHigh) ++
        (new TransformManager(Forms.MidForm, Forms.HighForm).flattenedTransformOrder)
    val tm = new TransformManager(Forms.MidForm :+ Dependency[Transforms.HighToHigh])
    compare(expected, tm)
  }

  it should "work for High -> Chirrtl" in {
    val expected =
      new TransformManager(Forms.HighForm).flattenedTransformOrder ++
        Some(new Transforms.HighToChirrtl) ++
        (new TransformManager(Forms.HighForm, Forms.ChirrtlForm).flattenedTransformOrder)
    val tm = new TransformManager(Forms.HighForm :+ Dependency[Transforms.HighToChirrtl])
    compare(expected, tm)
  }

  it should "work for Mid -> Mid" in {
    val expected =
      new TransformManager(Forms.MidForm).flattenedTransformOrder ++
        Some(new Transforms.MidToMid) ++
        (new TransformManager(Forms.LowForm, Forms.MidForm).flattenedTransformOrder)
    val tm = new TransformManager(Forms.LowForm :+ Dependency[Transforms.MidToMid])
    compare(expected, tm)
  }

  it should "work for Mid -> High" ignore {
    val expected =
      new TransformManager(Forms.MidForm).flattenedTransformOrder ++
        Some(new Transforms.MidToHigh) ++
        (new TransformManager(Forms.LowForm, Forms.MinimalHighForm).flattenedTransformOrder)
    val tm = new TransformManager(Forms.LowForm :+ Dependency[Transforms.MidToHigh])
    compare(expected, tm)
  }

  it should "work for Mid -> Chirrtl" ignore {
    val expected =
      new TransformManager(Forms.MidForm).flattenedTransformOrder ++
        Some(new Transforms.MidToChirrtl) ++
        (new TransformManager(Forms.LowForm, Forms.ChirrtlForm).flattenedTransformOrder)
    val tm = new TransformManager(Forms.LowForm :+ Dependency[Transforms.MidToChirrtl])
    compare(expected, tm)
  }

  it should "work for Low -> Low" in {
    val expected =
      new TransformManager(Forms.LowFormOptimized).flattenedTransformOrder ++
        Seq(new Transforms.LowToLow)
    val tm = new TransformManager(Forms.LowFormOptimized :+ Dependency[Transforms.LowToLow])
    compare(expected, tm)
  }

  it should "work for Low -> Mid" in {
    val expected =
      new TransformManager(Forms.LowFormOptimized).flattenedTransformOrder ++
        Seq(new Transforms.LowToMid) ++
        (new TransformManager(Forms.LowFormOptimized, Forms.MidForm).flattenedTransformOrder)
    val tm = new TransformManager(Forms.LowFormOptimized :+ Dependency[Transforms.LowToMid])
    compare(expected, tm)
  }

  it should "work for Low -> High" in {
    val expected =
      new TransformManager(Forms.LowFormOptimized).flattenedTransformOrder ++
        Seq(new Transforms.LowToHigh) ++
        (new TransformManager(Forms.LowFormOptimized, Forms.MinimalHighForm).flattenedTransformOrder)
    val tm = new TransformManager(Forms.LowFormOptimized :+ Dependency[Transforms.LowToHigh])
    compare(expected, tm)
  }

  it should "work for Low -> Chirrtl" in {
    val expected =
      new TransformManager(Forms.LowFormOptimized).flattenedTransformOrder ++
        Seq(new Transforms.LowToChirrtl) ++
        (new TransformManager(Forms.LowFormOptimized, Forms.ChirrtlForm).flattenedTransformOrder)
    val tm = new TransformManager(Forms.LowFormOptimized :+ Dependency[Transforms.LowToChirrtl])
    compare(expected, tm)
  }

}
