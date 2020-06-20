// See LICENSE for license details.

package firrtlTests

import org.scalatest.{FlatSpec, Matchers}

import firrtl._
import firrtl.passes
import firrtl.options.Dependency
import firrtl.stage.{Forms, TransformManager}
import firrtl.transforms.IdentityTransform

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

class LoweringCompilersSpec extends FlatSpec with Matchers {

  def legacyTransforms(a: CoreTransform): Seq[Transform] = a match {
    case _: ChirrtlToHighFirrtl => Seq(
      passes.CheckChirrtl,
      passes.CInferTypes,
      passes.CInferMDir,
      passes.RemoveCHIRRTL)
    case _: IRToWorkingIR => Seq(passes.ToWorkingIR)
    case _: ResolveAndCheck => Seq(
      passes.CheckHighForm,
      passes.ResolveKinds,
      passes.InferTypes,
      passes.CheckTypes,
      passes.Uniquify,
      passes.ResolveKinds,
      passes.InferTypes,
      passes.ResolveFlows,
      passes.CheckFlows,
      new passes.InferBinaryPoints,
      new passes.TrimIntervals,
      new passes.InferWidths,
      passes.CheckWidths,
      new firrtl.transforms.InferResets)
    case _: HighFirrtlToMiddleFirrtl => Seq(
      passes.PullMuxes,
      passes.ReplaceAccesses,
      passes.ExpandConnects,
      passes.ZeroLengthVecs,
      passes.RemoveAccesses,
      passes.Uniquify,
      passes.ExpandWhens,
      passes.CheckInitialization,
      passes.ResolveKinds,
      passes.InferTypes,
      passes.CheckTypes,
      passes.ResolveFlows,
      new passes.InferWidths,
      passes.CheckWidths,
      new passes.RemoveIntervals,
      passes.ConvertFixedToSInt,
      passes.ZeroWidth,
      passes.InferTypes)
    case _: MiddleFirrtlToLowFirrtl => Seq(
      passes.LowerTypes,
      passes.ResolveKinds,
      passes.InferTypes,
      passes.ResolveFlows,
      new passes.InferWidths,
      passes.Legalize,
      firrtl.transforms.RemoveReset,
      passes.ResolveFlows,
      new firrtl.transforms.CheckCombLoops,
      new checks.CheckResets,
      new firrtl.transforms.RemoveWires)
    case _: LowFirrtlOptimization => Seq(
      passes.RemoveValidIf,
      new firrtl.transforms.ConstantPropagation,
      passes.PadWidths,
      new firrtl.transforms.ConstantPropagation,
      passes.Legalize,
      passes.memlib.VerilogMemDelays, // TODO move to Verilog emitter
      new firrtl.transforms.ConstantPropagation,
      passes.SplitExpressions,
      new firrtl.transforms.CombineCats,
      passes.CommonSubexpressionElimination,
      new firrtl.transforms.DeadCodeElimination)
    case _: MinimumLowFirrtlOptimization => Seq(
      passes.RemoveValidIf,
      passes.PadWidths,
      passes.Legalize,
      passes.memlib.VerilogMemDelays, // TODO move to Verilog emitter
      passes.SplitExpressions)
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
    compare(legacyTransforms(new firrtl.ChirrtlToHighFirrtl), tm)
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
      Add(6, Seq(Dependency(firrtl.passes.ResolveKinds),
                 Dependency(firrtl.passes.InferTypes),
                 Dependency(firrtl.passes.ResolveFlows))),
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
                  Dependency(firrtl.passes.InferTypes)))
    )
    compare(legacyTransforms(new HighFirrtlToMiddleFirrtl), tm, patches)
  }

  behavior of "MiddleFirrtlToLowFirrtl"

  it should "replicate the old order" in {
    val tm = new TransformManager(Forms.LowForm, Forms.MidForm)
    compare(legacyTransforms(new MiddleFirrtlToLowFirrtl), tm)
  }

  behavior of "MinimumLowFirrtlOptimization"

  it should "replicate the old order" in {
    val tm = new TransformManager(Forms.LowFormMinimumOptimized, Forms.LowForm)
    val patches = Seq(
      Add(4, Seq(Dependency(firrtl.passes.ResolveFlows))),
      Add(5, Seq(Dependency(firrtl.passes.ResolveKinds)))
    )
    compare(legacyTransforms(new MinimumLowFirrtlOptimization), tm, patches)
  }

  behavior of "LowFirrtlOptimization"

  it should "replicate the old order" in {
    val tm = new TransformManager(Forms.LowFormOptimized, Forms.LowForm)
    val patches = Seq(
      Add(6, Seq(Dependency(firrtl.passes.ResolveFlows))),
      Add(7, Seq(Dependency(firrtl.passes.Legalize))),
      Add(8, Seq(Dependency(firrtl.passes.ResolveKinds)))
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

  it should "work for Mid -> High" in {
    val expected =
      new TransformManager(Forms.MidForm).flattenedTransformOrder ++
        Some(new Transforms.MidToHigh) ++
        (new TransformManager(Forms.LowForm, Forms.MinimalHighForm).flattenedTransformOrder)
    val tm = new TransformManager(Forms.LowForm :+ Dependency[Transforms.MidToHigh])
    compare(expected, tm)
  }

  it should "work for Mid -> Chirrtl" in {
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

  it should "schedule inputForm=LowForm after MiddleFirrtlToLowFirrtl for the LowFirrtlEmitter" in {
    val expected =
      new TransformManager(Forms.LowForm).flattenedTransformOrder ++
        Seq(new Transforms.LowToLow, new firrtl.LowFirrtlEmitter)
    val tm = (new TransformManager(Seq(Dependency[firrtl.LowFirrtlEmitter], Dependency[Transforms.LowToLow])))
    compare(expected, tm)
  }

  it should "schedule inputForm=LowForm after MinimumLowFirrtlOptimizations for the MinimalVerilogEmitter" in {
    val expected =
      new TransformManager(Forms.LowFormMinimumOptimized).flattenedTransformOrder ++
        Seq(new Transforms.LowToLow, new firrtl.MinimumVerilogEmitter)
    val tm = (new TransformManager(Seq(Dependency[firrtl.MinimumVerilogEmitter], Dependency[Transforms.LowToLow])))
    val patches = Seq(
      Add(62, Seq(Dependency[firrtl.transforms.LegalizeAndReductionsTransform]))
    )
    compare(expected, tm, patches)
  }

  it should "schedule inputForm=LowForm after LowFirrtlOptimizations for the VerilogEmitter" in {
    val expected =
      new TransformManager(Forms.LowFormOptimized).flattenedTransformOrder ++
        Seq(new Transforms.LowToLow, new firrtl.VerilogEmitter)
    val tm = (new TransformManager(Seq(Dependency[firrtl.VerilogEmitter], Dependency[Transforms.LowToLow])))
    val patches = Seq(
      Add(69, Seq(Dependency[firrtl.transforms.LegalizeAndReductionsTransform]))
    )
    compare(expected, tm, patches)
  }

}
