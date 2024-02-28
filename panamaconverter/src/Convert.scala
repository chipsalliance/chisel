// SPDX-License-Identifier: Apache-2.0

package chisel3.panamaconverter.stage

import chisel3.panamaconverter.PanamaCIRCTConverter
import chisel3.panamalib.option.FirtoolOptions
import chisel3.stage.ChiselCircuitAnnotation
import chisel3.stage.phases.Elaborate
import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.internal.WriteableCircuitAnnotation
import firrtl.options.{Dependency, Phase}

case class PanamaCIRCTConverterAnnotation(converter: PanamaCIRCTConverter) extends NoTargetAnnotation
case class FirtoolOptionsAnnotation(firtoolOptions: FirtoolOptions) extends NoTargetAnnotation

object Convert extends Phase {
  override def prerequisites = Seq(Dependency[Elaborate])
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  def transform(annotations: AnnotationSeq): AnnotationSeq =
    annotations.flatMap {
      case c @ ChiselCircuitAnnotation(circuit) =>
        Seq(
          c,
          PanamaCIRCTConverterAnnotation(
            PanamaCIRCTConverter.convert(
              circuit,
              annotations.collectFirst {
                case FirtoolOptionsAnnotation(firtoolOptions) => firtoolOptions
              },
              firrtl.annotations.JsonProtocol.serialize(circuit.firrtlAnnotations.filter { anno =>
                Seq(
                  // This is all annotations that circt can parse(but may not use)
                  // It should be updated form [[https://github.com/llvm/circt/blob/main/include/circt/Dialect/FIRRTL/AnnotationDetails.h]]
                  // format: off
                  "circt.ConventionAnnotation",
                  "firrtl.transforms.DontTouchAnnotation",
                  "chisel3.experimental.EnumAnnotations$EnumComponentAnnotation",
                  "chisel3.experimental.EnumAnnotations$EnumDefAnnotation",
                  "chisel3.experimental.EnumAnnotations$EnumVecAnnotation",
                  "chisel3.util.experimental.ForceNameAnnotation",
                  "chisel3.util.experimental.decode.DecodeTableAnnotation",
                  "firrtl.transforms.FlattenAnnotation",
                  "firrtl.passes.InlineAnnotation",
                  "chisel3.experimental.Trace$TraceNameAnnotation",
                  "chisel3.experimental.Trace$TraceAnnotation",
                  "freechips.rocketchip.objectmodel.OMIRAnnotation",
                  "freechips.rocketchip.objectmodel.OMIRFileAnnotation",
                  "freechips.rocketchip.objectmodel.OMIRTracker",
                  "firrtl.transforms.BlackBoxInlineAnno",
                  "firrtl.transforms.BlackBoxPathAnno",
                  "firrtl.transforms.BlackBoxTargetDirAnno",
                  "firrtl.transforms.BlackBoxResourceFileNameAnno",
                  "firrtl.transforms.BlackBox",
                  "firrtl.transforms.MustDeduplicateAnnotation",
                  "firrtl.stage.RunFirrtlTransformAnnotation",
                  "sifive.enterprise.firrtl.ExtractAssertionsAnnotation",
                  "sifive.enterprise.firrtl.ExtractAssumptionsAnnotation",
                  "sifive.enterprise.firrtl.ExtractCoverageAnnotation",
                  "sifive.enterprise.firrtl.TestBenchDirAnnotation",
                  "sifive.enterprise.firrtl.ModuleHierarchyAnnotation",
                  "sifive.enterprise.firrtl.TestHarnessHierarchyAnnotation",
                  "sifive.enterprise.firrtl.RetimeModulesAnnotation",
                  "freechips.rocketchip.util.RetimeModuleAnnotation",
                  "freechips.rocketchip.annotations.InternalVerifBlackBoxAnnotation",
                  "sifive.enterprise.firrtl.MetadataDirAnnotation",
                  "firrtl.transforms.NoDedupAnnotation",
                  "firrtl.transforms.DedupGroupAnnotation",
                  "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation",
                  "sifive.enterprise.firrtl.DFTClockDividerBypassAnnotation",
                  "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
                  "sifive.enterprise.grandcentral.ViewAnnotation",
                  "sifive.enterprise.grandcentral.ViewAnnotation.companion",
                  "sifive.enterprise.grandcentral.PrefixInterfacesAnnotation",
                  "sifive.enterprise.grandcentral.AugmentedGroundType",
                  "sifive.enterprise.grandcentral.AugmentedBundleType",
                  "sifive.enterprise.grandcentral.DataTapsAnnotation",
                  "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox",
                  "sifive.enterprise.grandcentral.MemTapAnnotation",
                  "sifive.enterprise.grandcentral.MemTapAnnotation.blackbox",
                  "sifive.enterprise.grandcentral.MemTapAnnotation.port",
                  "sifive.enterprise.grandcentral.MemTapAnnotation.source",
                  "sifive.enterprise.grandcentral.DeletedDataTapKey",
                  "sifive.enterprise.grandcentral.LiteralDataTapKey",
                  "sifive.enterprise.grandcentral.ReferenceDataTapKey",
                  "sifive.enterprise.grandcentral.ReferenceDataTapKey.port",
                  "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
                  "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
                  "sifive.enterprise.grandcentral.DataTapModuleSignalKey.port",
                  "sifive.enterprise.grandcentral.DataTapModuleSignalKey.source",
                  "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
                  "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
                  "sifive.enterprise.grandcentral.SignalDriverAnnotation",
                  "sifive.enterprise.grandcentral.SignalDriverAnnotation.target",
                  "sifive.enterprise.grandcentral.SignalDriverAnnotation.module",
                  "sifive.enterprise.firrtl.MarkDUTAnnotation",
                  "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation",
                  "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
                  "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
                  "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
                  "sifive.enterprise.firrtl.DontObfuscateModuleAnnotation",
                  "sifive.enterprise.firrtl.ScalaClassAnnotation",
                  "sifive.enterprise.firrtl.ElaborationArtefactsDirectory",
                  "sifive.enterprise.grandcentral.phases.SubCircuitsTargetDirectory",
                  "sifive.enterprise.firrtl.TestHarnessPathAnnotation",
                  "sifive.enterprise.grandcentral.SubCircuitDirAnnotation",
                  "sifive.enterprise.firrtl.FullAsyncResetAnnotation",
                  "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation",
                  "sifive.enterprise.firrtl.ConvertMemToRegOfVecAnnotation$",
                  "sifive.enterprise.firrtl.ExcludeMemFromMemToRegOfVec",
                  "sifive.enterprise.firrtl.ExtractBlackBoxAnnotation",
                  "sifive.enterprise.firrtl.ExtractClockGatesFileAnnotation",
                  "sifive.enterprise.firrtl.ExtractSeqMemsFileAnnotation",
                  "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
                  "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
                  "firrtl.annotations.LoadMemoryAnnotation",
                  "firrtl.annotations.MemoryFileInlineAnnotation",
                  "firrtl.passes.wiring.SinkAnnotation",
                  "firrtl.passes.wiring.SourceAnnotation",
                  "firrtl.AttributeAnnotation",
                  // format: on
                ).contains(anno.getClass.getName)
              }.toSeq)
            )
          )
        )
      case a => Seq(a)
    }
}
