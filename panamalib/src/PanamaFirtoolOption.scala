// SPDX-License-Identifier: Apache-2.0

package chisel3.panamalib.option

import chisel3.panamalib._

object PanamaFirtoolOption {
  implicit class FirtoolOptionToPanama(fo: FirtoolOption) {
    def toPanama(panamaCIRCT: PanamaCIRCT, options: CirctFirtoolFirtoolOptions): Unit = fo match {
      // format: off
      case AddMuxPragmas(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetAddMuxPragmas(options, value)
      case AddVivadoRAMAddressConflictSynthesisBugWorkaround(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetAddVivadoRAMAddressConflictSynthesisBugWorkaround(options, value)
      case BlackBoxRootPath(value: String) => panamaCIRCT.circtFirtoolOptionsSetBlackBoxRootPath(options, value)
      case BuildMode(value: BuildModeEnum) => panamaCIRCT.circtFirtoolOptionsSetBuildMode(options, value)
      case CkgEnableName(value: String) => panamaCIRCT.circtFirtoolOptionsSetCkgEnableName(options, value)
      case CkgInputName(value: String) => panamaCIRCT.circtFirtoolOptionsSetCkgInputName(options, value)
      case CkgModuleName(value: String) => panamaCIRCT.circtFirtoolOptionsSetCkgModuleName(options, value)
      case CkgOutputName(value: String) => panamaCIRCT.circtFirtoolOptionsSetCkgOutputName(options, value)
      case CkgTestEnableName(value: String) => panamaCIRCT.circtFirtoolOptionsSetCkgTestEnableName(options, value)
      case CompanionMode(value: CompanionModeEnum) => panamaCIRCT.circtFirtoolOptionsSetCompanionMode(options, value)
      case DisableAggressiveMergeConnections(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetDisableAggressiveMergeConnections(options, value)
      case DisableAnnotationsClassless(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetDisableAnnotationsClassless(options, value)
      case DisableOptimization(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetDisableOptimization(options, value)
      case DisableRandom(value: RandomKindEnum) => panamaCIRCT.circtFirtoolOptionsSetDisableRandom(options, value)
      case DisableUnknownAnnotations(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetDisableUnknownAnnotations(options, value)
      case VerificationFlavor(value: VerificationFlavorEnum) => panamaCIRCT.circtFirtoolOptionsSetVerificationFlavor(options, value)
      case EmitSeparateAlwaysBlocks(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetEmitSeparateAlwaysBlocks(options, value)
      case EnableAnnotationWarning(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetEnableAnnotationWarning(options, value)
      case EtcDisableInstanceExtraction(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetEtcDisableInstanceExtraction(options, value)
      case EtcDisableModuleInlining(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetEtcDisableModuleInlining(options, value)
      case EtcDisableRegisterExtraction(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetEtcDisableRegisterExtraction(options, value)
      case ExportModuleHierarchy(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetExportModuleHierarchy(options, value)
      case ExtractTestCode(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetExtractTestCode(options, value)
      case IgnoreReadEnableMem(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetIgnoreReadEnableMem(options, value)
      case LowerAnnotationsNoRefTypePorts(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetLowerAnnotationsNoRefTypePorts(options, value)
      case LowerMemories(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetLowerMemories(options, value)
      case NoDedup(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetNoDedup(options, value)
      case OutputAnnotationFilename(value: String) => panamaCIRCT.circtFirtoolOptionsSetOutputAnnotationFilename(options, value)
      case OutputFilename(value: String) => panamaCIRCT.circtFirtoolOptionsSetOutputFilename(options, value)
      case PreserveAggregate(value: PreserveAggregateModeEnum) => panamaCIRCT.circtFirtoolOptionsSetPreserveAggregate(options, value)
      case PreserveValues(value: PreserveValuesModeEnum) => panamaCIRCT.circtFirtoolOptionsSetPreserveValues(options, value)
      case ReplSeqMem(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetReplSeqMem(options, value)
      case ReplSeqMemFile(value: String) => panamaCIRCT.circtFirtoolOptionsSetReplSeqMemFile(options, value)
      case StripDebugInfo(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetStripDebugInfo(options, value)
      case StripFirDebugInfo(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetStripFirDebugInfo(options, value)
      case VbToBv(value: Boolean) => panamaCIRCT.circtFirtoolOptionsSetVbToBv(options, value)
      // format: on
    }
  }
  implicit class FirtoolOptionsToPanama(fos: FirtoolOptions) {
    def toPanama(panamaCIRCT: PanamaCIRCT): CirctFirtoolFirtoolOptions = {
      val firtoolOptions = panamaCIRCT.circtFirtoolOptionsCreateDefault()
      fos.options.foreach(fo => fo.toPanama(panamaCIRCT, firtoolOptions))
      firtoolOptions
    }
  }

  implicit def buildModeEnumtoPanama(e: BuildModeEnum): CirctFirtoolBuildMode = e match {
    case BuildModeDefault => CirctFirtoolBuildMode.Default
    case BuildModeDebug   => CirctFirtoolBuildMode.Debug
    case BuildModeRelease => CirctFirtoolBuildMode.Release
  }
  implicit def companionModeEnumtoPanama(e: CompanionModeEnum): CirctFirtoolCompanionMode = e match {
    case CompanionModeBind        => CirctFirtoolCompanionMode.Bind
    case CompanionModeInstantiate => CirctFirtoolCompanionMode.Instantiate
    case CompanionModeDrop        => CirctFirtoolCompanionMode.Drop
  }
  implicit def randomKindEnumtoPanama(e: RandomKindEnum): CirctFirtoolRandomKind = e match {
    case RandomKindNone => CirctFirtoolRandomKind.None
    case RandomKindMem  => CirctFirtoolRandomKind.Mem
    case RandomKindReg  => CirctFirtoolRandomKind.Reg
    case RandomKindAll  => CirctFirtoolRandomKind.All
  }
  implicit def preserveAggregateModeEnumtoPanama(e: PreserveAggregateModeEnum): CirctFirtoolPreserveAggregateMode =
    e match {
      case PreserveAggregateModeNone      => CirctFirtoolPreserveAggregateMode.None
      case PreserveAggregateModeOneDimVec => CirctFirtoolPreserveAggregateMode.OneDimVec
      case PreserveAggregateModeVec       => CirctFirtoolPreserveAggregateMode.Vec
      case PreserveAggregateModeAll       => CirctFirtoolPreserveAggregateMode.All
    }
  implicit def preserveValuesModeEnumtoPanama(e: PreserveValuesModeEnum): CirctFirtoolPreserveValuesMode = e match {
    case PreserveValuesModeStrip => CirctFirtoolPreserveValuesMode.Strip
    case PreserveValuesModeNone  => CirctFirtoolPreserveValuesMode.None
    case PreserveValuesModeNamed => CirctFirtoolPreserveValuesMode.Named
    case PreserveValuesModeAll   => CirctFirtoolPreserveValuesMode.All
  }
  implicit def verificationFlavor(e: VerificationFlavorEnum): CirctFirtoolVerificationFlavor = e match {
    case VerificationFlavorNone        => CirctFirtoolVerificationFlavor.None
    case VerificationFlavorIfElseFatal => CirctFirtoolVerificationFlavor.IfElseFatal
    case VerificationFlavorImmediate   => CirctFirtoolVerificationFlavor.Immediate
    case VerificationFlavorSva         => CirctFirtoolVerificationFlavor.Sva
  }
}
