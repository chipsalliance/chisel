// SPDX-License-Identifier: Apache-2.0

// This should be the Scala API for user to invoke Firtool
// I(jiuyang) may want to promote it into a global user API for type safety in the userspace.(e.g. circt.options)
// It only declares types, converting them to cli/panama need to import corresponding implicit class
package chisel3.panamalib.option

sealed trait FirtoolOption

// All Firtool Options
case class AddMuxPragmas(value: Boolean) extends FirtoolOption
case class AddVivadoRAMAddressConflictSynthesisBugWorkaround(value: Boolean) extends FirtoolOption
case class BlackBoxRootPath(value: String) extends FirtoolOption
case class BuildMode(value: BuildModeEnum) extends FirtoolOption
case class CkgEnableName(value: String) extends FirtoolOption
case class CkgInputName(value: String) extends FirtoolOption
case class CkgModuleName(value: String) extends FirtoolOption
case class CkgOutputName(value: String) extends FirtoolOption
case class CkgTestEnableName(value: String) extends FirtoolOption
case class CompanionMode(value: CompanionModeEnum) extends FirtoolOption
case class DisableAggressiveMergeConnections(value: Boolean) extends FirtoolOption
case class DisableAnnotationsClassless(value: Boolean) extends FirtoolOption
case class DisableOptimization(value: Boolean) extends FirtoolOption
case class DisableRandom(value: RandomKindEnum) extends FirtoolOption
case class DisableUnknownAnnotations(value: Boolean) extends FirtoolOption
case class VerificationFlavor(value: VerificationFlavorEnum) extends FirtoolOption
case class EmitSeparateAlwaysBlocks(value: Boolean) extends FirtoolOption
case class EnableAnnotationWarning(value: Boolean) extends FirtoolOption
case class EtcDisableInstanceExtraction(value: Boolean) extends FirtoolOption
case class EtcDisableModuleInlining(value: Boolean) extends FirtoolOption
case class EtcDisableRegisterExtraction(value: Boolean) extends FirtoolOption
case class ExportModuleHierarchy(value: Boolean) extends FirtoolOption
case class ExtractTestCode(value: Boolean) extends FirtoolOption
case class IgnoreReadEnableMem(value: Boolean) extends FirtoolOption
case class LowerAnnotationsNoRefTypePorts(value: Boolean) extends FirtoolOption
case class LowerMemories(value: Boolean) extends FirtoolOption
case class NoDedup(value: Boolean) extends FirtoolOption
case class OutputAnnotationFilename(value: String) extends FirtoolOption
case class OutputFilename(value: String) extends FirtoolOption
case class PreserveAggregate(value: PreserveAggregateModeEnum) extends FirtoolOption
case class PreserveValues(value: PreserveValuesModeEnum) extends FirtoolOption
case class ReplSeqMem(value: Boolean) extends FirtoolOption
case class ReplSeqMemFile(value: String) extends FirtoolOption
case class StripDebugInfo(value: Boolean) extends FirtoolOption
case class StripFirDebugInfo(value: Boolean) extends FirtoolOption
case class VbToBv(value: Boolean) extends FirtoolOption

// All Enums
sealed trait BuildModeEnum
case object BuildModeDefault extends BuildModeEnum
case object BuildModeDebug extends BuildModeEnum
case object BuildModeRelease extends BuildModeEnum

sealed trait CompanionModeEnum
case object CompanionModeBind extends CompanionModeEnum
case object CompanionModeInstantiate extends CompanionModeEnum
case object CompanionModeDrop extends CompanionModeEnum

sealed trait RandomKindEnum
case object RandomKindNone extends RandomKindEnum
case object RandomKindMem extends RandomKindEnum
case object RandomKindReg extends RandomKindEnum
case object RandomKindAll extends RandomKindEnum

sealed trait PreserveAggregateModeEnum
case object PreserveAggregateModeNone extends PreserveAggregateModeEnum
case object PreserveAggregateModeOneDimVec extends PreserveAggregateModeEnum
case object PreserveAggregateModeVec extends PreserveAggregateModeEnum
case object PreserveAggregateModeAll extends PreserveAggregateModeEnum

sealed trait PreserveValuesModeEnum
case object PreserveValuesModeStrip extends PreserveValuesModeEnum
case object PreserveValuesModeNone extends PreserveValuesModeEnum
case object PreserveValuesModeNamed extends PreserveValuesModeEnum
case object PreserveValuesModeAll extends PreserveValuesModeEnum

sealed trait VerificationFlavorEnum
case object VerificationFlavorNone extends VerificationFlavorEnum
case object VerificationFlavorIfElseFatal extends VerificationFlavorEnum
case object VerificationFlavorImmediate extends VerificationFlavorEnum
case object VerificationFlavorSva extends VerificationFlavorEnum

case class FirtoolOptions(options: Set[FirtoolOption])
