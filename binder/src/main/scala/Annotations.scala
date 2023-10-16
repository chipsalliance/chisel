// SPDX-License-Identifier: Apache-2.0

package chisel3.libfirtool

import chisel3.internal.panama.circt.PanamaCIRCTConverter
import firrtl.annotations.NoTargetAnnotation

object PreserveAggregateMode extends Enumeration {
  val None, OneDimVec, Vec, All = Value
}

object BuildMode extends Enumeration {
  val Debug, Release = Value
}

object CompanionMode extends Enumeration {
  val Bind, Instantiate, Drop = Value
}

object PreserveValuesMode extends Enumeration {
  val None, Named, All = Value
}

object RandomKind extends Enumeration {
  val None, Mem, Reg, All = Value
}

case class FirtoolLibOption(outputFilename: Option[String] = None,
                            disableAnnotationsUnknown: Option[Boolean] = None,
                            disableAnnotationsClassless: Option[Boolean] = None,
                            lowerAnnotationsNoRefTypePorts: Option[Boolean] = None,
                            preserveAggregate: Option[PreserveAggregateMode.Value] = None,
                            preserveValues: Option[PreserveValuesMode.Value] = None,
                            buildMode: Option[BuildMode.Value] = None,
                            disableOptimization: Option[Boolean] = None,
                            exportChiselInterface: Option[Boolean] = None,
                            chiselInterfaceOutDirectory: Option[String] = None,
                            vbToBv: Option[Boolean] = None,
                            dedup: Option[Boolean] = None,
                            companionMode: Option[CompanionMode.Value] = None,
                            disableAggressiveMergeConnections: Option[Boolean] = None,
                            emitOMIR: Option[Boolean] = None,
                            omirOutFile: Option[String] = None,
                            lowerMemories: Option[Boolean] = None,
                            blackBoxRootPath: Option[String] = None,
                            replSeqMem: Option[Boolean] = None,
                            replSeqMemFile: Option[String] = None,
                            extractTestCode: Option[Boolean] = None,
                            ignoreReadEnableMem: Option[Boolean] = None,
                            disableRandom: Option[RandomKind.Value] = None,
                            outputAnnotationFilename: Option[String] = None,
                            enableAnnotationWarning: Option[Boolean] = None,
                            addMuxPragmas: Option[Boolean] = None,
                            emitChiselAssertsAsSVA: Option[Boolean] = None,
                            emitSeparateAlwaysBlocks: Option[Boolean] = None,
                            etcDisableInstanceExtraction: Option[Boolean] = None,
                            etcDisableRegisterExtraction: Option[Boolean] = None,
                            etcDisableModuleInlining: Option[Boolean] = None,
                            addVivadoRAMAddressConflictSynthesisBugWorkaround: Option[Boolean] = None,
                            ckgModuleName: Option[String] = None,
                            ckgInputName: Option[String] = None,
                            ckgOutputName: Option[String] = None,
                            ckgEnableName: Option[String] = None,
                            ckgTestEnableName: Option[String] = None,
                            exportModuleHierarchy: Option[Boolean] = None,
                            stripFirDebugInfo: Option[Boolean] = None,
                            stripDebugInfo: Option[Boolean] = None)

case class FirtoolLibOptionAnnotation(option: FirtoolLibOption) extends NoTargetAnnotation

case class PanamaCIRCTConverterAnnotation(converter: PanamaCIRCTConverter) extends NoTargetAnnotation
