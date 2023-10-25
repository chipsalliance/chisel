// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.panama.circt

import java.lang.foreign._
import java.lang.foreign.MemorySegment.NULL
import java.lang.foreign.ValueLayout._

import org.llvm.circt
import org.llvm.circt.CAPI

// Wrapper for CIRCT APIs with Panama framework
class PanamaCIRCT {
  // Open an arena for memory management of MLIR API calling in this context instance
  private val arena = Arena.ofConfined()

  // Create MLIR context and register dialects we need
  private val mlirCtx = {
    val mlirCtx = CAPI.mlirContextCreate(arena)

    // Register dialects
    Seq(
      CAPI.mlirGetDialectHandle__firrtl__(arena),
      CAPI.mlirGetDialectHandle__chirrtl__(arena),
      CAPI.mlirGetDialectHandle__sv__(arena),
      CAPI.mlirGetDialectHandle__seq__(arena)
    ).foreach(CAPI.mlirDialectHandleLoadDialect(arena, _, mlirCtx))

    mlirCtx
  }

  // Public items for outside

  // Constants for this instance
  val unkLoc = mlirLocationUnknownGet()
  val emptyArrayAttr = mlirArrayAttrGet(Seq.empty)

  //////////////////////////////////////////////////
  // Helpers
  //

  private def newString(string: String): MlirStringRef = {
    val bytes = string.getBytes()
    val buffer = arena.allocate(bytes.length + 1)
    buffer.copyFrom(MemorySegment.ofArray(bytes))
    MlirStringRef(CAPI.mlirStringRefCreateFromCString(arena, buffer))
  }

  private def newStringCallback(callback: String => Unit): MlirStringCallback = {
    val cb = new circt.MlirStringCallback {
      def apply(message: MemorySegment, userData: MemorySegment) = {
        callback(MlirStringRef(message).toString)
      }
    }
    MlirStringCallback(circt.MlirStringCallback.allocate(cb, arena))
  }

  private def seqToArray[T <: ForeignType[_]](xs: Seq[T]): (MemorySegment, Int) = {
    if (xs.nonEmpty) {
      val sizeOfT = xs(0).sizeof

      val buffer = arena.allocate(sizeOfT * xs.length)
      xs.zipWithIndex.foreach {
        case (x, i) =>
          x.get match {
            case value: MemorySegment => buffer.asSlice(sizeOfT * i, sizeOfT).copyFrom(value)
            case value: Int           => buffer.setAtIndex(CAPI.C_INT, i, value)
          }
      }
      (buffer, xs.length)
    } else {
      (NULL, 0)
    }
  }

  //////////////////////////////////////////////////
  // CIRCT APIs
  //

  def mlirModuleCreateEmpty(location: MlirLocation) = MlirModule(CAPI.mlirModuleCreateEmpty(arena, location.get))

  def mlirModuleGetBody(module: MlirModule) = MlirBlock(CAPI.mlirModuleGetBody(arena, module.get))

  def mlirModuleGetOperation(module: MlirModule) = MlirOperation(CAPI.mlirModuleGetOperation(arena, module.get))

  def mlirOperationStateGet(name: String, loc: MlirLocation) = MlirOperationState(
    CAPI.mlirOperationStateGet(arena, newString(name).get, loc.get)
  )

  def mlirOperationStateAddAttributes(state: MlirOperationState, attrs: Seq[MlirNamedAttribute]) = {
    if (attrs.nonEmpty) {
      val (ptr, length) = seqToArray(attrs)
      CAPI.mlirOperationStateAddAttributes(state.get, length, ptr)
    }
  }

  def mlirOperationStateAddOperands(state: MlirOperationState, operands: Seq[MlirValue]) = {
    if (operands.nonEmpty) {
      val (ptr, length) = seqToArray(operands)
      CAPI.mlirOperationStateAddOperands(state.get, length, ptr)
    }
  }

  def mlirOperationStateAddResults(state: MlirOperationState, results: Seq[MlirType]) = {
    if (results.nonEmpty) {
      val (ptr, length) = seqToArray(results)
      CAPI.mlirOperationStateAddResults(state.get, length, ptr)
    }
  }

  def mlirOperationStateAddOwnedRegions(state: MlirOperationState, regions: Seq[MlirRegion]) = {
    if (regions.nonEmpty) {
      val (ptr, length) = seqToArray(regions)
      CAPI.mlirOperationStateAddOwnedRegions(state.get, length, ptr)
    }
  }

  def mlirOperationCreate(state: MlirOperationState) = MlirOperation(CAPI.mlirOperationCreate(arena, state.get))

  def mlirOperationGetResult(operation: MlirOperation, pos: Int) = MlirValue(
    CAPI.mlirOperationGetResult(arena, operation.get, pos)
  )

  def mlirBlockCreate(args: Seq[MlirType], locs: Seq[MlirLocation]): MlirBlock = {
    assert(args.length == locs.length)
    val length = args.length
    MlirBlock(CAPI.mlirBlockCreate(arena, length, seqToArray(args)._1, seqToArray(locs)._1))
  }

  def mlirBlockGetArgument(block: MlirBlock, pos: Int) = MlirValue(CAPI.mlirBlockGetArgument(arena, block.get, pos))

  def mlirBlockAppendOwnedOperation(block: MlirBlock, operation: MlirOperation) = {
    CAPI.mlirBlockAppendOwnedOperation(block.get, operation.get)
  }

  def mlirBlockInsertOwnedOperationAfter(block: MlirBlock, reference: MlirOperation, operation: MlirOperation) = {
    CAPI.mlirBlockInsertOwnedOperationAfter(block.get, reference.get, operation.get)
  }

  def mlirBlockInsertOwnedOperationBefore(block: MlirBlock, reference: MlirOperation, operation: MlirOperation) = {
    CAPI.mlirBlockInsertOwnedOperationBefore(block.get, reference.get, operation.get)
  }

  def mlirRegionCreate() = MlirRegion(CAPI.mlirRegionCreate(arena))

  def mlirRegionAppendOwnedBlock(region: MlirRegion, block: MlirBlock) = {
    CAPI.mlirRegionAppendOwnedBlock(region.get, block.get)
  }

  def mlirLocationGetAttribute(loc: MlirLocation) = MlirAttribute(CAPI.mlirLocationGetAttribute(arena, loc.get))

  def mlirLocationUnknownGet() = MlirLocation(CAPI.mlirLocationUnknownGet(arena, mlirCtx))

  def mlirLocationFileLineColGet(filename: String, line: Int, col: Int) = MlirLocation(
    CAPI.mlirLocationFileLineColGet(arena, mlirCtx, newString(filename).get, line, col)
  )

  def mlirIdentifierGet(string: String) = MlirIdentifier(CAPI.mlirIdentifierGet(arena, mlirCtx, newString(string).get))

  def mlirNamedAttributeGet(name: String, attr: MlirAttribute) = MlirNamedAttribute(
    CAPI.mlirNamedAttributeGet(arena, mlirIdentifierGet(name).get, attr.get)
  )

  def mlirArrayAttrGet(elements: Seq[MlirAttribute]): MlirAttribute = {
    val (ptr, length) = seqToArray(elements)
    MlirAttribute(CAPI.mlirArrayAttrGet(arena, mlirCtx, length, ptr))
  }

  def mlirTypeAttrGet(tpe: MlirType) = MlirAttribute(CAPI.mlirTypeAttrGet(arena, tpe.get))

  def mlirStringAttrGet(string: String) = MlirAttribute(CAPI.mlirStringAttrGet(arena, mlirCtx, newString(string).get))

  def mlirIntegerAttrGet(tpe: MlirType, value: Int) = MlirAttribute(CAPI.mlirIntegerAttrGet(arena, tpe.get, value))

  def mlirFloatAttrDoubleGet(tpe: MlirType, value: Double) = MlirAttribute(
    CAPI.mlirFloatAttrDoubleGet(arena, mlirCtx, tpe.get, value)
  )

  def mlirFlatSymbolRefAttrGet(symbol: String) = MlirAttribute(
    CAPI.mlirFlatSymbolRefAttrGet(arena, mlirCtx, newString(symbol).get)
  )

  def mlirNoneTypeGet() = MlirType(CAPI.mlirNoneTypeGet(arena, mlirCtx))

  def mlirIntegerTypeGet(bitwidth: Int) = MlirType(CAPI.mlirIntegerTypeGet(arena, mlirCtx, bitwidth))

  def mlirIntegerTypeUnsignedGet(bitwidth: Int) = MlirType(CAPI.mlirIntegerTypeUnsignedGet(arena, mlirCtx, bitwidth))

  def mlirIntegerTypeSignedGet(bitwidth: Int) = MlirType(CAPI.mlirIntegerTypeSignedGet(arena, mlirCtx, bitwidth))

  def mlirF64TypeGet() = MlirType(CAPI.mlirF64TypeGet(arena, mlirCtx))

  def mlirOperationPrint(op: MlirOperation, callback: String => Unit) =
    CAPI.mlirOperationPrint(op.get, newStringCallback(callback).get, NULL)

  def mlirExportFIRRTL(module: MlirModule, callback: String => Unit) = {
    CAPI.mlirExportFIRRTL(arena, module.get, newStringCallback(callback).get, NULL)
  }

  def mlirPassManagerCreate() = MlirPassManager(CAPI.mlirPassManagerCreate(arena, mlirCtx))

  def mlirPassManagerAddOwnedPass(pm: MlirPassManager, pass: MlirPass) = {
    CAPI.mlirPassManagerAddOwnedPass(pm.get, pass.get)
  }

  def mlirPassManagerGetNestedUnder(pm: MlirPassManager, operationName: String) = MlirOpPassManager(
    CAPI.mlirPassManagerGetNestedUnder(arena, pm.get, newString(operationName).get)
  )

  def mlirPassManagerRunOnOp(pm: MlirPassManager, op: MlirOperation) = MlirLogicalResult(
    CAPI.mlirPassManagerRunOnOp(arena, pm.get, op.get)
  )

  def mlirOpPassManagerAddOwnedPass(pm: MlirOpPassManager, pass: MlirPass) = {
    CAPI.mlirOpPassManagerAddOwnedPass(pm.get, pass.get)
  }

  def mlirOpPassManagerGetNestedUnder(pm: MlirOpPassManager, operationName: String) = MlirOpPassManager(
    CAPI.mlirOpPassManagerGetNestedUnder(arena, pm.get, newString(operationName).get)
  )

  def firtoolOptionsCreateDefault() = FirtoolOptions(CAPI.firtoolOptionsCreateDefault(arena))
  def firtoolOptionsDestroy(options:           FirtoolOptions) = CAPI.firtoolOptionsDestroy(options.get)
  def firtoolOptionsSetOutputFilename(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetOutputFilename(options.get, newString(value).get)
  def firtoolOptionsGetOutputFilename(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetOutputFilename(arena, options.get)).toString
  def firtoolOptionsSetDisableAnnotationsUnknown(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetDisableAnnotationsUnknown(options.get, value)
  def firtoolOptionsGetDisableAnnotationsUnknown(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetDisableAnnotationsUnknown(options.get)
  def firtoolOptionsSetDisableAnnotationsClassless(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetDisableAnnotationsClassless(options.get, value)
  def firtoolOptionsGetDisableAnnotationsClassless(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetDisableAnnotationsClassless(options.get)
  def firtoolOptionsSetLowerAnnotationsNoRefTypePorts(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetLowerAnnotationsNoRefTypePorts(options.get, value)
  def firtoolOptionsGetLowerAnnotationsNoRefTypePorts(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetLowerAnnotationsNoRefTypePorts(options.get)
  def firtoolOptionsSetPreserveAggregate(options: FirtoolOptions, value: FirtoolPreserveAggregateMode) =
    CAPI.firtoolOptionsSetPreserveAggregate(options.get, value.get)
  def firtoolOptionsGetPreserveAggregate(options: FirtoolOptions) = new FirtoolPreserveAggregateMode(
    CAPI.firtoolOptionsGetPreserveAggregate(options.get)
  )
  def firtoolOptionsSetPreserveValues(options: FirtoolOptions, value: FirtoolPreserveValuesMode) =
    CAPI.firtoolOptionsSetPreserveValues(options.get, value.get)
  def firtoolOptionsGetPreserveValues(options: FirtoolOptions) = new FirtoolPreserveValuesMode(
    CAPI.firtoolOptionsGetPreserveValues(options.get)
  )
  def firtoolOptionsSetBuildMode(options: FirtoolOptions, value: FirtoolBuildMode) =
    CAPI.firtoolOptionsSetBuildMode(options.get, value.get)
  def firtoolOptionsGetBuildMode(options: FirtoolOptions) = new FirtoolBuildMode(
    CAPI.firtoolOptionsGetBuildMode(options.get)
  )
  def firtoolOptionsSetDisableOptimization(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetDisableOptimization(options.get, value)
  def firtoolOptionsGetDisableOptimization(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetDisableOptimization(options.get)
  def firtoolOptionsSetExportChiselInterface(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetExportChiselInterface(options.get, value)
  def firtoolOptionsGetExportChiselInterface(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetExportChiselInterface(options.get)
  def firtoolOptionsSetChiselInterfaceOutDirectory(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetChiselInterfaceOutDirectory(options.get, newString(value).get)
  def firtoolOptionsGetChiselInterfaceOutDirectory(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetChiselInterfaceOutDirectory(arena, options.get)).toString
  def firtoolOptionsSetVbToBv(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetVbToBv(options.get, value)
  def firtoolOptionsGetVbToBv(options:        FirtoolOptions): Boolean = CAPI.firtoolOptionsGetVbToBv(options.get)
  def firtoolOptionsSetDedup(options:         FirtoolOptions, value: Boolean) = CAPI.firtoolOptionsSetDedup(options.get, value)
  def firtoolOptionsGetDedup(options:         FirtoolOptions): Boolean = CAPI.firtoolOptionsGetDedup(options.get)
  def firtoolOptionsSetCompanionMode(options: FirtoolOptions, value: FirtoolCompanionMode) =
    CAPI.firtoolOptionsSetCompanionMode(options.get, value.get)
  def firtoolOptionsGetCompanionMode(options:                     FirtoolOptions) = CAPI.firtoolOptionsGetCompanionMode(options.get)
  def firtoolOptionsSetDisableAggressiveMergeConnections(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetDisableAggressiveMergeConnections(options.get, value)
  def firtoolOptionsGetDisableAggressiveMergeConnections(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetDisableAggressiveMergeConnections(options.get)
  def firtoolOptionsSetEmitOMIR(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetEmitOMIR(options.get, value)
  def firtoolOptionsGetEmitOMIR(options:    FirtoolOptions): Boolean = CAPI.firtoolOptionsGetEmitOMIR(options.get)
  def firtoolOptionsSetOMIROutFile(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetOMIROutFile(options.get, newString(value).get)
  def firtoolOptionsGetOMIROutFile(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetOMIROutFile(arena, options.get)).toString
  def firtoolOptionsSetLowerMemories(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetLowerMemories(options.get, value)
  def firtoolOptionsGetLowerMemories(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetLowerMemories(options.get)
  def firtoolOptionsSetBlackBoxRootPath(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetBlackBoxRootPath(options.get, newString(value).get)
  def firtoolOptionsGetBlackBoxRootPath(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetBlackBoxRootPath(arena, options.get)).toString
  def firtoolOptionsSetReplSeqMem(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetReplSeqMem(options.get, value)
  def firtoolOptionsGetReplSeqMem(options:     FirtoolOptions): Boolean = CAPI.firtoolOptionsGetReplSeqMem(options.get)
  def firtoolOptionsSetReplSeqMemFile(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetReplSeqMemFile(options.get, newString(value).get)
  def firtoolOptionsGetReplSeqMemFile(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetReplSeqMemFile(arena, options.get)).toString
  def firtoolOptionsSetExtractTestCode(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetExtractTestCode(options.get, value)
  def firtoolOptionsGetExtractTestCode(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetExtractTestCode(options.get)
  def firtoolOptionsSetIgnoreReadEnableMem(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetIgnoreReadEnableMem(options.get, value)
  def firtoolOptionsGetIgnoreReadEnableMem(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetIgnoreReadEnableMem(options.get)
  def firtoolOptionsSetDisableRandom(options: FirtoolOptions, value: FirtoolRandomKind) =
    CAPI.firtoolOptionsSetDisableRandom(options.get, value.get)
  def firtoolOptionsGetDisableRandom(options: FirtoolOptions) = new FirtoolRandomKind(
    CAPI.firtoolOptionsGetDisableRandom(options.get)
  )
  def firtoolOptionsSetOutputAnnotationFilename(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetOutputAnnotationFilename(options.get, newString(value).get)
  def firtoolOptionsGetOutputAnnotationFilename(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetOutputAnnotationFilename(arena, options.get)).toString
  def firtoolOptionsSetEnableAnnotationWarning(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetEnableAnnotationWarning(options.get, value)
  def firtoolOptionsGetEnableAnnotationWarning(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetEnableAnnotationWarning(options.get)
  def firtoolOptionsSetAddMuxPragmas(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetAddMuxPragmas(options.get, value)
  def firtoolOptionsGetAddMuxPragmas(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetAddMuxPragmas(options.get)
  def firtoolOptionsSetEmitChiselAssertsAsSVA(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetEmitChiselAssertsAsSVA(options.get, value)
  def firtoolOptionsGetEmitChiselAssertsAsSVA(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetEmitChiselAssertsAsSVA(options.get)
  def firtoolOptionsSetEmitSeparateAlwaysBlocks(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetEmitSeparateAlwaysBlocks(options.get, value)
  def firtoolOptionsGetEmitSeparateAlwaysBlocks(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetEmitSeparateAlwaysBlocks(options.get)
  def firtoolOptionsSetEtcDisableInstanceExtraction(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetEtcDisableInstanceExtraction(options.get, value)
  def firtoolOptionsGetEtcDisableInstanceExtraction(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetEtcDisableInstanceExtraction(options.get)
  def firtoolOptionsSetEtcDisableRegisterExtraction(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetEtcDisableRegisterExtraction(options.get, value)
  def firtoolOptionsGetEtcDisableRegisterExtraction(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetEtcDisableRegisterExtraction(options.get)
  def firtoolOptionsSetEtcDisableModuleInlining(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetEtcDisableModuleInlining(options.get, value)
  def firtoolOptionsGetEtcDisableModuleInlining(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetEtcDisableModuleInlining(options.get)
  def firtoolOptionsSetAddVivadoRAMAddressConflictSynthesisBugWorkaround(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetAddVivadoRAMAddressConflictSynthesisBugWorkaround(options.get, value)
  def firtoolOptionsGetAddVivadoRAMAddressConflictSynthesisBugWorkaround(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetAddVivadoRAMAddressConflictSynthesisBugWorkaround(options.get)
  def firtoolOptionsSetCkgModuleName(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetCkgModuleName(options.get, newString(value).get)
  def firtoolOptionsGetCkgModuleName(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetCkgModuleName(arena, options.get)).toString
  def firtoolOptionsSetCkgInputName(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetCkgInputName(options.get, newString(value).get)
  def firtoolOptionsGetCkgInputName(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetCkgInputName(arena, options.get)).toString
  def firtoolOptionsSetCkgOutputName(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetCkgOutputName(options.get, newString(value).get)
  def firtoolOptionsGetCkgOutputName(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetCkgOutputName(arena, options.get)).toString
  def firtoolOptionsSetCkgEnableName(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetCkgEnableName(options.get, newString(value).get)
  def firtoolOptionsGetCkgEnableName(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetCkgEnableName(arena, options.get)).toString
  def firtoolOptionsSetCkgTestEnableName(options: FirtoolOptions, value: String) =
    CAPI.firtoolOptionsSetCkgTestEnableName(options.get, newString(value).get)
  def firtoolOptionsGetCkgTestEnableName(options: FirtoolOptions): String =
    (CAPI.firtoolOptionsGetCkgTestEnableName(arena, options.get)).toString
  def firtoolOptionsSetExportModuleHierarchy(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetExportModuleHierarchy(options.get, value)
  def firtoolOptionsGetExportModuleHierarchy(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetExportModuleHierarchy(options.get)
  def firtoolOptionsSetStripFirDebugInfo(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetStripFirDebugInfo(options.get, value)
  def firtoolOptionsGetStripFirDebugInfo(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetStripFirDebugInfo(options.get)
  def firtoolOptionsSetStripDebugInfo(options: FirtoolOptions, value: Boolean) =
    CAPI.firtoolOptionsSetStripDebugInfo(options.get, value)
  def firtoolOptionsGetStripDebugInfo(options: FirtoolOptions): Boolean =
    CAPI.firtoolOptionsGetStripDebugInfo(options.get)

  def firtoolPopulatePreprocessTransforms(pm: MlirPassManager, options: FirtoolOptions) = MlirLogicalResult(
    CAPI.firtoolPopulatePreprocessTransforms(arena, pm.get, options.get)
  )

  def firtoolPopulateCHIRRTLToLowFIRRTL(
    pm:            MlirPassManager,
    options:       FirtoolOptions,
    module:        MlirModule,
    inputFilename: String
  ) =
    MlirLogicalResult(
      CAPI.firtoolPopulateCHIRRTLToLowFIRRTL(arena, pm.get, options.get, newString(inputFilename).get)
    )

  def firtoolPopulateLowFIRRTLToHW(pm: MlirPassManager, options: FirtoolOptions) = MlirLogicalResult(
    CAPI.firtoolPopulateLowFIRRTLToHW(arena, pm.get, options.get)
  )

  def firtoolPopulateHWToSV(pm: MlirPassManager, options: FirtoolOptions) = MlirLogicalResult(
    CAPI.firtoolPopulateHWToSV(arena, pm.get, options.get)
  )

  def firtoolPopulateExportVerilog(pm: MlirPassManager, options: FirtoolOptions, callback: String => Unit) =
    MlirLogicalResult(
      CAPI.firtoolPopulateExportVerilog(arena, pm.get, options.get, newStringCallback(callback).get, NULL)
    )

  def firtoolPopulateExportSplitVerilog(pm: MlirPassManager, options: FirtoolOptions, directory: String) =
    MlirLogicalResult(
      CAPI.firtoolPopulateExportSplitVerilog(arena, pm.get, options.get, newString(directory).get)
    )

  def mlirLogicalResultIsSuccess(res: MlirLogicalResult): Boolean = circt.MlirLogicalResult.value$get(res.get) != 0

  def mlirLogicalResultIsFailure(res: MlirLogicalResult): Boolean = circt.MlirLogicalResult.value$get(res.get) == 0

  def firrtlTypeGetUInt(width: Int) = MlirType(CAPI.firrtlTypeGetUInt(arena, mlirCtx, width))

  def firrtlTypeGetSInt(width: Int) = MlirType(CAPI.firrtlTypeGetSInt(arena, mlirCtx, width))

  def firrtlTypeGetClock() = MlirType(CAPI.firrtlTypeGetClock(arena, mlirCtx))

  def firrtlTypeGetReset() = MlirType(CAPI.firrtlTypeGetReset(arena, mlirCtx))

  def firrtlTypeGetAsyncReset() = MlirType(CAPI.firrtlTypeGetAsyncReset(arena, mlirCtx))

  def firrtlTypeGetAnalog(width: Int) = MlirType(CAPI.firrtlTypeGetAnalog(arena, mlirCtx, width))

  def firrtlTypeGetVector(element: MlirType, count: Int) = MlirType(
    CAPI.firrtlTypeGetVector(arena, mlirCtx, element.get, count)
  )

  def firrtlTypeGetBundle(fields: Seq[FIRRTLBundleField]): MlirType = {
    val buffer = circt.FIRRTLBundleField.allocateArray(fields.length, arena)
    fields.zipWithIndex.foreach {
      case (field, i) =>
        val fieldBuffer = buffer.asSlice(circt.FIRRTLBundleField.sizeof() * i, circt.FIRRTLBundleField.sizeof())
        circt.FIRRTLBundleField.name$slice(fieldBuffer).copyFrom(mlirIdentifierGet(field.name).get)
        circt.FIRRTLBundleField.isFlip$set(fieldBuffer, field.isFlip)
        circt.FIRRTLBundleField.type$slice(fieldBuffer).copyFrom(field.tpe.get)
    }
    MlirType(CAPI.firrtlTypeGetBundle(arena, mlirCtx, fields.length, buffer))
  }

  def firrtlAttrGetPortDirs(dirs: Seq[FIRRTLPortDir]): MlirAttribute = {
    val (ptr, length) = seqToArray(dirs)
    MlirAttribute(CAPI.firrtlAttrGetPortDirs(arena, mlirCtx, length, ptr))
  }

  def firrtlAttrGetParamDecl(name: String, tpe: MlirType, value: MlirAttribute) = MlirAttribute(
    CAPI.firrtlAttrGetParamDecl(arena, mlirCtx, mlirIdentifierGet(name).get, tpe.get, value.get)
  )

  def firrtlAttrGetConvention(convention: FIRRTLConvention) = MlirAttribute(
    CAPI.firrtlAttrGetConvention(arena, mlirCtx, convention.value)
  )

  def firrtlAttrGetNameKind(nameKind: FIRRTLNameKind) = MlirAttribute(
    CAPI.firrtlAttrGetNameKind(arena, mlirCtx, nameKind.value)
  )

  def firrtlAttrGetRUW(ruw: firrtlAttrGetRUW) = MlirAttribute(CAPI.firrtlAttrGetRUW(arena, mlirCtx, ruw.value))

  def firrtlAttrGetMemDir(dir: FIRRTLMemDir) = MlirAttribute(CAPI.firrtlAttrGetMemDir(arena, mlirCtx, dir.value))

  def chirrtlTypeGetCMemory(elementType: MlirType, numElements: Int) = MlirType(
    CAPI.chirrtlTypeGetCMemory(arena, mlirCtx, elementType.get, numElements)
  )

  def chirrtlTypeGetCMemoryPort() = MlirType(CAPI.chirrtlTypeGetCMemoryPort(arena, mlirCtx))
}

//
// MLIR & CIRCT Types
//
// Since all structs returned from Panama framework are `MemorySegment`, which is like a `void *` in C.
// We create these type wrappers to wrap it, make them type-safe.
//

trait ForeignType[T] {
  private[circt] def get: T
  private[circt] val sizeof: Int
}

final case class MlirAttribute(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirAttribute.sizeof().toInt
}
object MlirAttribute {
  private[circt] def apply(ptr: MemorySegment) = new MlirAttribute(ptr)
}

final case class MlirNamedAttribute(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirNamedAttribute.sizeof().toInt
}
object MlirNamedAttribute {
  private[circt] def apply(ptr: MemorySegment) = new MlirNamedAttribute(ptr)
}

final case class MlirBlock(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirBlock.sizeof().toInt
}
object MlirBlock {
  private[circt] def apply(ptr: MemorySegment) = new MlirBlock(ptr)
}

final case class MlirRegion(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirRegion.sizeof().toInt
}
object MlirRegion {
  private[circt] def apply(ptr: MemorySegment) = new MlirRegion(ptr)
}

final case class MlirIdentifier(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirIdentifier.sizeof().toInt
}
object MlirIdentifier {
  private[circt] def apply(ptr: MemorySegment) = new MlirIdentifier(ptr)
}

final case class MlirLocation(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirLocation.sizeof().toInt
}
object MlirLocation {
  private[circt] def apply(ptr: MemorySegment) = new MlirLocation(ptr)
}

final case class MlirModule(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirModule.sizeof().toInt
}
object MlirModule {
  private[circt] def apply(ptr: MemorySegment) = new MlirModule(ptr)
}

final case class MlirOperation(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirOperation.sizeof().toInt
}
object MlirOperation {
  private[circt] def apply(ptr: MemorySegment) = new MlirOperation(ptr)
}

final case class MlirOperationState(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirOperationState.sizeof().toInt
}
object MlirOperationState {
  private[circt] def apply(ptr: MemorySegment) = new MlirOperationState(ptr)
}

final case class MlirType(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirType.sizeof().toInt
}
object MlirType {
  private[circt] def apply(ptr: MemorySegment) = new MlirType(ptr)
}

final case class MlirValue(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirValue.sizeof().toInt
}
object MlirValue {
  private[circt] def apply(ptr: MemorySegment) = new MlirValue(ptr)
}

final case class MlirStringRef(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirStringRef.sizeof().toInt

  override def toString: String = {
    var slice = circt.MlirStringRef.data$get(ptr).asSlice(0, circt.MlirStringRef.length$get(ptr))
    new String(slice.toArray(JAVA_BYTE))
  }
}
object MlirStringRef {
  private[circt] def apply(ptr: MemorySegment) = new MlirStringRef(ptr)
}

final case class MlirLogicalResult(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirLogicalResult.sizeof().toInt
}
object MlirLogicalResult {
  private[circt] def apply(ptr: MemorySegment) = new MlirLogicalResult(ptr)
}

final case class MlirStringCallback(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = CAPI.C_POINTER.byteSize().toInt
}
object MlirStringCallback {
  private[circt] def apply(ptr: MemorySegment) = new MlirStringCallback(ptr)
}

final case class MlirPassManager(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirPassManager.sizeof().toInt
}
object MlirPassManager {
  private[circt] def apply(ptr: MemorySegment) = new MlirPassManager(ptr)
}

final case class MlirOpPassManager(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirOpPassManager.sizeof().toInt
}
object MlirOpPassManager {
  private[circt] def apply(ptr: MemorySegment) = new MlirOpPassManager(ptr)
}

final case class MlirPass(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.MlirPass.sizeof().toInt
}
object MlirPass {
  private[circt] def apply(ptr: MemorySegment) = new MlirPass(ptr)
}

final case class FIRRTLBundleField(name: String, isFlip: Boolean, tpe: MlirType)

final case class FirtoolOptions(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.FirtoolOptions.sizeof().toInt
}
object FirtoolOptions {
  private[circt] def apply(ptr: MemorySegment) = new FirtoolOptions(ptr)
}

//
// MLIR & CIRCT Enums
//

sealed abstract class FIRRTLConvention(val value: Int) extends ForeignType[Int] {
  private[circt] def get = value
  private[circt] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FIRRTLConvention {
  final case object Internal extends FIRRTLConvention(value = CAPI.FIRRTL_CONVENTION_INTERNAL())
  final case object Scalarized extends FIRRTLConvention(value = CAPI.FIRRTL_CONVENTION_SCALARIZED())
}

sealed abstract class FIRRTLNameKind(val value: Int) extends ForeignType[Int] {
  private[circt] def get = value
  private[circt] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FIRRTLNameKind {
  final case object DroppableName extends FIRRTLNameKind(value = CAPI.FIRRTL_NAME_KIND_DROPPABLE_NAME())
  final case object InterestingName extends FIRRTLNameKind(value = CAPI.FIRRTL_NAME_KIND_INTERESTING_NAME())
}

sealed abstract class FIRRTLPortDir(val value: Int) extends ForeignType[Int] {
  private[circt] def get = value
  private[circt] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FIRRTLPortDir {
  final case object Input extends FIRRTLPortDir(value = CAPI.FIRRTL_PORT_DIR_INPUT())
  final case object Output extends FIRRTLPortDir(value = CAPI.FIRRTL_PORT_DIR_OUTPUT())
}

sealed abstract class firrtlAttrGetRUW(val value: Int) extends ForeignType[Int] {
  private[circt] def get = value
  private[circt] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object firrtlAttrGetRUW {
  final case object Undefined extends firrtlAttrGetRUW(value = CAPI.FIRRTL_RUW_UNDEFINED())
  final case object Old extends firrtlAttrGetRUW(value = CAPI.FIRRTL_RUW_OLD())
  final case object New extends firrtlAttrGetRUW(value = CAPI.FIRRTL_RUW_NEW())
}

sealed abstract class FIRRTLMemDir(val value: Int) extends ForeignType[Int] {
  private[circt] def get = value
  private[circt] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FIRRTLMemDir {
  final case object Infer extends FIRRTLMemDir(value = CAPI.FIRRTL_MEM_DIR_INFER())
  final case object Read extends FIRRTLMemDir(value = CAPI.FIRRTL_MEM_DIR_READ())
  final case object Write extends FIRRTLMemDir(value = CAPI.FIRRTL_MEM_DIR_WRITE())
  final case object ReadWrite extends FIRRTLMemDir(value = CAPI.FIRRTL_MEM_DIR_READ_WRITE())
}

sealed class FirtoolPreserveAggregateMode(val value: Int) extends ForeignType[Int] {
  private[circt] def get = value
  private[circt] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FirtoolPreserveAggregateMode {
  final case object None extends FirtoolPreserveAggregateMode(value = CAPI.FIRTOOL_PRESERVE_AGGREGATE_MODE_NONE())
  final case object OneDimVec
      extends FirtoolPreserveAggregateMode(value = CAPI.FIRTOOL_PRESERVE_AGGREGATE_MODE_ONE_DIM_VEC())
  final case object Vec extends FirtoolPreserveAggregateMode(value = CAPI.FIRTOOL_PRESERVE_AGGREGATE_MODE_VEC())
  final case object All extends FirtoolPreserveAggregateMode(value = CAPI.FIRTOOL_PRESERVE_AGGREGATE_MODE_ALL())
}

sealed class FirtoolPreserveValuesMode(val value: Int) extends ForeignType[Int] {
  private[circt] def get = value
  private[circt] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FirtoolPreserveValuesMode {
  final case object None extends FirtoolPreserveValuesMode(value = CAPI.FIRTOOL_PRESERVE_VALUES_MODE_NONE())
  final case object Named extends FirtoolPreserveValuesMode(value = CAPI.FIRTOOL_PRESERVE_VALUES_MODE_NAMED())
  final case object All extends FirtoolPreserveValuesMode(value = CAPI.FIRTOOL_PRESERVE_VALUES_MODE_ALL())
}

sealed class FirtoolBuildMode(val value: Int) extends ForeignType[Int] {
  private[circt] def get = value
  private[circt] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FirtoolBuildMode {
  final case object Debug extends FirtoolBuildMode(value = CAPI.FIRTOOL_BUILD_MODE_DEBUG())
  final case object Release extends FirtoolBuildMode(value = CAPI.FIRTOOL_BUILD_MODE_RELEASE())
}

sealed class FirtoolRandomKind(val value: Int) extends ForeignType[Int] {
  private[circt] def get = value
  private[circt] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FirtoolRandomKind {
  final case object None extends FirtoolRandomKind(value = CAPI.FIRTOOL_RANDOM_KIND_NONE())
  final case object Mem extends FirtoolRandomKind(value = CAPI.FIRTOOL_RANDOM_KIND_MEM())
  final case object Reg extends FirtoolRandomKind(value = CAPI.FIRTOOL_RANDOM_KIND_REG())
  final case object All extends FirtoolRandomKind(value = CAPI.FIRTOOL_RANDOM_KIND_ALL())
}

sealed class FirtoolCompanionMode(val value: Int) extends ForeignType[Int] {
  private[circt] def get = value
  private[circt] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FirtoolCompanionMode {
  final case object Bind extends FirtoolCompanionMode(value = CAPI.FIRTOOL_COMPANION_MODE_BIND())
  final case object Instantiate extends FirtoolCompanionMode(value = CAPI.FIRTOOL_COMPANION_MODE_INSTANTIATE())
  final case object Drop extends FirtoolCompanionMode(value = CAPI.FIRTOOL_COMPANION_MODE_DROP())
}
