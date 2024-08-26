// SPDX-License-Identifier: Apache-2.0

package chisel3.panamalib

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
      CAPI.mlirGetDialectHandle__seq__(arena),
      CAPI.mlirGetDialectHandle__emit__(arena)
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

  private def stringRefFromBytes(bytes: Array[Byte]): MlirStringRef = {
    val buffer = arena.allocate(bytes.length)
    buffer.copyFrom(MemorySegment.ofArray(bytes))

    val stringRef = circt.MlirStringRef.allocate(arena)
    circt.MlirStringRef.data$set(stringRef, buffer)
    circt.MlirStringRef.length$set(stringRef, bytes.length)

    MlirStringRef(stringRef)
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

  def mlirModuleCreateParse(module: String) = MlirModule(
    CAPI.mlirModuleCreateParse(arena, mlirCtx, newString(module).get)
  )
  def mlirModuleCreateParseBytes(module: Array[Byte]) = MlirModule(
    CAPI.mlirModuleCreateParse(arena, mlirCtx, stringRefFromBytes(module).get)
  )

  def mlirModuleFromOperation(op: MlirOperation) = MlirModule(CAPI.mlirModuleFromOperation(arena, op.get))

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

  def mlirAttributeDump(attr: MlirAttribute) = CAPI.mlirAttributeDump(attr.get)

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

  def mlirOperationStateEnableResultTypeInference(state: MlirOperationState): Unit =
    CAPI.mlirOperationStateEnableResultTypeInference(state.get)

  def mlirOperationCreate(state: MlirOperationState) = MlirOperation(CAPI.mlirOperationCreate(arena, state.get))

  def mlirOperationGetResult(operation: MlirOperation, pos: Long) = MlirValue(
    CAPI.mlirOperationGetResult(arena, operation.get, pos)
  )

  def mlirOperationGetAttributeByName(op: MlirOperation, name: String) = MlirAttribute(
    CAPI.mlirOperationGetAttributeByName(arena, op.get, newString(name).get)
  )

  def mlirOperationSetInherentAttributeByName(op: MlirOperation, name: String, attr: MlirAttribute): Unit =
    CAPI.mlirOperationSetInherentAttributeByName(op.get, newString(name).get, attr.get)

  def mlirBlockCreate(args: Seq[MlirType], locs: Seq[MlirLocation]): MlirBlock = {
    assert(args.length == locs.length)
    val length = args.length
    MlirBlock(CAPI.mlirBlockCreate(arena, length, seqToArray(args)._1, seqToArray(locs)._1))
  }

  def mlirBlockGetArgument(block: MlirBlock, pos: Long) = MlirValue(CAPI.mlirBlockGetArgument(arena, block.get, pos))

  def mlirBlockGetFirstOperation(block: MlirBlock) = MlirOperation(CAPI.mlirBlockGetFirstOperation(arena, block.get))

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

  def mlirArrayAttrGetNumElements(attr: MlirAttribute): Long = CAPI.mlirArrayAttrGetNumElements(attr.get)

  def mlirArrayAttrGetElement(attr: MlirAttribute, pos: Long): MlirAttribute = MlirAttribute(
    CAPI.mlirArrayAttrGetElement(arena, attr.get, pos)
  )

  def mlirTypeAttrGet(tpe: MlirType) = MlirAttribute(CAPI.mlirTypeAttrGet(arena, tpe.get))

  def mlirBoolAttrGet(value: Boolean) = MlirAttribute(CAPI.mlirBoolAttrGet(arena, mlirCtx, if (value) 1 else 0))

  def mlirStringAttrGet(string: String) = MlirAttribute(CAPI.mlirStringAttrGet(arena, mlirCtx, newString(string).get))

  def mlirStringAttrGetValue(attr: MlirAttribute): String = {
    val string = CAPI.mlirStringAttrGetValue(arena, attr.get)
    MlirStringRef(string).toString
  }

  def mlirAttributeIsAInteger(attr: MlirAttribute): Boolean = CAPI.mlirAttributeIsAInteger(attr.get)

  def mlirAttributeIsAString(attr: MlirAttribute): Boolean = CAPI.mlirAttributeIsAString(attr.get)

  def mlirIntegerAttrGet(tpe: MlirType, value: Long) = MlirAttribute(CAPI.mlirIntegerAttrGet(arena, tpe.get, value))

  def mlirIntegerAttrGetValueInt(attr: MlirAttribute): Long = CAPI.mlirIntegerAttrGetValueInt(attr.get)

  def mlirIntegerAttrGetValueSInt(attr: MlirAttribute): Long = CAPI.mlirIntegerAttrGetValueSInt(attr.get)

  def mlirIntegerAttrGetValueUInt(attr: MlirAttribute): Long = CAPI.mlirIntegerAttrGetValueUInt(attr.get)

  def mlirFloatAttrDoubleGet(tpe: MlirType, value: Double) = MlirAttribute(
    CAPI.mlirFloatAttrDoubleGet(arena, mlirCtx, tpe.get, value)
  )

  def mlirFlatSymbolRefAttrGet(symbol: String) = MlirAttribute(
    CAPI.mlirFlatSymbolRefAttrGet(arena, mlirCtx, newString(symbol).get)
  )

  def mlirValueGetType(tpe: MlirValue) = MlirType(CAPI.mlirValueGetType(arena, tpe.get))

  def mlirNoneTypeGet() = MlirType(CAPI.mlirNoneTypeGet(arena, mlirCtx))

  def mlirIntegerTypeGet(bitwidth: Int) = MlirType(CAPI.mlirIntegerTypeGet(arena, mlirCtx, bitwidth))

  def mlirIntegerTypeUnsignedGet(bitwidth: Int) = MlirType(CAPI.mlirIntegerTypeUnsignedGet(arena, mlirCtx, bitwidth))

  def mlirIntegerTypeSignedGet(bitwidth: Int) = MlirType(CAPI.mlirIntegerTypeSignedGet(arena, mlirCtx, bitwidth))

  def mlirF64TypeGet() = MlirType(CAPI.mlirF64TypeGet(arena, mlirCtx))

  def mlirOperationPrint(op: MlirOperation, callback: String => Unit) =
    CAPI.mlirOperationPrint(op.get, newStringCallback(callback).get, NULL)

  def mlirOperationWriteBytecode(op: MlirOperation, callback: Array[Byte] => Unit) = {
    val cb = new circt.MlirStringCallback {
      def apply(message: MemorySegment, userData: MemorySegment) = {
        callback(MlirStringRef(message).toBytes)
      }
    }
    val mlirCallback = MlirStringCallback(circt.MlirStringCallback.allocate(cb, arena))
    CAPI.mlirOperationWriteBytecode(op.get, mlirCallback.get, NULL)
  }

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

  def circtFirtoolOptionsCreateDefault() = CirctFirtoolFirtoolOptions(CAPI.circtFirtoolOptionsCreateDefault(arena))
  def circtFirtoolOptionsDestroy(options:           CirctFirtoolFirtoolOptions) = CAPI.circtFirtoolOptionsDestroy(options.get)
  def circtFirtoolOptionsSetOutputFilename(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetOutputFilename(options.get, newString(value).get)
  def circtFirtoolOptionsSetDisableUnknownAnnotations(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetDisableUnknownAnnotations(options.get, value)
  def circtFirtoolOptionsSetDisableAnnotationsClassless(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetDisableAnnotationsClassless(options.get, value)
  def circtFirtoolOptionsSetLowerAnnotationsNoRefTypePorts(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetLowerAnnotationsNoRefTypePorts(options.get, value)
  def circtFirtoolOptionsSetPreserveAggregate(
    options: CirctFirtoolFirtoolOptions,
    value:   CirctFirtoolPreserveAggregateMode
  ) = CAPI.circtFirtoolOptionsSetPreserveAggregate(options.get, value.get)
  def circtFirtoolOptionsSetPreserveValues(options: CirctFirtoolFirtoolOptions, value: CirctFirtoolPreserveValuesMode) =
    CAPI.circtFirtoolOptionsSetPreserveValues(options.get, value.get)
  def circtFirtoolOptionsSetBuildMode(options: CirctFirtoolFirtoolOptions, value: CirctFirtoolBuildMode) =
    CAPI.circtFirtoolOptionsSetBuildMode(options.get, value.get)
  def circtFirtoolOptionsSetDisableOptimization(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetDisableOptimization(options.get, value)
  def circtFirtoolOptionsSetExportChiselInterface(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetExportChiselInterface(options.get, value)
  def circtFirtoolOptionsSetChiselInterfaceOutDirectory(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetChiselInterfaceOutDirectory(options.get, newString(value).get)
  def circtFirtoolOptionsSetVbToBv(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetVbToBv(options.get, value)
  def circtFirtoolOptionsSetNoDedup(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetNoDedup(options.get, value)
  def circtFirtoolOptionsSetCompanionMode(options: CirctFirtoolFirtoolOptions, value: CirctFirtoolCompanionMode) =
    CAPI.circtFirtoolOptionsSetCompanionMode(options.get, value.get)
  def circtFirtoolOptionsSetDisableAggressiveMergeConnections(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetDisableAggressiveMergeConnections(options.get, value)
  def circtFirtoolOptionsSetEmitOmir(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetEmitOmir(options.get, value)
  def circtFirtoolOptionsSetOmirOutFile(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetOmirOutFile(options.get, newString(value).get)
  def circtFirtoolOptionsSetLowerMemories(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetLowerMemories(options.get, value)
  def circtFirtoolOptionsSetBlackBoxRootPath(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetBlackBoxRootPath(options.get, newString(value).get)
  def circtFirtoolOptionsSetReplSeqMem(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetReplSeqMem(options.get, value)
  def circtFirtoolOptionsSetReplSeqMemFile(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetReplSeqMemFile(options.get, newString(value).get)
  def circtFirtoolOptionsSetExtractTestCode(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetExtractTestCode(options.get, value)
  def circtFirtoolOptionsSetIgnoreReadEnableMem(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetIgnoreReadEnableMem(options.get, value)
  def circtFirtoolOptionsSetDisableRandom(options: CirctFirtoolFirtoolOptions, value: CirctFirtoolRandomKind) =
    CAPI.circtFirtoolOptionsSetDisableRandom(options.get, value.get)
  def circtFirtoolOptionsSetOutputAnnotationFilename(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetOutputAnnotationFilename(options.get, newString(value).get)
  def circtFirtoolOptionsSetEnableAnnotationWarning(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetEnableAnnotationWarning(options.get, value)
  def circtFirtoolOptionsSetAddMuxPragmas(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetAddMuxPragmas(options.get, value)
  def circtFirtoolOptionsSetVerificationFlavor(
    options: CirctFirtoolFirtoolOptions,
    value:   CirctFirtoolVerificationFlavor
  ) =
    CAPI.circtFirtoolOptionsSetVerificationFlavor(options.get, value.get)
  def circtFirtoolOptionsSetEmitSeparateAlwaysBlocks(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetEmitSeparateAlwaysBlocks(options.get, value)
  def circtFirtoolOptionsSetEtcDisableInstanceExtraction(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetEtcDisableInstanceExtraction(options.get, value)
  def circtFirtoolOptionsSetEtcDisableRegisterExtraction(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetEtcDisableRegisterExtraction(options.get, value)
  def circtFirtoolOptionsSetEtcDisableModuleInlining(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetEtcDisableModuleInlining(options.get, value)
  def circtFirtoolOptionsSetAddVivadoRAMAddressConflictSynthesisBugWorkaround(
    options: CirctFirtoolFirtoolOptions,
    value:   Boolean
  ) = CAPI.circtFirtoolOptionsSetAddVivadoRAMAddressConflictSynthesisBugWorkaround(options.get, value)
  def circtFirtoolOptionsSetCkgModuleName(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetCkgModuleName(options.get, newString(value).get)
  def circtFirtoolOptionsSetCkgInputName(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetCkgInputName(options.get, newString(value).get)
  def circtFirtoolOptionsSetCkgOutputName(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetCkgOutputName(options.get, newString(value).get)
  def circtFirtoolOptionsSetCkgEnableName(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetCkgEnableName(options.get, newString(value).get)
  def circtFirtoolOptionsSetCkgTestEnableName(options: CirctFirtoolFirtoolOptions, value: String) =
    CAPI.circtFirtoolOptionsSetCkgTestEnableName(options.get, newString(value).get)
  def circtFirtoolOptionsSetExportModuleHierarchy(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetExportModuleHierarchy(options.get, value)
  def circtFirtoolOptionsSetStripFirDebugInfo(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetStripFirDebugInfo(options.get, value)
  def circtFirtoolOptionsSetStripDebugInfo(options: CirctFirtoolFirtoolOptions, value: Boolean) =
    CAPI.circtFirtoolOptionsSetStripDebugInfo(options.get, value)

  def circtFirtoolPopulatePreprocessTransforms(pm: MlirPassManager, options: CirctFirtoolFirtoolOptions) =
    MlirLogicalResult(
      CAPI.circtFirtoolPopulatePreprocessTransforms(arena, pm.get, options.get)
    )

  def circtFirtoolPopulateCHIRRTLToLowFIRRTL(
    pm:            MlirPassManager,
    options:       CirctFirtoolFirtoolOptions,
    module:        MlirModule,
    inputFilename: String
  ) = MlirLogicalResult(
    CAPI.circtFirtoolPopulateCHIRRTLToLowFIRRTL(arena, pm.get, options.get, newString(inputFilename).get)
  )

  def circtFirtoolPopulateLowFIRRTLToHW(pm: MlirPassManager, options: CirctFirtoolFirtoolOptions) = MlirLogicalResult(
    CAPI.circtFirtoolPopulateLowFIRRTLToHW(arena, pm.get, options.get)
  )

  def circtFirtoolPopulateHWToSV(pm: MlirPassManager, options: CirctFirtoolFirtoolOptions) = MlirLogicalResult(
    CAPI.circtFirtoolPopulateHWToSV(arena, pm.get, options.get)
  )

  def circtFirtoolPopulateExportVerilog(
    pm:       MlirPassManager,
    options:  CirctFirtoolFirtoolOptions,
    callback: String => Unit
  ) = MlirLogicalResult(
    CAPI.circtFirtoolPopulateExportVerilog(arena, pm.get, options.get, newStringCallback(callback).get, NULL)
  )

  def circtFirtoolPopulateExportSplitVerilog(
    pm:        MlirPassManager,
    options:   CirctFirtoolFirtoolOptions,
    directory: String
  ) = MlirLogicalResult(
    CAPI.circtFirtoolPopulateExportSplitVerilog(arena, pm.get, options.get, newString(directory).get)
  )

  def circtFirtoolPopulateFinalizeIR(pm: MlirPassManager, options: CirctFirtoolFirtoolOptions) =
    MlirLogicalResult(CAPI.circtFirtoolPopulateFinalizeIR(arena, pm.get, options.get))

  def mlirLogicalResultIsSuccess(res: MlirLogicalResult): Boolean = circt.MlirLogicalResult.value$get(res.get) != 0

  def mlirLogicalResultIsFailure(res: MlirLogicalResult): Boolean = circt.MlirLogicalResult.value$get(res.get) == 0

  def firrtlTypeGetUInt(width: Int) = MlirType(CAPI.firrtlTypeGetUInt(arena, mlirCtx, width))

  def firrtlTypeGetSInt(width: Int) = MlirType(CAPI.firrtlTypeGetSInt(arena, mlirCtx, width))

  def firrtlTypeGetClock() = MlirType(CAPI.firrtlTypeGetClock(arena, mlirCtx))

  def firrtlTypeGetReset() = MlirType(CAPI.firrtlTypeGetReset(arena, mlirCtx))

  def firrtlTypeGetAsyncReset() = MlirType(CAPI.firrtlTypeGetAsyncReset(arena, mlirCtx))

  def firrtlTypeGetAnalog(width: Int) = MlirType(CAPI.firrtlTypeGetAnalog(arena, mlirCtx, width))

  def firrtlTypeGetVector(element: MlirType, count: Long) = MlirType(
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

  def firrtlTypeIsAOpenBundle(tpe: MlirType): Boolean = CAPI.firrtlTypeIsAOpenBundle(tpe.get)

  def firrtlTypeGetBundleFieldIndex(tpe: MlirType, fieldName: String): Int =
    CAPI.firrtlTypeGetBundleFieldIndex(tpe.get, newString(fieldName).get)

  def firrtlTypeGetRef(target: MlirType, forceable: Boolean) = MlirType(
    CAPI.firrtlTypeGetRef(arena, target.get, forceable)
  )

  def firrtlTypeGetAnyRef() = MlirType(CAPI.firrtlTypeGetAnyRef(arena, mlirCtx))

  def firrtlTypeGetInteger() = MlirType(CAPI.firrtlTypeGetInteger(arena, mlirCtx))

  def firrtlTypeGetDouble() = MlirType(CAPI.firrtlTypeGetDouble(arena, mlirCtx))

  def firrtlTypeGetString() = MlirType(CAPI.firrtlTypeGetString(arena, mlirCtx))

  def firrtlTypeGetBoolean() = MlirType(CAPI.firrtlTypeGetBoolean(arena, mlirCtx))

  def firrtlTypeGetPath() = MlirType(CAPI.firrtlTypeGetPath(arena, mlirCtx))

  def firrtlTypeGetList(elementType: MlirType) = MlirType(CAPI.firrtlTypeGetList(arena, mlirCtx, elementType.get))

  def firrtlTypeGetClass(name: MlirAttribute /* FlatSymbolRefAttr */, elements: Seq[FIRRTLClassElement]): MlirType = {
    val buffer = circt.FIRRTLClassElement.allocateArray(elements.length, arena)
    elements.zipWithIndex.foreach {
      case (element, i) =>
        val elementBuffer = buffer.asSlice(circt.FIRRTLClassElement.sizeof() * i, circt.FIRRTLClassElement.sizeof())
        circt.FIRRTLClassElement.name$slice(elementBuffer).copyFrom(mlirIdentifierGet(element.name).get)
        circt.FIRRTLClassElement.type$slice(elementBuffer).copyFrom(element.tpe.get)
        circt.FIRRTLClassElement.direction$set(elementBuffer, element.direction.get)
    }
    MlirType(CAPI.firrtlTypeGetClass(arena, mlirCtx, name.get, elements.length, buffer))
  }

  def firrtlTypeGetMaskType(tpe: MlirType) = MlirType(CAPI.firrtlTypeGetMaskType(arena, tpe.get))

  def firrtlAttrGetPortDirs(dirs: Seq[FIRRTLDirection]): MlirAttribute = {
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

  def firrtlAttrGetRUW(ruw: FIRRTLRUW) = MlirAttribute(CAPI.firrtlAttrGetRUW(arena, mlirCtx, ruw.value))

  def firrtlAttrGetMemDir(dir: FIRRTLMemDir) = MlirAttribute(CAPI.firrtlAttrGetMemDir(arena, mlirCtx, dir.value))

  def firrtlAttrGetIntegerFromString(tpe: MlirType, numBits: Int, str: String, radix: Byte) = MlirAttribute(
    CAPI.firrtlAttrGetIntegerFromString(arena, tpe.get, numBits, newString(str).get, radix)
  )

  def firrtlValueFoldFlow(value: MlirValue, flow: Int): Int = CAPI.firrtlValueFoldFlow(value.get, flow)

  def firrtlImportAnnotationsFromJSONRaw(annotationsStr: String): Option[MlirAttribute] = {
    val attr = circt.MlirAttribute.allocate(arena)
    val result = CAPI.firrtlImportAnnotationsFromJSONRaw(mlirCtx, newString(annotationsStr).get, attr)
    if (result) {
      Some(MlirAttribute(attr))
    } else {
      None
    }
  }

  def chirrtlTypeGetCMemory(elementType: MlirType, numElements: Long) = MlirType(
    CAPI.chirrtlTypeGetCMemory(arena, mlirCtx, elementType.get, numElements)
  )

  def chirrtlTypeGetCMemoryPort() = MlirType(CAPI.chirrtlTypeGetCMemoryPort(arena, mlirCtx))

  def hwInnerRefAttrGet(moduleName: String, innerSym: String) =
    MlirAttribute(CAPI.hwInnerRefAttrGet(arena, mlirStringAttrGet(moduleName).get, mlirStringAttrGet(innerSym).get))

  def hwInnerSymAttrGet(symName: String) =
    MlirAttribute(CAPI.hwInnerSymAttrGet(arena, mlirStringAttrGet(symName).get))

  def hwInnerSymAttrGetEmpty() = MlirAttribute(CAPI.hwInnerSymAttrGetEmpty(arena, mlirCtx))

  def hwInstanceGraphGet(operation: MlirOperation) = HWInstanceGraph(CAPI.hwInstanceGraphGet(arena, operation.get))

  def hwInstanceGraphGetTopLevelNode(instanceGraph: HWInstanceGraph) = HWInstanceGraphNode(
    CAPI.hwInstanceGraphGetTopLevelNode(arena, instanceGraph.get)
  )

  def hwInstanceGraphForEachNode(instaceGraph: HWInstanceGraph, callback: HWInstanceGraphNode => Unit) = {
    val cb = HWInstanceGraphNodeCallback(
      circt.HWInstanceGraphNodeCallback.allocate(
        new circt.HWInstanceGraphNodeCallback {
          def apply(node: MemorySegment, userData: MemorySegment) = {
            callback(HWInstanceGraphNode(node))
          }
        },
        arena
      )
    )
    CAPI.hwInstanceGraphForEachNode(instaceGraph.get, cb.get, NULL)
  }

  def hwInstanceGraphNodeEqual(lhs: HWInstanceGraphNode, rhs: HWInstanceGraphNode) =
    CAPI.hwInstanceGraphNodeEqual(lhs.get, rhs.get)

  def hwInstanceGraphNodeGetModuleOp(node: HWInstanceGraphNode) = MlirOperation(
    CAPI.hwInstanceGraphNodeGetModuleOp(arena, node.get)
  )

  //
  // OM C-API
  //

  def omTypeIsAClassType(tpe:          MlirType): Boolean = CAPI.omTypeIsAClassType(tpe.get)
  def omClassTypeGetName(tpe:          MlirType) = MlirIdentifier(CAPI.omClassTypeGetName(arena, tpe.get))
  def omTypeIsAFrozenBasePathType(tpe: MlirType): Boolean = CAPI.omTypeIsAFrozenBasePathType(tpe.get)
  def omTypeIsAFrozenPathType(tpe:     MlirType): Boolean = CAPI.omTypeIsAFrozenPathType(tpe.get)
  def omTypeIsAMapType(tpe:            MlirType): Boolean = CAPI.omTypeIsAMapType(tpe.get)
  def omMapTypeGetKeyType(tpe:         MlirType) = MlirType(CAPI.omMapTypeGetKeyType(arena, tpe.get))
  def omTypeIsAStringType(tpe:         MlirType): Boolean = CAPI.omTypeIsAStringType(tpe.get)
  def omEvaluatorNew(mod:              MlirModule) = OMEvaluator(CAPI.omEvaluatorNew(arena, mod.get))
  def omEvaluatorInstantiate(evaluator: OMEvaluator, className: String, actualParams: Seq[OMEvaluatorValue]) = {
    val params = seqToArray(actualParams);
    OMEvaluatorValue(
      CAPI.omEvaluatorInstantiate(arena, evaluator.get, mlirStringAttrGet(className).get, params._2, params._1)
    )
  }
  def omEvaluatorGetModule(evaluator: OMEvaluator) = MlirModule(CAPI.omEvaluatorGetModule(arena, evaluator.get))
  def omEvaluatorObjectIsNull(obj:    OMEvaluatorValue):      Boolean = CAPI.omEvaluatorObjectIsNull(obj.get)
  def omEvaluatorObjectGetType(obj:   OMEvaluatorValue) = MlirType(CAPI.omEvaluatorObjectGetType(arena, obj.get))
  def omEvaluatorObjectGetField(obj:  OMEvaluatorValue, name: String) = OMEvaluatorValue(
    CAPI.omEvaluatorObjectGetField(arena, obj.get, mlirStringAttrGet(name).get)
  )
  def omEvaluatorObjectGetHash(obj: OMEvaluatorValue): Int = CAPI.omEvaluatorObjectGetHash(obj.get)
  def omEvaluatorObjectIsEq(obj:    OMEvaluatorValue, other: OMEvaluatorValue): Boolean =
    CAPI.omEvaluatorObjectIsEq(obj.get, other.get)
  def omEvaluatorObjectGetFieldNames(obj: OMEvaluatorValue) = MlirAttribute(
    CAPI.omEvaluatorObjectGetFieldNames(arena, obj.get)
  )
  def omEvaluatorValueGetLoc(evaluatorValue: OMEvaluatorValue) = MlirLocation(
    CAPI.omEvaluatorValueGetLoc(arena, evaluatorValue.get)
  )
  def omEvaluatorValueIsNull(evaluatorValue: OMEvaluatorValue): Boolean =
    CAPI.omEvaluatorValueIsNull(evaluatorValue.get)
  def omEvaluatorValueIsAObject(evaluatorValue: OMEvaluatorValue): Boolean =
    CAPI.omEvaluatorValueIsAObject(evaluatorValue.get)
  def omEvaluatorValueIsAPrimitive(evaluatorValue: OMEvaluatorValue): Boolean =
    CAPI.omEvaluatorValueIsAPrimitive(evaluatorValue.get)
  def omEvaluatorValueGetPrimitive(evaluatorValue: OMEvaluatorValue) = MlirAttribute(
    CAPI.omEvaluatorValueGetPrimitive(arena, evaluatorValue.get)
  )
  def omEvaluatorValueFromPrimitive(primitive: MlirAttribute) = OMEvaluatorValue(
    CAPI.omEvaluatorValueFromPrimitive(arena, primitive.get)
  )
  def omEvaluatorValueIsAList(evaluatorValue: OMEvaluatorValue): Boolean =
    CAPI.omEvaluatorValueIsAList(evaluatorValue.get)
  def omEvaluatorListGetNumElements(evaluatorValue: OMEvaluatorValue): Long =
    CAPI.omEvaluatorListGetNumElements(evaluatorValue.get)
  def omEvaluatorListGetElement(evaluatorValue: OMEvaluatorValue, pos: Long) = OMEvaluatorValue(
    CAPI.omEvaluatorListGetElement(arena, evaluatorValue.get, pos)
  )
  def omEvaluatorValueIsATuple(evaluatorValue: OMEvaluatorValue): Boolean =
    CAPI.omEvaluatorValueIsATuple(evaluatorValue.get)
  def omEvaluatorTupleGetNumElements(evaluatorValue: OMEvaluatorValue): Long =
    CAPI.omEvaluatorTupleGetNumElements(evaluatorValue.get)
  def omEvaluatorTupleGetElement(evaluatorValue: OMEvaluatorValue, pos: Long) = OMEvaluatorValue(
    CAPI.omEvaluatorTupleGetElement(arena, evaluatorValue.get, pos)
  )
  def omEvaluatorMapGetElement(evaluatorValue: OMEvaluatorValue, attr: MlirAttribute) = OMEvaluatorValue(
    CAPI.omEvaluatorMapGetElement(arena, evaluatorValue.get, attr.get)
  )
  def omEvaluatorMapGetKeys(obj:             OMEvaluatorValue) = MlirAttribute(CAPI.omEvaluatorMapGetKeys(arena, obj.get))
  def omEvaluatorValueIsAMap(evaluatorValue: OMEvaluatorValue): Boolean =
    CAPI.omEvaluatorValueIsAMap(evaluatorValue.get)
  def omEvaluatorMapGetType(evaluatorValue: OMEvaluatorValue): MlirType = MlirType(
    CAPI.omEvaluatorMapGetType(arena, evaluatorValue.get)
  )
  def omEvaluatorValueIsABasePath(evaluatorValue: OMEvaluatorValue): Boolean =
    CAPI.omEvaluatorValueIsABasePath(evaluatorValue.get)
  def omEvaluatorBasePathGetEmpty() = OMEvaluatorValue(CAPI.omEvaluatorBasePathGetEmpty(arena, mlirCtx))
  def omEvaluatorValueIsAPath(evaluatorValue: OMEvaluatorValue): Boolean =
    CAPI.omEvaluatorValueIsAPath(evaluatorValue.get)
  def omEvaluatorPathGetAsString(evaluatorValue: OMEvaluatorValue): String = mlirStringAttrGetValue(
    MlirAttribute(CAPI.omEvaluatorPathGetAsString(arena, evaluatorValue.get))
  )
  def omEvaluatorValueIsAReference(evaluatorValue: OMEvaluatorValue): Boolean =
    CAPI.omEvaluatorValueIsAReference(evaluatorValue.get)
  def omEvaluatorValueGetReferenceValue(evaluatorValue: OMEvaluatorValue): OMEvaluatorValue =
    OMEvaluatorValue(CAPI.omEvaluatorValueGetReferenceValue(arena, evaluatorValue.get))
  def omAttrIsAReferenceAttr(attr:     MlirAttribute): Boolean = CAPI.omAttrIsAReferenceAttr(attr.get)
  def omReferenceAttrGetInnerRef(attr: MlirAttribute) = MlirAttribute(CAPI.omReferenceAttrGetInnerRef(arena, attr.get))
  def omAttrIsAIntegerAttr(attr:       MlirAttribute): Boolean = CAPI.omAttrIsAIntegerAttr(attr.get)
  def omIntegerAttrGetInt(attr:        MlirAttribute) = MlirAttribute(CAPI.omIntegerAttrGetInt(arena, attr.get))
  def omIntegerAttrGet(attr:           MlirAttribute) = MlirAttribute(CAPI.omIntegerAttrGet(arena, attr.get))
  def omAttrIsAListAttr(attr:          MlirAttribute): Boolean = CAPI.omAttrIsAListAttr(attr.get)
  def omListAttrGetNumElements(attr:   MlirAttribute): Long = CAPI.omListAttrGetNumElements(attr.get)
  def omListAttrGetElement(attr:       MlirAttribute, pos: Long) = MlirAttribute(
    CAPI.omListAttrGetElement(arena, attr.get, pos)
  )
  def omAttrIsAMapAttr(attr:        MlirAttribute): Boolean = CAPI.omAttrIsAMapAttr(attr.get)
  def omMapAttrGetNumElements(attr: MlirAttribute): Long = CAPI.omMapAttrGetNumElements(attr.get)
  def omMapAttrGetElementKey(attr:  MlirAttribute, pos: Long) = MlirIdentifier(
    CAPI.omMapAttrGetElementKey(arena, attr.get, pos)
  )
  def omMapAttrGetElementValue(attr: MlirAttribute, pos: Long) = MlirAttribute(
    CAPI.omMapAttrGetElementValue(arena, attr.get, pos)
  )
}

//
// MLIR & CIRCT Types
//
// Since all structs returned from Panama framework are `MemorySegment`, which is like a `void *` in C.
// We create these type wrappers to wrap it, make them type-safe.
//

trait ForeignType[T] {
  private[panamalib] def get: T
  private[panamalib] val sizeof: Int
}

final case class MlirAttribute(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirAttribute.sizeof().toInt
}
object MlirAttribute {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirAttribute(ptr)
}

final case class MlirNamedAttribute(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirNamedAttribute.sizeof().toInt
}
object MlirNamedAttribute {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirNamedAttribute(ptr)
}

final case class MlirBlock(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirBlock.sizeof().toInt
}
object MlirBlock {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirBlock(ptr)
}

final case class MlirRegion(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirRegion.sizeof().toInt
}
object MlirRegion {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirRegion(ptr)
}

final case class MlirIdentifier(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirIdentifier.sizeof().toInt
}
object MlirIdentifier {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirIdentifier(ptr)
}

final case class MlirLocation(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirLocation.sizeof().toInt
}
object MlirLocation {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirLocation(ptr)
}

final case class MlirModule(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirModule.sizeof().toInt
}
object MlirModule {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirModule(ptr)
}

final case class MlirOperation(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirOperation.sizeof().toInt
}
object MlirOperation {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirOperation(ptr)
}

final case class MlirOperationState(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirOperationState.sizeof().toInt
}
object MlirOperationState {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirOperationState(ptr)
}

final case class MlirType(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirType.sizeof().toInt
}
object MlirType {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirType(ptr)
}

final case class MlirValue(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirValue.sizeof().toInt
}
object MlirValue {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirValue(ptr)
}

final case class MlirStringRef(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirStringRef.sizeof().toInt

  def toBytes: Array[Byte] = {
    var slice = circt.MlirStringRef.data$get(ptr).asSlice(0, circt.MlirStringRef.length$get(ptr))
    slice.toArray(JAVA_BYTE)
  }

  override def toString: String = new String(toBytes)
}
object MlirStringRef {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirStringRef(ptr)
}

final case class MlirLogicalResult(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirLogicalResult.sizeof().toInt
}
object MlirLogicalResult {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirLogicalResult(ptr)
}

final case class MlirStringCallback(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = CAPI.C_POINTER.byteSize().toInt
}
object MlirStringCallback {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirStringCallback(ptr)
}

final case class MlirPassManager(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirPassManager.sizeof().toInt
}
object MlirPassManager {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirPassManager(ptr)
}

final case class MlirOpPassManager(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirOpPassManager.sizeof().toInt
}
object MlirOpPassManager {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirOpPassManager(ptr)
}

final case class MlirPass(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.MlirPass.sizeof().toInt
}
object MlirPass {
  private[panamalib] def apply(ptr: MemorySegment) = new MlirPass(ptr)
}

final case class FIRRTLBundleField(name: String, isFlip: Boolean, tpe: MlirType)

final case class FIRRTLClassElement(name: String, tpe: MlirType, direction: FIRRTLDirection)

final case class CirctFirtoolFirtoolOptions(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.CirctFirtoolFirtoolOptions.sizeof().toInt
}
object CirctFirtoolFirtoolOptions {
  private[panamalib] def apply(ptr: MemorySegment) = new CirctFirtoolFirtoolOptions(ptr)
}

final case class OMEvaluator(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.OMEvaluator.sizeof().toInt
}
object OMEvaluator {
  private[panamalib] def apply(ptr: MemorySegment) = new OMEvaluator(ptr)
}

final case class OMEvaluatorValue(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.OMEvaluatorValue.sizeof().toInt
}
object OMEvaluatorValue {
  private[panamalib] def apply(ptr: MemorySegment) = new OMEvaluatorValue(ptr)
}

final case class HWInstanceGraph(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.HWInstanceGraph.sizeof().toInt
}
object HWInstanceGraph {
  private[panamalib] def apply(ptr: MemorySegment) = new HWInstanceGraph(ptr)
}

final case class HWInstanceGraphNode(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = circt.HWInstanceGraphNode.sizeof().toInt
}
object HWInstanceGraphNode {
  private[panamalib] def apply(ptr: MemorySegment) = new HWInstanceGraphNode(ptr)
}

final case class HWInstanceGraphNodeCallback(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[panamalib] def get = ptr
  private[panamalib] val sizeof = CAPI.C_POINTER.byteSize().toInt
}
object HWInstanceGraphNodeCallback {
  private[panamalib] def apply(ptr: MemorySegment) = new HWInstanceGraphNodeCallback(ptr)
}

//
// MLIR & CIRCT Enums
//

sealed abstract class FIRRTLConvention(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FIRRTLConvention {
  final case object Internal extends FIRRTLConvention(value = CAPI.FIRRTL_CONVENTION_INTERNAL())
  final case object Scalarized extends FIRRTLConvention(value = CAPI.FIRRTL_CONVENTION_SCALARIZED())
}

sealed abstract class FIRRTLNameKind(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FIRRTLNameKind {
  final case object DroppableName extends FIRRTLNameKind(value = CAPI.FIRRTL_NAME_KIND_DROPPABLE_NAME())
  final case object InterestingName extends FIRRTLNameKind(value = CAPI.FIRRTL_NAME_KIND_INTERESTING_NAME())
}

sealed abstract class FIRRTLDirection(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FIRRTLDirection {
  final case object In extends FIRRTLDirection(value = CAPI.FIRRTL_DIRECTION_IN())
  final case object Out extends FIRRTLDirection(value = CAPI.FIRRTL_DIRECTION_OUT())
}

sealed abstract class FIRRTLRUW(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FIRRTLRUW {
  final case object Undefined extends FIRRTLRUW(value = CAPI.FIRRTL_RUW_UNDEFINED())
  final case object Old extends FIRRTLRUW(value = CAPI.FIRRTL_RUW_OLD())
  final case object New extends FIRRTLRUW(value = CAPI.FIRRTL_RUW_NEW())
}

sealed abstract class FIRRTLMemDir(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object FIRRTLMemDir {
  final case object Infer extends FIRRTLMemDir(value = CAPI.FIRRTL_MEM_DIR_INFER())
  final case object Read extends FIRRTLMemDir(value = CAPI.FIRRTL_MEM_DIR_READ())
  final case object Write extends FIRRTLMemDir(value = CAPI.FIRRTL_MEM_DIR_WRITE())
  final case object ReadWrite extends FIRRTLMemDir(value = CAPI.FIRRTL_MEM_DIR_READ_WRITE())
}

sealed class CirctFirtoolPreserveAggregateMode(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object CirctFirtoolPreserveAggregateMode {
  final case object None
      extends CirctFirtoolPreserveAggregateMode(value = CAPI.CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_NONE())
  final case object OneDimVec
      extends CirctFirtoolPreserveAggregateMode(value = CAPI.CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_ONE_DIM_VEC())
  final case object Vec
      extends CirctFirtoolPreserveAggregateMode(value = CAPI.CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_VEC())
  final case object All
      extends CirctFirtoolPreserveAggregateMode(value = CAPI.CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_ALL())
}

sealed class CirctFirtoolPreserveValuesMode(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object CirctFirtoolPreserveValuesMode {
  final case object Strip
      extends CirctFirtoolPreserveValuesMode(value = CAPI.CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_STRIP())
  final case object None extends CirctFirtoolPreserveValuesMode(value = CAPI.CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_NONE())
  final case object Named
      extends CirctFirtoolPreserveValuesMode(value = CAPI.CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_NAMED())
  final case object All extends CirctFirtoolPreserveValuesMode(value = CAPI.CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_ALL())
}

sealed class CirctFirtoolBuildMode(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object CirctFirtoolBuildMode {
  final case object Default extends CirctFirtoolBuildMode(value = CAPI.CIRCT_FIRTOOL_BUILD_MODE_DEFAULT())
  final case object Debug extends CirctFirtoolBuildMode(value = CAPI.CIRCT_FIRTOOL_BUILD_MODE_DEBUG())
  final case object Release extends CirctFirtoolBuildMode(value = CAPI.CIRCT_FIRTOOL_BUILD_MODE_RELEASE())
}

sealed class CirctFirtoolRandomKind(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object CirctFirtoolRandomKind {
  final case object None extends CirctFirtoolRandomKind(value = CAPI.CIRCT_FIRTOOL_RANDOM_KIND_NONE())
  final case object Mem extends CirctFirtoolRandomKind(value = CAPI.CIRCT_FIRTOOL_RANDOM_KIND_MEM())
  final case object Reg extends CirctFirtoolRandomKind(value = CAPI.CIRCT_FIRTOOL_RANDOM_KIND_REG())
  final case object All extends CirctFirtoolRandomKind(value = CAPI.CIRCT_FIRTOOL_RANDOM_KIND_ALL())
}

sealed class CirctFirtoolCompanionMode(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object CirctFirtoolCompanionMode {
  final case object Bind extends CirctFirtoolCompanionMode(value = CAPI.CIRCT_FIRTOOL_COMPANION_MODE_BIND())
  final case object Instantiate
      extends CirctFirtoolCompanionMode(value = CAPI.CIRCT_FIRTOOL_COMPANION_MODE_INSTANTIATE())
  final case object Drop extends CirctFirtoolCompanionMode(value = CAPI.CIRCT_FIRTOOL_COMPANION_MODE_DROP())
}

sealed class CirctFirtoolVerificationFlavor(val value: Int) extends ForeignType[Int] {
  private[panamalib] def get = value
  private[panamalib] val sizeof = 4 // FIXME: jextract doesn't export type for C enum
}
object CirctFirtoolVerificationFlavor {
  final case object None extends CirctFirtoolVerificationFlavor(value = CAPI.CIRCT_FIRTOOL_VERIFICATION_FLAVOR_NONE())
  final case object IfElseFatal
      extends CirctFirtoolVerificationFlavor(value = CAPI.CIRCT_FIRTOOL_VERIFICATION_FLAVOR_IF_ELSE_FATAL())
  final case object Immediate
      extends CirctFirtoolVerificationFlavor(value = CAPI.CIRCT_FIRTOOL_VERIFICATION_FLAVOR_IMMEDIATE())
  final case object Sva extends CirctFirtoolVerificationFlavor(value = CAPI.CIRCT_FIRTOOL_VERIFICATION_FLAVOR_SVA())
}
