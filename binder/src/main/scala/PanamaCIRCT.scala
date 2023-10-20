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

  def firtoolGeneralOptionsCreateDefault() = FirtoolGeneralOptions(CAPI.firtoolGeneralOptionsCreateDefault(arena))
  def firtoolGeneralOptionsDestroy(options: FirtoolGeneralOptions) = CAPI.firtoolGeneralOptionsDestroy(options.get)
  def firtoolGeneralOptionsSetDisableOptimization(options: FirtoolGeneralOptions, value: Boolean) = CAPI.firtoolGeneralOptionsSetDisableOptimization(options.get, value)
  def firtoolGeneralOptionsGetDisableOptimization(options: FirtoolGeneralOptions): Boolean = CAPI.firtoolGeneralOptionsGetDisableOptimization(options.get)
  def firtoolGeneralOptionsSetReplSeqMem(options: FirtoolGeneralOptions, value: Boolean) = CAPI.firtoolGeneralOptionsSetReplSeqMem(options.get, value)
  def firtoolGeneralOptionsGetReplSeqMem(options: FirtoolGeneralOptions): Boolean = CAPI.firtoolGeneralOptionsGetReplSeqMem(options.get)
  def firtoolGeneralOptionsSetReplSeqMemFile(options: FirtoolGeneralOptions, value: String) = CAPI.firtoolGeneralOptionsSetReplSeqMemFile(options.get, newString(value).get)
  def firtoolGeneralOptionsGetReplSeqMemFile(options: FirtoolGeneralOptions): String = CAPI.firtoolGeneralOptionsGetReplSeqMemFile(arena, options.get).toString
  def firtoolGeneralOptionsSetIgnoreReadEnableMem(options: FirtoolGeneralOptions, value: Boolean) = CAPI.firtoolGeneralOptionsSetIgnoreReadEnableMem(options.get, value)
  def firtoolGeneralOptionsGetIgnoreReadEnableMem(options: FirtoolGeneralOptions): Boolean = CAPI.firtoolGeneralOptionsGetIgnoreReadEnableMem(options.get)
  def firtoolGeneralOptionsSetDisableRandom(options: FirtoolGeneralOptions, value: FirtoolRandomKind) = CAPI.firtoolGeneralOptionsSetDisableRandom(options.get, value.get)
  def firtoolGeneralOptionsGetDisableRandom(options: FirtoolGeneralOptions) = new FirtoolRandomKind(CAPI.firtoolGeneralOptionsGetDisableRandom(options.get))
  def firtoolPreprocessTransformsOptionsCreateDefault(general: FirtoolGeneralOptions) = FirtoolPreprocessTransformsOptions(CAPI.firtoolPreprocessTransformsOptionsCreateDefault(arena, general.get))
  def firtoolPreprocessTransformsOptionsDestroy(options: FirtoolPreprocessTransformsOptions) = CAPI.firtoolPreprocessTransformsOptionsDestroy(options.get)
  def firtoolPreprocessTransformsOptionsSetGeneral(options: FirtoolPreprocessTransformsOptions, general: FirtoolGeneralOptions) = CAPI.firtoolPreprocessTransformsOptionsSetGeneral(options.get, general.get)
  def firtoolPreprocessTransformsOptionsGetGeneral(options: FirtoolPreprocessTransformsOptions) = FirtoolGeneralOptions(CAPI.firtoolPreprocessTransformsOptionsGetGeneral(arena, options.get))
  def firtoolPreprocessTransformsOptionsSetDisableAnnotationsUnknown(options: FirtoolPreprocessTransformsOptions, value: Boolean) = CAPI.firtoolPreprocessTransformsOptionsSetDisableAnnotationsUnknown(options.get, value)
  def firtoolPreprocessTransformsOptionsGetDisableAnnotationsUnknown(options: FirtoolPreprocessTransformsOptions): Boolean = CAPI.firtoolPreprocessTransformsOptionsGetDisableAnnotationsUnknown(options.get)
  def firtoolPreprocessTransformsOptionsSetDisableAnnotationsClassless(options: FirtoolPreprocessTransformsOptions, value: Boolean) = CAPI.firtoolPreprocessTransformsOptionsSetDisableAnnotationsClassless(options.get, value)
  def firtoolPreprocessTransformsOptionsGetDisableAnnotationsClassless(options: FirtoolPreprocessTransformsOptions): Boolean = CAPI.firtoolPreprocessTransformsOptionsGetDisableAnnotationsClassless(options.get)
  def firtoolPreprocessTransformsOptionsSetLowerAnnotationsNoRefTypePorts(options: FirtoolPreprocessTransformsOptions, value: Boolean) = CAPI.firtoolPreprocessTransformsOptionsSetLowerAnnotationsNoRefTypePorts(options.get, value)
  def firtoolPreprocessTransformsOptionsGetLowerAnnotationsNoRefTypePorts(options: FirtoolPreprocessTransformsOptions): Boolean = CAPI.firtoolPreprocessTransformsOptionsGetLowerAnnotationsNoRefTypePorts(options.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsCreateDefault(general: FirtoolGeneralOptions) = FirtoolCHIRRTLToLowFIRRTLOptions(CAPI.firtoolCHIRRTLToLowFIRRTLOptionsCreateDefault(arena, general.get))
  def firtoolCHIRRTLToLowFIRRTLOptionsDestroy(options: FirtoolCHIRRTLToLowFIRRTLOptions) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsDestroy(options.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsSetGeneral(options: FirtoolCHIRRTLToLowFIRRTLOptions, general: FirtoolGeneralOptions) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetGeneral(options.get, general.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetGeneral(options: FirtoolCHIRRTLToLowFIRRTLOptions) = FirtoolGeneralOptions(CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetGeneral(arena, options.get))
  def firtoolCHIRRTLToLowFIRRTLOptionsSetPreserveValues(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: FirtoolPreserveValuesMode) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetPreserveValues(options.get, value.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetPreserveValues(options: FirtoolCHIRRTLToLowFIRRTLOptions) = new FirtoolPreserveValuesMode(CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetPreserveValues(options.get))
  def firtoolCHIRRTLToLowFIRRTLOptionsSetPreserveAggregate(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: FirtoolPreserveAggregateMode) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetPreserveAggregate(options.get, value.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetPreserveAggregate(options: FirtoolCHIRRTLToLowFIRRTLOptions) = new FirtoolPreserveAggregateMode(CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetPreserveAggregate(options.get))
  def firtoolCHIRRTLToLowFIRRTLOptionsSetExportChiselInterface(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: Boolean) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetExportChiselInterface(options.get, value)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetExportChiselInterface(options: FirtoolCHIRRTLToLowFIRRTLOptions): Boolean = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetExportChiselInterface(options.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsSetChiselInterfaceOutDirectory(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: String) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetChiselInterfaceOutDirectory(options.get, newString(value).get)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetChiselInterfaceOutDirectory(options: FirtoolCHIRRTLToLowFIRRTLOptions): String = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetChiselInterfaceOutDirectory(arena, options.get).toString
  def firtoolCHIRRTLToLowFIRRTLOptionsSetDisableHoistingHWPassthrough(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: Boolean) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetDisableHoistingHWPassthrough(options.get, value)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetDisableHoistingHWPassthrough(options: FirtoolCHIRRTLToLowFIRRTLOptions): Boolean = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetDisableHoistingHWPassthrough(options.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsSetDedup(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: Boolean) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetDedup(options.get, value)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetDedup(options: FirtoolCHIRRTLToLowFIRRTLOptions): Boolean = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetDedup(options.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsSetNoDedup(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: Boolean) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetNoDedup(options.get, value)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetNoDedup(options: FirtoolCHIRRTLToLowFIRRTLOptions): Boolean = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetNoDedup(options.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsSetVbToBv(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: Boolean) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetVbToBv(options.get, value)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetVbToBv(options: FirtoolCHIRRTLToLowFIRRTLOptions): Boolean = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetVbToBv(options.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsSetLowerMemories(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: Boolean) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetLowerMemories(options.get, value)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetLowerMemories(options: FirtoolCHIRRTLToLowFIRRTLOptions): Boolean = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetLowerMemories(options.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsSetCompanionMode(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: FirtoolCompanionMode) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetCompanionMode(options.get, value.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetCompanionMode(options: FirtoolCHIRRTLToLowFIRRTLOptions) = new FirtoolCompanionMode(CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetCompanionMode(options.get))
  def firtoolCHIRRTLToLowFIRRTLOptionsSetBlackBoxRootPath(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: String) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetBlackBoxRootPath(options.get, newString(value).get)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetBlackBoxRootPath(options: FirtoolCHIRRTLToLowFIRRTLOptions): String = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetBlackBoxRootPath(arena, options.get).toString
  def firtoolCHIRRTLToLowFIRRTLOptionsSetEmitOMIR(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: Boolean) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetEmitOMIR(options.get, value)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetEmitOMIR(options: FirtoolCHIRRTLToLowFIRRTLOptions): Boolean = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetEmitOMIR(options.get)
  def firtoolCHIRRTLToLowFIRRTLOptionsSetOMIROutFile(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: String) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetOMIROutFile(options.get, newString(value).get)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetOMIROutFile(options: FirtoolCHIRRTLToLowFIRRTLOptions): String = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetOMIROutFile(arena, options.get).toString
  def firtoolCHIRRTLToLowFIRRTLOptionsSetDisableAggressiveMergeConnections(options: FirtoolCHIRRTLToLowFIRRTLOptions, value: Boolean) = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsSetDisableAggressiveMergeConnections(options.get, value)
  def firtoolCHIRRTLToLowFIRRTLOptionsGetDisableAggressiveMergeConnections(options: FirtoolCHIRRTLToLowFIRRTLOptions): Boolean = CAPI.firtoolCHIRRTLToLowFIRRTLOptionsGetDisableAggressiveMergeConnections(options.get)
  def firtoolLowFIRRTLToHWOptionsCreateDefault(general: FirtoolGeneralOptions) = FirtoolLowFIRRTLToHWOptions(CAPI.firtoolLowFIRRTLToHWOptionsCreateDefault(arena, general.get))
  def firtoolLowFIRRTLToHWOptionsDestroy(options: FirtoolLowFIRRTLToHWOptions) = CAPI.firtoolLowFIRRTLToHWOptionsDestroy(options.get)
  def firtoolLowFIRRTLToHWOptionsSetGeneral(options: FirtoolLowFIRRTLToHWOptions, general: FirtoolGeneralOptions) = CAPI.firtoolLowFIRRTLToHWOptionsSetGeneral(options.get, general.get)
  def firtoolLowFIRRTLToHWOptionsGetGeneral(options: FirtoolLowFIRRTLToHWOptions) = FirtoolGeneralOptions(CAPI.firtoolLowFIRRTLToHWOptionsGetGeneral(arena, options.get))
  def firtoolLowFIRRTLToHWOptionsSetOutputAnnotationFilename(options: FirtoolLowFIRRTLToHWOptions, value: String) = CAPI.firtoolLowFIRRTLToHWOptionsSetOutputAnnotationFilename(options.get, newString(value).get)
  def firtoolLowFIRRTLToHWOptionsGetOutputAnnotationFilename(options: FirtoolLowFIRRTLToHWOptions): String = CAPI.firtoolLowFIRRTLToHWOptionsGetOutputAnnotationFilename(arena, options.get).toString
  def firtoolLowFIRRTLToHWOptionsSetEnableAnnotationWarning(options: FirtoolLowFIRRTLToHWOptions, value: Boolean) = CAPI.firtoolLowFIRRTLToHWOptionsSetEnableAnnotationWarning(options.get, value)
  def firtoolLowFIRRTLToHWOptionsGetEnableAnnotationWarning(options: FirtoolLowFIRRTLToHWOptions): Boolean = CAPI.firtoolLowFIRRTLToHWOptionsGetEnableAnnotationWarning(options.get)
  def firtoolLowFIRRTLToHWOptionsSetEmitChiselAssertsAsSVA(options: FirtoolLowFIRRTLToHWOptions, value: Boolean) = CAPI.firtoolLowFIRRTLToHWOptionsSetEmitChiselAssertsAsSVA(options.get, value)
  def firtoolLowFIRRTLToHWOptionsGetEmitChiselAssertsAsSVA(options: FirtoolLowFIRRTLToHWOptions): Boolean = CAPI.firtoolLowFIRRTLToHWOptionsGetEmitChiselAssertsAsSVA(options.get)
  def firtoolHWToSVOptionsCreateDefault(general: FirtoolGeneralOptions) = FirtoolHWToSVOptions(CAPI.firtoolHWToSVOptionsCreateDefault(arena, general.get))
  def firtoolHWToSVOptionsDestroy(options: FirtoolHWToSVOptions) = CAPI.firtoolHWToSVOptionsDestroy(options.get)
  def firtoolHWToSVOptionsSetGeneral(options: FirtoolHWToSVOptions, general: FirtoolGeneralOptions) = CAPI.firtoolHWToSVOptionsSetGeneral(options.get, general.get)
  def firtoolHWToSVOptionsGetGeneral(options: FirtoolHWToSVOptions) = FirtoolGeneralOptions(CAPI.firtoolHWToSVOptionsGetGeneral(arena, options.get))
  def firtoolHWToSVOptionsSetExtractTestCode(options: FirtoolHWToSVOptions, value: Boolean) = CAPI.firtoolHWToSVOptionsSetExtractTestCode(options.get, value)
  def firtoolHWToSVOptionsGetExtractTestCode(options: FirtoolHWToSVOptions): Boolean = CAPI.firtoolHWToSVOptionsGetExtractTestCode(options.get)
  def firtoolHWToSVOptionsSetEtcDisableInstanceExtraction(options: FirtoolHWToSVOptions, value: Boolean) = CAPI.firtoolHWToSVOptionsSetEtcDisableInstanceExtraction(options.get, value)
  def firtoolHWToSVOptionsGetEtcDisableInstanceExtraction(options: FirtoolHWToSVOptions): Boolean = CAPI.firtoolHWToSVOptionsGetEtcDisableInstanceExtraction(options.get)
  def firtoolHWToSVOptionsSetEtcDisableRegisterExtraction(options: FirtoolHWToSVOptions, value: Boolean) = CAPI.firtoolHWToSVOptionsSetEtcDisableRegisterExtraction(options.get, value)
  def firtoolHWToSVOptionsGetEtcDisableRegisterExtraction(options: FirtoolHWToSVOptions): Boolean = CAPI.firtoolHWToSVOptionsGetEtcDisableRegisterExtraction(options.get)
  def firtoolHWToSVOptionsSetEtcDisableModuleInlining(options: FirtoolHWToSVOptions, value: Boolean) = CAPI.firtoolHWToSVOptionsSetEtcDisableModuleInlining(options.get, value)
  def firtoolHWToSVOptionsGetEtcDisableModuleInlining(options: FirtoolHWToSVOptions): Boolean = CAPI.firtoolHWToSVOptionsGetEtcDisableModuleInlining(options.get)
  def firtoolHWToSVOptionsSetCkgModuleName(options: FirtoolHWToSVOptions, value: String) = CAPI.firtoolHWToSVOptionsSetCkgModuleName(options.get, newString(value).get)
  def firtoolHWToSVOptionsGetCkgModuleName(options: FirtoolHWToSVOptions): String = CAPI.firtoolHWToSVOptionsGetCkgModuleName(arena, options.get).toString
  def firtoolHWToSVOptionsSetCkgInputName(options: FirtoolHWToSVOptions, value: String) = CAPI.firtoolHWToSVOptionsSetCkgInputName(options.get, newString(value).get)
  def firtoolHWToSVOptionsGetCkgInputName(options: FirtoolHWToSVOptions): String = CAPI.firtoolHWToSVOptionsGetCkgInputName(arena, options.get).toString
  def firtoolHWToSVOptionsSetCkgOutputName(options: FirtoolHWToSVOptions, value: String) = CAPI.firtoolHWToSVOptionsSetCkgOutputName(options.get, newString(value).get)
  def firtoolHWToSVOptionsGetCkgOutputName(options: FirtoolHWToSVOptions): String = CAPI.firtoolHWToSVOptionsGetCkgOutputName(arena, options.get).toString
  def firtoolHWToSVOptionsSetCkgEnableName(options: FirtoolHWToSVOptions, value: String) = CAPI.firtoolHWToSVOptionsSetCkgEnableName(options.get, newString(value).get)
  def firtoolHWToSVOptionsGetCkgEnableName(options: FirtoolHWToSVOptions): String = CAPI.firtoolHWToSVOptionsGetCkgEnableName(arena, options.get).toString
  def firtoolHWToSVOptionsSetCkgTestEnableName(options: FirtoolHWToSVOptions, value: String) = CAPI.firtoolHWToSVOptionsSetCkgTestEnableName(options.get, newString(value).get)
  def firtoolHWToSVOptionsGetCkgTestEnableName(options: FirtoolHWToSVOptions): String = CAPI.firtoolHWToSVOptionsGetCkgTestEnableName(arena, options.get).toString
  def firtoolHWToSVOptionsSetCkgInstName(options: FirtoolHWToSVOptions, value: String) = CAPI.firtoolHWToSVOptionsSetCkgInstName(options.get, newString(value).get)
  def firtoolHWToSVOptionsGetCkgInstName(options: FirtoolHWToSVOptions): String = CAPI.firtoolHWToSVOptionsGetCkgInstName(arena, options.get).toString
  def firtoolHWToSVOptionsSetEmitSeparateAlwaysBlocks(options: FirtoolHWToSVOptions, value: Boolean) = CAPI.firtoolHWToSVOptionsSetEmitSeparateAlwaysBlocks(options.get, value)
  def firtoolHWToSVOptionsGetEmitSeparateAlwaysBlocks(options: FirtoolHWToSVOptions): Boolean = CAPI.firtoolHWToSVOptionsGetEmitSeparateAlwaysBlocks(options.get)
  def firtoolHWToSVOptionsSetAddMuxPragmas(options: FirtoolHWToSVOptions, value: Boolean) = CAPI.firtoolHWToSVOptionsSetAddMuxPragmas(options.get, value)
  def firtoolHWToSVOptionsGetAddMuxPragmas(options: FirtoolHWToSVOptions): Boolean = CAPI.firtoolHWToSVOptionsGetAddMuxPragmas(options.get)
  def firtoolHWToSVOptionsSetAddVivadoRAMAddressConflictSynthesisBugWorkaround(options: FirtoolHWToSVOptions, value: Boolean) = CAPI.firtoolHWToSVOptionsSetAddVivadoRAMAddressConflictSynthesisBugWorkaround(options.get, value)
  def firtoolHWToSVOptionsGetAddVivadoRAMAddressConflictSynthesisBugWorkaround(options: FirtoolHWToSVOptions): Boolean = CAPI.firtoolHWToSVOptionsGetAddVivadoRAMAddressConflictSynthesisBugWorkaround(options.get)
  def firtoolExportVerilogOptionsCreateDefault(general: FirtoolGeneralOptions) = FirtoolExportVerilogOptions(CAPI.firtoolExportVerilogOptionsCreateDefault(arena, general.get))
  def firtoolExportVerilogOptionsDestroy(options: FirtoolExportVerilogOptions) = CAPI.firtoolExportVerilogOptionsDestroy(options.get)
  def firtoolExportVerilogOptionsSetGeneral(options: FirtoolExportVerilogOptions, general: FirtoolGeneralOptions) = CAPI.firtoolExportVerilogOptionsSetGeneral(options.get, general.get)
  def firtoolExportVerilogOptionsGetGeneral(options: FirtoolExportVerilogOptions) = FirtoolGeneralOptions(CAPI.firtoolExportVerilogOptionsGetGeneral(arena, options.get))
  def firtoolExportVerilogOptionsSetStripFirDebugInfo(options: FirtoolExportVerilogOptions, value: Boolean) = CAPI.firtoolExportVerilogOptionsSetStripFirDebugInfo(options.get, value)
  def firtoolExportVerilogOptionsGetStripFirDebugInfo(options: FirtoolExportVerilogOptions): Boolean = CAPI.firtoolExportVerilogOptionsGetStripFirDebugInfo(options.get)
  def firtoolExportVerilogOptionsSetStripDebugInfo(options: FirtoolExportVerilogOptions, value: Boolean) = CAPI.firtoolExportVerilogOptionsSetStripDebugInfo(options.get, value)
  def firtoolExportVerilogOptionsGetStripDebugInfo(options: FirtoolExportVerilogOptions): Boolean = CAPI.firtoolExportVerilogOptionsGetStripDebugInfo(options.get)
  def firtoolExportVerilogOptionsSetExportModuleHierarchy(options: FirtoolExportVerilogOptions, value: Boolean) = CAPI.firtoolExportVerilogOptionsSetExportModuleHierarchy(options.get, value)
  def firtoolExportVerilogOptionsGetExportModuleHierarchy(options: FirtoolExportVerilogOptions): Boolean = CAPI.firtoolExportVerilogOptionsGetExportModuleHierarchy(options.get)
  def firtoolExportVerilogOptionsSetOutputPath(options: FirtoolExportVerilogOptions, value: String) = CAPI.firtoolExportVerilogOptionsSetOutputPath(options.get, newString(value).get)
  def firtoolExportVerilogOptionsGetOutputPath(options: FirtoolExportVerilogOptions): String = CAPI.firtoolExportVerilogOptionsGetOutputPath(arena, options.get).toString

  def firtoolPopulatePreprocessTransforms(pm: MlirPassManager, options: FirtoolPreprocessTransformsOptions) = MlirLogicalResult(
    CAPI.firtoolPopulatePreprocessTransforms(arena, pm.get, options.get)
  )

  def firtoolPopulateCHIRRTLToLowFIRRTL(pm: MlirPassManager, options: FirtoolCHIRRTLToLowFIRRTLOptions) = MlirLogicalResult(
    CAPI.firtoolPopulateCHIRRTLToLowFIRRTL(arena, pm.get, options.get)
  )

  def firtoolPopulateLowFIRRTLToHW(pm: MlirPassManager, options: FirtoolLowFIRRTLToHWOptions) = MlirLogicalResult(
    CAPI.firtoolPopulateLowFIRRTLToHW(arena, pm.get, options.get)
  )

  def firtoolPopulateHWToSV(pm: MlirPassManager, options: FirtoolHWToSVOptions) = MlirLogicalResult(
    CAPI.firtoolPopulateHWToSV(arena, pm.get, options.get)
  )

  def firtoolPopulateExportVerilog(pm: MlirPassManager, options: FirtoolExportVerilogOptions, callback: String => Unit) =
    MlirLogicalResult(
      CAPI.firtoolPopulateExportVerilog(arena, pm.get, options.get, newStringCallback(callback).get, NULL)
    )

  def firtoolPopulateExportSplitVerilog(pm: MlirPassManager, options: FirtoolExportVerilogOptions) =
    MlirLogicalResult(
      CAPI.firtoolPopulateExportSplitVerilog(arena, pm.get, options.get)
    )

  def firtoolPopulateFinalizeIR(pm: MlirPassManager, options: FirtoolFinalizeIROptions) = MlirLogicalResult(
    CAPI.firtoolPopulateFinalizeIR(arena, pm.get, options.get)
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

final case class FirtoolGeneralOptions(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.FirtoolGeneralOptions.sizeof().toInt
}
object FirtoolGeneralOptions {
  private[circt] def apply(ptr: MemorySegment) = new FirtoolGeneralOptions(ptr)
}

final case class FirtoolPreprocessTransformsOptions(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.FirtoolPreprocessTransformsOptions.sizeof().toInt
}
object FirtoolPreprocessTransformsOptions {
  private[circt] def apply(ptr: MemorySegment) = new FirtoolPreprocessTransformsOptions(ptr)
}

final case class FirtoolCHIRRTLToLowFIRRTLOptions(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.FirtoolCHIRRTLToLowFIRRTLOptions.sizeof().toInt
}
object FirtoolCHIRRTLToLowFIRRTLOptions {
  private[circt] def apply(ptr: MemorySegment) = new FirtoolCHIRRTLToLowFIRRTLOptions(ptr)
}

final case class FirtoolLowFIRRTLToHWOptions(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.FirtoolLowFIRRTLToHWOptions.sizeof().toInt
}
object FirtoolLowFIRRTLToHWOptions {
  private[circt] def apply(ptr: MemorySegment) = new FirtoolLowFIRRTLToHWOptions(ptr)
}

final case class FirtoolHWToSVOptions(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.FirtoolHWToSVOptions.sizeof().toInt
}
object FirtoolHWToSVOptions {
  private[circt] def apply(ptr: MemorySegment) = new FirtoolHWToSVOptions(ptr)
}

final case class FirtoolExportVerilogOptions(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.FirtoolExportVerilogOptions.sizeof().toInt
}
object FirtoolExportVerilogOptions {
  private[circt] def apply(ptr: MemorySegment) = new FirtoolExportVerilogOptions(ptr)
}

final case class FirtoolFinalizeIROptions(ptr: MemorySegment) extends ForeignType[MemorySegment] {
  private[circt] def get = ptr
  private[circt] val sizeof = circt.FirtoolFinalizeIROptions.sizeof().toInt
}
object FirtoolFinalizeIROptions {
  private[circt] def apply(ptr: MemorySegment) = new FirtoolFinalizeIROptions(ptr)
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
