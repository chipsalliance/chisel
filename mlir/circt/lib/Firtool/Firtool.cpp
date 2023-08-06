//===- Firtool.cpp - Definitions for the firtool pipeline setup -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Firtool/Firtool.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace circt;

LogicalResult firtool::populateCHIRRTLToLowFIRRTL(mlir::PassManager &pm,
                                                  const FirtoolOptions &opt,
                                                  ModuleOp module,
                                                  StringRef inputFilename) {
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerIntrinsicsPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInjectDUTHierarchyPass());

  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createDropNamesPass(opt.getPreserveMode()));

  if (!opt.disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        mlir::createCSEPass());

  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createLowerCHIRRTLPass());

  // Run LowerMatches before InferWidths, as the latter does not support the
  // match statement, but it does support what they lower to.
  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createLowerMatchesPass());

  // Width inference creates canonicalization opportunities.
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferWidthsPass());

  pm.nest<firrtl::CircuitOp>().addPass(
      firrtl::createMemToRegOfVecPass(opt.replSeqMem, opt.ignoreReadEnableMem));

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferResetsPass());

  if (opt.exportChiselInterface) {
    if (opt.chiselInterfaceOutDirectory.empty()) {
      pm.nest<firrtl::CircuitOp>().addPass(createExportChiselInterfacePass());
    } else {
      pm.nest<firrtl::CircuitOp>().addPass(createExportSplitChiselInterfacePass(
          opt.chiselInterfaceOutDirectory));
    }
  }

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createDropConstPass());

  if (opt.dedup)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createDedupPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createWireDFTPass());

  if (opt.vbToBV) {
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass(
        firrtl::PreserveAggregate::All, firrtl::PreserveAggregate::All));
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createVBToBVPass());
  }

  if (!opt.lowerMemories)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createFlattenMemoryPass());

  // The input mlir file could be firrtl dialect so we might need to clean
  // things up.
  pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass(
      opt.preserveAggregate, firrtl::PreserveAggregate::None));
  // Only enable expand whens if lower types is also enabled.
  auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
  modulePM.addPass(firrtl::createExpandWhensPass());
  modulePM.addPass(firrtl::createSFCCompatPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInlinerPass());

  // Preset the random initialization parameters for each module. The current
  // implementation assumes it can run at a time where every register is
  // currently in the final module it will be emitted in, all registers have
  // been created, and no registers have yet been removed.
  if (opt.isRandomEnabled(FirtoolOptions::RandomKind::Reg))
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createRandomizeRegisterInitPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createCheckCombLoopsPass());

  // If we parsed a FIRRTL file and have optimizations enabled, clean it up.
  if (!opt.disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createSimpleCanonicalizerPass());

  // Run the infer-rw pass, which merges read and write ports of a memory with
  // mutually exclusive enables.
  if (!opt.disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createInferReadWritePass());

  if (opt.replSeqMem)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerMemoryPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createPrefixModulesPass());

  if (!opt.disableOptimization)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMConstPropPass());

  pm.addNestedPass<firrtl::CircuitOp>(firrtl::createAddSeqMemPortsPass());

  pm.addPass(firrtl::createCreateSiFiveMetadataPass(opt.replSeqMem,
                                                    opt.replSeqMemFile));

  pm.addNestedPass<firrtl::CircuitOp>(firrtl::createExtractInstancesPass());

  // Run passes to resolve Grand Central features.  This should run before
  // BlackBoxReader because Grand Central needs to inform BlackBoxReader where
  // certain black boxes should be placed.  Note: all Grand Central Taps related
  // collateral is resolved entirely by LowerAnnotations.
  pm.addNestedPass<firrtl::CircuitOp>(
      firrtl::createGrandCentralPass(opt.grandCentralInstantiateCompanionOnly));

  // Read black box source files into the IR.
  StringRef blackBoxRoot = opt.blackBoxRootPath.empty()
                               ? llvm::sys::path::parent_path(inputFilename)
                               : opt.blackBoxRootPath;
  pm.nest<firrtl::CircuitOp>().addPass(
      firrtl::createBlackBoxReaderPass(blackBoxRoot));

  // Run SymbolDCE as late as possible, but before InnerSymbolDCE. This is for
  // hierpathop's and just for general cleanup.
  pm.addNestedPass<firrtl::CircuitOp>(mlir::createSymbolDCEPass());

  // Run InnerSymbolDCE as late as possible, but before IMDCE.
  pm.addPass(firrtl::createInnerSymbolDCEPass());

  // The above passes, IMConstProp in particular, introduce additional
  // canonicalization opportunities that we should pick up here before we
  // proceed to output-specific pipelines.
  if (!opt.disableOptimization) {
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createSimpleCanonicalizerPass());
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        circt::firrtl::createRegisterOptimizerPass());
    pm.addPass(firrtl::createIMDeadCodeElimPass());
  }

  if (opt.emitOMIR)
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createEmitOMIRPass(opt.omirOutFile));

  // Always run this, required for legalization.
  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createMergeConnectionsPass(
          !opt.disableAggressiveMergeConnections.getValue()));

  if (!opt.disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createVectorizationPass());

  return success();
}

LogicalResult firtool::populateLowFIRRTLToHW(mlir::PassManager &pm,
                                             const FirtoolOptions &opt) {
  // Remove TraceAnnotations and write their updated paths to an output
  // annotation file.
  if (opt.outputAnnotationFilename.empty())
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createResolveTracesPass());
  else
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createResolveTracesPass(
        opt.outputAnnotationFilename.getValue()));

  // Lower the ref.resolve and ref.send ops and remove the RefType ports.
  // LowerToHW cannot handle RefType so, this pass must be run to remove all
  // RefType ports and ops.
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerXMRPass());

  pm.addPass(createLowerFIRRTLToHWPass(
      opt.enableAnnotationWarning.getValue(),
      opt.emitChiselAssertsAsSVA.getValue(),
      !opt.isRandomEnabled(FirtoolOptions::RandomKind::Mem),
      !opt.isRandomEnabled(FirtoolOptions::RandomKind::Reg)));

  if (!opt.disableOptimization) {
    auto &modulePM = pm.nest<hw::HWModuleOp>();
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(createSimpleCanonicalizerPass());
  }

  return success();
}

LogicalResult firtool::populateHWToSV(mlir::PassManager &pm,
                                      const FirtoolOptions &opt) {
  if (opt.extractTestCode)
    pm.addPass(sv::createSVExtractTestCodePass(opt.etcDisableInstanceExtraction,
                                               opt.etcDisableRegisterExtraction,
                                               opt.etcDisableModuleInlining));

  pm.addPass(seq::createExternalizeClockGatePass(opt.clockGateOpts));
  auto &modulePM = pm.nest<hw::HWModuleOp>();
  modulePM.addPass(seq::createSeqFIRRTLLowerToSVPass(
      {/*disableRandomization=*/!opt.isRandomEnabled(
           FirtoolOptions::RandomKind::Reg),
       /*emitSeparateAlwaysBlocks=*/
       opt.emitSeparateAlwaysBlocks}));
  modulePM.addPass(createLowerVerifToSVPass());
  pm.addPass(seq::createLowerFirMemPass());
  pm.addPass(sv::createHWMemSimImplPass(
      opt.replSeqMem, opt.ignoreReadEnableMem, opt.addMuxPragmas,
      !opt.isRandomEnabled(FirtoolOptions::RandomKind::Mem),
      !opt.isRandomEnabled(FirtoolOptions::RandomKind::Reg),
      opt.addVivadoRAMAddressConflictSynthesisBugWorkaround));

  // If enabled, run the optimizer.
  if (!opt.disableOptimization) {
    auto &modulePM = pm.nest<hw::HWModuleOp>();
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(createSimpleCanonicalizerPass());
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(sv::createHWCleanupPass(
        /*mergeAlwaysBlocks=*/!opt.emitSeparateAlwaysBlocks));
  }
  return success();
}
