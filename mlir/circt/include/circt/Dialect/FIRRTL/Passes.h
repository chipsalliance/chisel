//===- Passes.h - FIRRTL pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_PASSES_H
#define CIRCT_DIALECT_FIRRTL_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace firrtl {

std::unique_ptr<mlir::Pass>
createLowerFIRRTLAnnotationsPass(bool ignoreUnhandledAnnotations = false,
                                 bool ignoreClasslessAnnotations = false,
                                 bool noRefTypePorts = false);

std::unique_ptr<mlir::Pass> createLowerOpenAggsPass();

/// Configure which aggregate values will be preserved by the LowerTypes pass.
namespace PreserveAggregate {
enum PreserveMode {
  /// Don't preserve aggregate at all. This has been default behaivor and
  /// compatible with SFC.
  None,

  /// Preserve only 1d vectors of ground type (e.g. UInt<2>[3]).
  OneDimVec,

  /// Preserve only vectors (e.g. UInt<2>[3][3]).
  Vec,

  /// Preserve all aggregate values.
  All,
};
}

std::unique_ptr<mlir::Pass> createLowerFIRRTLTypesPass(
    PreserveAggregate::PreserveMode mode = PreserveAggregate::None,
    PreserveAggregate::PreserveMode memoryMode = PreserveAggregate::None);

std::unique_ptr<mlir::Pass> createLowerBundleVectorTypesPass();

std::unique_ptr<mlir::Pass> createLowerCHIRRTLPass();

std::unique_ptr<mlir::Pass> createLowerIntrinsicsPass();

std::unique_ptr<mlir::Pass> createIMConstPropPass();

std::unique_ptr<mlir::Pass>
createRemoveUnusedPortsPass(bool ignoreDontTouch = false);

std::unique_ptr<mlir::Pass> createInlinerPass();

std::unique_ptr<mlir::Pass> createInferReadWritePass();

std::unique_ptr<mlir::Pass>
createCreateSiFiveMetadataPass(bool replSeqMem = false,
                               mlir::StringRef replSeqMemFile = "");

std::unique_ptr<mlir::Pass> createWireDFTPass();

std::unique_ptr<mlir::Pass> createVBToBVPass();

std::unique_ptr<mlir::Pass> createAddSeqMemPortsPass();

std::unique_ptr<mlir::Pass> createDedupPass();

std::unique_ptr<mlir::Pass>
createEmitOMIRPass(mlir::StringRef outputFilename = "");

std::unique_ptr<mlir::Pass> createLowerMatchesPass();

std::unique_ptr<mlir::Pass> createExpandWhensPass();

std::unique_ptr<mlir::Pass> createFlattenMemoryPass();

std::unique_ptr<mlir::Pass> createInferWidthsPass();

std::unique_ptr<mlir::Pass> createInferResetsPass();

std::unique_ptr<mlir::Pass> createLowerMemoryPass();

std::unique_ptr<mlir::Pass>
createMemToRegOfVecPass(bool replSeqMem = false, bool ignoreReadEnable = false);

std::unique_ptr<mlir::Pass> createPrefixModulesPass();

std::unique_ptr<mlir::Pass> createFIRRTLFieldSourcePass();

std::unique_ptr<mlir::Pass> createPrintInstanceGraphPass();

std::unique_ptr<mlir::Pass> createPrintNLATablePass();

std::unique_ptr<mlir::Pass>
createBlackBoxReaderPass(std::optional<mlir::StringRef> inputPrefix = {});

std::unique_ptr<mlir::Pass>
createGrandCentralPass(bool instantiateCompanionOnly = false);

std::unique_ptr<mlir::Pass> createCheckCombLoopsPass();

std::unique_ptr<mlir::Pass> createSFCCompatPass();

std::unique_ptr<mlir::Pass>
createMergeConnectionsPass(bool enableAggressiveMerging = false);

std::unique_ptr<mlir::Pass> createVectorizationPass();

std::unique_ptr<mlir::Pass> createInjectDUTHierarchyPass();

std::unique_ptr<mlir::Pass> createDropConstPass();

/// Configure which values will be explicitly preserved by the DropNames pass.
namespace PreserveValues {
enum PreserveMode {
  /// Don't explicitly preserve any named values. Every named operation could
  /// be optimized away by the compiler.
  None,
  // Explicitly preserved values with meaningful names.  If a name begins with
  // an "_" it is not considered meaningful.
  Named,
  // Explicitly preserve all values.  No named operation should be optimized
  // away by the compiler.
  All,
};
}

std::unique_ptr<mlir::Pass>
createDropNamesPass(PreserveValues::PreserveMode mode = PreserveValues::None);

std::unique_ptr<mlir::Pass> createExtractInstancesPass();

std::unique_ptr<mlir::Pass> createIMDeadCodeElimPass();

std::unique_ptr<mlir::Pass> createRandomizeRegisterInitPass();

std::unique_ptr<mlir::Pass> createRegisterOptimizerPass();

std::unique_ptr<mlir::Pass> createLowerXMRPass();

std::unique_ptr<mlir::Pass>
createResolveTracesPass(mlir::StringRef outputAnnotationFilename = "");

std::unique_ptr<mlir::Pass> createInnerSymbolDCEPass();

std::unique_ptr<mlir::Pass> createFinalizeIRPass();

std::unique_ptr<mlir::Pass> createExtractClassesPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/FIRRTL/Passes.h.inc"

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_PASSES_H
