//===- ArcPasses.h - Arc dialect passes -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCPASSES_H
#define CIRCT_DIALECT_ARC_ARCPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <optional>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace arc {

std::unique_ptr<mlir::Pass>
createAddTapsPass(std::optional<bool> tapPorts = {},
                  std::optional<bool> tapWires = {},
                  std::optional<bool> tapNamedValues = {});
std::unique_ptr<mlir::Pass> createAllocateStatePass();
std::unique_ptr<mlir::Pass> createArcCanonicalizerPass();
std::unique_ptr<mlir::Pass> createDedupPass();
std::unique_ptr<mlir::Pass> createGroupResetsAndEnablesPass();
std::unique_ptr<mlir::Pass>
createInferMemoriesPass(std::optional<bool> tapPorts = {});
std::unique_ptr<mlir::Pass> createInferStatePropertiesPass();
std::unique_ptr<mlir::Pass> createInlineArcsPass();
std::unique_ptr<mlir::Pass> createInlineModulesPass();
std::unique_ptr<mlir::Pass> createIsolateClocksPass();
std::unique_ptr<mlir::Pass> createLatencyRetimingPass();
std::unique_ptr<mlir::Pass> createLegalizeStateUpdatePass();
std::unique_ptr<mlir::Pass> createLowerClocksToFuncsPass();
std::unique_ptr<mlir::Pass> createLowerLUTPass();
std::unique_ptr<mlir::Pass> createLowerStatePass();
std::unique_ptr<mlir::Pass> createMakeTablesPass();
std::unique_ptr<mlir::Pass> createMuxToControlFlowPass();
std::unique_ptr<mlir::Pass>
createPrintStateInfoPass(llvm::StringRef stateFile = "");
std::unique_ptr<mlir::Pass> createSimplifyVariadicOpsPass();
std::unique_ptr<mlir::Pass> createSplitLoopsPass();
std::unique_ptr<mlir::Pass> createStripSVPass();

#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Arc/ArcPasses.h.inc"

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_ARCPASSES_H
