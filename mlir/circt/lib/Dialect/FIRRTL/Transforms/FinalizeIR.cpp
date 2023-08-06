//===- FinalizeIR.cpp - Finalize IR after ExportVerilog -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FinalizeIR pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVOps.h"

using namespace circt;

namespace {
struct FinalizeIRPass : public firrtl::FinalizeIRBase<FinalizeIRPass> {
  void runOnOperation() override;
};
} // namespace

/// Finalize the IR after ExportVerilog and before the final IR is emitted.
void FinalizeIRPass::runOnOperation() {
  // Erase any sv.verbatim ops for sideband files.
  for (auto verbatim : llvm::make_early_inc_range(
           getOperation().getBodyRegion().getOps<sv::VerbatimOp>()))
    if (auto outputFile = verbatim->getAttrOfType<hw::OutputFileAttr>(
            hw::OutputFileAttr::getMnemonic()))
      if (!outputFile.isDirectory() &&
          outputFile.getExcludeFromFilelist().getValue())
        verbatim.erase();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createFinalizeIRPass() {
  return std::make_unique<FinalizeIRPass>();
}
