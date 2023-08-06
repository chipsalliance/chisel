//===- MSFTExportTcl.cpp - TCL export pass ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Support/Namespace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Export tcl -- create tcl verbatim ops
//===----------------------------------------------------------------------===//

namespace {
template <typename PhysOpTy>
struct RemovePhysOpLowering : public OpConversionPattern<PhysOpTy> {
  using OpConversionPattern<PhysOpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<PhysOpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(PhysOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};
} // anonymous namespace

namespace {
struct ExportTclPass : public ExportTclBase<ExportTclPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ExportTclPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();
  TclEmitter emitter(top);

  // Traverse MSFT location attributes and export the required Tcl into
  // templated `sv::VerbatimOp`s with symbolic references to the instance paths.
  for (const std::string &moduleName : tops) {
    Operation *hwmod =
        emitter.getDefinition(FlatSymbolRefAttr::get(ctxt, moduleName));
    if (!hwmod) {
      top.emitError("Failed to find module '") << moduleName << "'";
      signalPassFailure();
      return;
    }
    if (failed(emitter.emit(hwmod, tclFile))) {
      hwmod->emitError("failed to emit tcl");
      signalPassFailure();
      return;
    }
  }

  ConversionTarget target(*ctxt);
  target.addIllegalDialect<msft::MSFTDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<sv::SVDialect>();

  RewritePatternSet patterns(ctxt);
  patterns.insert<RemovePhysOpLowering<PDPhysLocationOp>>(ctxt);
  patterns.insert<RemovePhysOpLowering<PDRegPhysLocationOp>>(ctxt);
  patterns.insert<RemovePhysOpLowering<PDPhysRegionOp>>(ctxt);
  patterns.insert<RemovePhysOpLowering<InstanceHierarchyOp>>(ctxt);
  patterns.insert<RemovePhysOpLowering<DynamicInstanceVerbatimAttrOp>>(ctxt);
  patterns.insert<RemoveOpLowering<DeclPhysicalRegionOp>>(ctxt);
  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();

  target.addDynamicallyLegalOp<hw::GlobalRefOp>([&](hw::GlobalRefOp ref) {
    return !emitter.getRefsUsed().contains(ref);
  });
  patterns.clear();
  patterns.insert<RemoveOpLowering<hw::GlobalRefOp>>(ctxt);
  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::msft::createExportTclPass() {
  return std::make_unique<ExportTclPass>();
}
