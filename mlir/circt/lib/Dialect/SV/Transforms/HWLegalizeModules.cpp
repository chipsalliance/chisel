//===- HWLegalizeModulesPass.cpp - Lower unsupported IR features away -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers away features in the SV/Comb/HW dialects that are
// unsupported by some tools (e.g. multidimensional arrays) as specified by
// LoweringOptions.  This pass is run relatively late in the pipeline in
// preparation for emission.  Any passes run after this (e.g. PrettifyVerilog)
// must be aware they cannot introduce new invalid constructs.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/Builders.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// HWLegalizeModulesPass
//===----------------------------------------------------------------------===//

namespace {
struct HWLegalizeModulesPass
    : public sv::HWLegalizeModulesBase<HWLegalizeModulesPass> {
  void runOnOperation() override;

private:
  void processPostOrder(Block &block);
  Operation *tryLoweringArrayGet(hw::ArrayGetOp getOp);

  /// This is the current hw.module being processed.
  hw::HWModuleOp thisHWModule;

  bool anythingChanged;

  /// This tells us what language features we're allowed to use in generated
  /// Verilog.
  LoweringOptions options;

  /// This pass will be run on multiple hw.modules, this keeps track of the
  /// contents of LoweringOptions so we don't have to reparse the
  /// LoweringOptions for every hw.module.
  StringAttr lastParsedOptions;
};
} // end anonymous namespace

/// Try to lower a hw.array_get in module that doesn't support packed arrays.
/// This returns a replacement operation if lowering was successful, null
/// otherwise.
Operation *HWLegalizeModulesPass::tryLoweringArrayGet(hw::ArrayGetOp getOp) {
  SmallVector<Value> caseValues;
  OpBuilder builder(&thisHWModule.getBodyBlock()->front());
  // If the operand is an array_create or aggregate constant, then we can lower
  // this into a casez.
  if (auto createOp = getOp.getInput().getDefiningOp<hw::ArrayCreateOp>())
    caseValues = SmallVector<Value>(llvm::reverse(createOp.getOperands()));
  else if (auto aggregateConstant =
               getOp.getInput().getDefiningOp<hw::AggregateConstantOp>()) {
    for (auto elem : llvm::reverse(aggregateConstant.getFields())) {
      if (auto intAttr = dyn_cast<IntegerAttr>(elem))
        caseValues.push_back(builder.create<hw::ConstantOp>(
            aggregateConstant.getLoc(), intAttr));
      else
        caseValues.push_back(builder.create<hw::AggregateConstantOp>(
            aggregateConstant.getLoc(), getOp.getType(),
            elem.cast<ArrayAttr>()));
    }
  } else {
    return nullptr;
  }

  // array_get(idx, array_create(a,b,c,d)) ==> casez(idx).
  Value index = getOp.getIndex();

  // Create the wire for the result of the casez in the hw.module.
  auto theWire = builder.create<sv::RegOp>(getOp.getLoc(), getOp.getType(),
                                           builder.getStringAttr("casez_tmp"));
  builder.setInsertionPoint(getOp);

  auto loc = getOp.getInput().getDefiningOp()->getLoc();
  // A casez is a procedural operation, so if we're in a non-procedural region
  // we need to inject an always_comb block.
  if (!getOp->getParentOp()->hasTrait<sv::ProceduralRegion>()) {
    auto alwaysComb = builder.create<sv::AlwaysCombOp>(loc);
    builder.setInsertionPointToEnd(alwaysComb.getBodyBlock());
  }

  // If we are missing elements in the array (it is non-power of two), then
  // add a default 'X' value.
  if (1ULL << index.getType().getIntOrFloatBitWidth() != caseValues.size()) {
    caseValues.push_back(
        builder.create<sv::ConstantXOp>(getOp.getLoc(), getOp.getType()));
  }

  APInt caseValue(index.getType().getIntOrFloatBitWidth(), 0);
  auto *context = builder.getContext();

  // Create the casez itself.
  builder.create<sv::CaseOp>(
      loc, CaseStmtType::CaseZStmt, index, caseValues.size(),
      [&](size_t caseIdx) -> std::unique_ptr<sv::CasePattern> {
        // Use a default pattern for the last value, even if we are complete.
        // This avoids tools thinking they need to insert a latch due to
        // potentially incomplete case coverage.
        bool isDefault = caseIdx == caseValues.size() - 1;
        Value theValue = caseValues[caseIdx];
        std::unique_ptr<sv::CasePattern> thePattern;

        if (isDefault)
          thePattern = std::make_unique<sv::CaseDefaultPattern>(context);
        else
          thePattern = std::make_unique<sv::CaseBitPattern>(caseValue, context);
        ++caseValue;
        builder.create<sv::BPAssignOp>(loc, theWire, theValue);
        return thePattern;
      });

  // Ok, emit the read from the wire to get the value out.
  builder.setInsertionPoint(getOp);
  auto readWire = builder.create<sv::ReadInOutOp>(getOp.getLoc(), theWire);
  getOp.getResult().replaceAllUsesWith(readWire);
  getOp->erase();
  return readWire;
}

void HWLegalizeModulesPass::processPostOrder(Block &body) {
  if (body.empty())
    return;

  // Walk the block bottom-up, processing the region tree inside out.
  Block::iterator it = std::prev(body.end());
  while (it != body.end()) {
    auto &op = *it;

    // Advance the iterator, using the end iterator as a sentinel that we're at
    // the top of the block.
    if (it == body.begin())
      it = body.end();
    else
      --it;

    if (op.getNumRegions()) {
      for (auto &region : op.getRegions())
        for (auto &regionBlock : region.getBlocks())
          processPostOrder(regionBlock);
    }

    if (options.disallowPackedArrays) {
      // Try idioms for lowering array_get operations.
      if (auto getOp = dyn_cast<hw::ArrayGetOp>(op))
        if (auto *replacement = tryLoweringArrayGet(getOp)) {
          it = Block::iterator(replacement);
          anythingChanged = true;
          continue;
        }

      // If this is a dead array, then we can just delete it.  This is
      // probably left over from get/create lowering.
      if (isa<hw::ArrayCreateOp, hw::AggregateConstantOp>(op) &&
          op.use_empty()) {
        op.erase();
        continue;
      }

      // Otherwise, if we aren't allowing multi-dimensional arrays, reject the
      // IR as invalid.
      // TODO: We should eventually implement a "lower types" like feature in
      // this pass.
      for (auto value : op.getResults()) {
        if (value.getType().isa<hw::ArrayType>()) {
          op.emitError("unsupported packed array expression");
          signalPassFailure();
        }
      }
    }
  }
}

void HWLegalizeModulesPass::runOnOperation() {
  thisHWModule = getOperation();

  // Parse the lowering options if necessary.
  auto optionsAttr = LoweringOptions::getAttributeFrom(
      cast<ModuleOp>(thisHWModule->getParentOp()));
  if (optionsAttr != lastParsedOptions) {
    if (optionsAttr)
      options = LoweringOptions(optionsAttr.getValue(), [&](Twine error) {
        thisHWModule.emitError(error);
      });
    else
      options = LoweringOptions();
    lastParsedOptions = optionsAttr;
  }

  // Keeps track if anything changed during this pass, used to determine if
  // the analyses were preserved.
  anythingChanged = false;

  // Walk the operations in post-order, transforming any that are interesting.
  processPostOrder(*thisHWModule.getBodyBlock());

  // If we did not change anything in the IR mark all analysis as preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass> circt::sv::createHWLegalizeModulesPass() {
  return std::make_unique<HWLegalizeModulesPass>();
}
