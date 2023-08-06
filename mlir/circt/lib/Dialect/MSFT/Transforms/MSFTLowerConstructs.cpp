//===- MSFTLowerConstructs.cpp - MSFT constructs lowerings ------*- C++ -*-===//
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
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Support/Namespace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Lower MSFT constructs
//===----------------------------------------------------------------------===//

namespace {

struct LowerConstructsPass : public LowerConstructsBase<LowerConstructsPass>,
                             PassCommon {
  void runOnOperation() override;

  /// For naming purposes, get the inner Namespace for a module, building it
  /// lazily.
  Namespace &getNamespaceFor(Operation *mod) {
    auto ns = moduleNamespaces.find(mod);
    if (ns != moduleNamespaces.end())
      return ns->getSecond();
    Namespace &nsNew = moduleNamespaces[mod];
    SymbolCache syms;
    syms.addDefinitions(mod);
    nsNew.add(syms);
    return nsNew;
  }

private:
  DenseMap<Operation *, circt::Namespace> moduleNamespaces;
};
} // anonymous namespace

namespace {
/// Lower MSFT's OutputOp to HW's.
struct SystolicArrayOpLowering : public OpConversionPattern<SystolicArrayOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SystolicArrayOp array, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctxt = getContext();
    Location loc = array.getLoc();
    Block &peBlock = array.getPe().front();
    rewriter.setInsertionPointAfter(array);

    // For the row broadcasts, break out the row values which must be broadcast
    // to each PE.
    hw::ArrayType rowInputs =
        hw::type_cast<hw::ArrayType>(array.getRowInputs().getType());
    IntegerType rowIdxType = rewriter.getIntegerType(
        std::max(1u, llvm::Log2_64_Ceil(rowInputs.getSize())));
    SmallVector<Value> rowValues;
    for (size_t rowNum = 0, numRows = rowInputs.getSize(); rowNum < numRows;
         ++rowNum) {
      Value rowNumVal =
          rewriter.create<hw::ConstantOp>(loc, rowIdxType, rowNum);
      auto rowValue =
          rewriter.create<hw::ArrayGetOp>(loc, array.getRowInputs(), rowNumVal);
      rowValue->setAttr("sv.namehint",
                        StringAttr::get(ctxt, "row_" + Twine(rowNum)));
      rowValues.push_back(rowValue);
    }

    // For the column broadcasts, break out the column values which must be
    // broadcast to each PE.
    hw::ArrayType colInputs =
        hw::type_cast<hw::ArrayType>(array.getColInputs().getType());
    IntegerType colIdxType = rewriter.getIntegerType(
        std::max(1u, llvm::Log2_64_Ceil(colInputs.getSize())));
    SmallVector<Value> colValues;
    for (size_t colNum = 0, numCols = colInputs.getSize(); colNum < numCols;
         ++colNum) {
      Value colNumVal =
          rewriter.create<hw::ConstantOp>(loc, colIdxType, colNum);
      auto colValue =
          rewriter.create<hw::ArrayGetOp>(loc, array.getColInputs(), colNumVal);
      colValue->setAttr("sv.namehint",
                        StringAttr::get(ctxt, "col_" + Twine(colNum)));
      colValues.push_back(colValue);
    }

    // Build the PE matrix.
    SmallVector<Value> peOutputs;
    for (size_t rowNum = 0, numRows = rowInputs.getSize(); rowNum < numRows;
         ++rowNum) {
      Value rowValue = rowValues[rowNum];
      SmallVector<Value> colPEOutputs;
      for (size_t colNum = 0, numCols = colInputs.getSize(); colNum < numCols;
           ++colNum) {
        Value colValue = colValues[colNum];
        // Clone the PE block, substituting %row (arg 0) and %col (arg 1) for
        // the corresponding row/column broadcast value.
        // NOTE: the PE region is NOT a graph region so we don't have to deal
        // with backedges.
        IRMapping mapper;
        mapper.map(peBlock.getArgument(0), rowValue);
        mapper.map(peBlock.getArgument(1), colValue);
        for (Operation &peOperation : peBlock)
          // If we see the output op (which should be the block terminator), add
          // its operand to the output matrix.
          if (auto outputOp = dyn_cast<PEOutputOp>(peOperation)) {
            colPEOutputs.push_back(mapper.lookup(outputOp.getOutput()));
          } else {
            Operation *clone = rewriter.clone(peOperation, mapper);

            StringRef nameSource = "name";
            auto name = clone->getAttrOfType<StringAttr>(nameSource);
            if (!name) {
              nameSource = "sv.namehint";
              name = clone->getAttrOfType<StringAttr>(nameSource);
            }
            if (name)
              clone->setAttr(nameSource,
                             StringAttr::get(ctxt, name.getValue() + "_" +
                                                       Twine(rowNum) + "_" +
                                                       Twine(colNum)));
          }
      }
      // Reverse the vector since ArrayCreateOp has the opposite ordering to C
      // vectors.
      std::reverse(colPEOutputs.begin(), colPEOutputs.end());
      peOutputs.push_back(
          rewriter.create<hw::ArrayCreateOp>(loc, colPEOutputs));
    }

    std::reverse(peOutputs.begin(), peOutputs.end());
    rewriter.replaceOp(array,
                       rewriter.create<hw::ArrayCreateOp>(loc, peOutputs));
    return success();
  }
};
} // anonymous namespace

namespace {
/// Lower MSFT's ChannelOp to a set of registers.
struct ChannelOpLowering : public OpConversionPattern<ChannelOp> {
public:
  ChannelOpLowering(MLIRContext *ctxt, LowerConstructsPass &pass)
      : OpConversionPattern(ctxt), pass(pass) {}

  LogicalResult
  matchAndRewrite(ChannelOp chan, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = chan.getLoc();
    Operation *mod = chan->getParentOfType<MSFTModuleOp>();
    assert(mod && "ChannelOp must be contained by module");
    Namespace &ns = pass.getNamespaceFor(mod);
    Value clk = chan.getClk();
    Value v = chan.getInput();
    for (uint64_t stageNum = 0, e = chan.getDefaultStages(); stageNum < e;
         ++stageNum)
      v = rewriter.create<seq::CompRegOp>(loc, v, clk,
                                          ns.newName(chan.getSymName()));
    rewriter.replaceOp(chan, {v});
    return success();
  }

protected:
  LowerConstructsPass &pass;
};
} // namespace

void LowerConstructsPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();

  ConversionTarget target(*ctxt);
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  RewritePatternSet patterns(ctxt);
  patterns.insert<SystolicArrayOpLowering>(ctxt);
  target.addIllegalOp<SystolicArrayOp>();
  patterns.insert<ChannelOpLowering>(ctxt, *this);
  target.addIllegalOp<ChannelOp>();

  if (failed(mlir::applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::msft::createLowerConstructsPass() {
  return std::make_unique<LowerConstructsPass>();
}
