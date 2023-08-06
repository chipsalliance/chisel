//===- Vectorization.cpp -  Vectorize primitive operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass performs vectorization for primitive operations, e.g:
// vector_create (or a[0], b[0]), (or a[1], b[1]), (or a[2], b[2])
// => elementwise_or a, b
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-vectorization"

using namespace circt;
using namespace firrtl;

namespace {
//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {

template <typename OpTy, typename ResultOpType>
class VectorCreateToLogicElementwise : public mlir::RewritePattern {
public:
  VectorCreateToLogicElementwise(MLIRContext *context)
      : RewritePattern(VectorCreateOp::getOperationName(), 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto vectorCreateOp = cast<VectorCreateOp>(op);
    FVectorType type = vectorCreateOp.getType();
    if (type.hasUninferredWidth() || !type_isa<UIntType>(type.getElementType()))
      return failure();

    SmallVector<Value> lhs, rhs;

    // Vectorize if all operands are `OpTy`. Currently there is no other
    // condition so it could be too aggressive.
    if (llvm::all_of(op->getOperands(), [&](Value operand) {
          auto op = operand.getDefiningOp<OpTy>();
          if (!op)
            return false;
          lhs.push_back(op.getLhs());
          rhs.push_back(op.getRhs());
          return true;
        })) {
      auto lhsVec = rewriter.createOrFold<VectorCreateOp>(
          op->getLoc(), vectorCreateOp.getType(), lhs);
      auto rhsVec = rewriter.createOrFold<VectorCreateOp>(
          op->getLoc(), vectorCreateOp.getType(), rhs);
      rewriter.replaceOpWithNewOp<ResultOpType>(op, lhsVec, rhsVec);
      return success();
    }
    return failure();
  }
};
} // namespace

struct VectorizationPass : public VectorizationBase<VectorizationPass> {
  VectorizationPass() = default;
  void runOnOperation() override;
};

} // namespace

void VectorizationPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Running Vectorization "
                             "--------------------------------------===\n"
                          << "Module: '" << getOperation().getName() << "'\n";);

  RewritePatternSet patterns(&getContext());
  patterns
      .insert<VectorCreateToLogicElementwise<OrPrimOp, ElementwiseOrPrimOp>,
              VectorCreateToLogicElementwise<AndPrimOp, ElementwiseAndPrimOp>,
              VectorCreateToLogicElementwise<XorPrimOp, ElementwiseXorPrimOp>>(
          &getContext());
  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  (void)applyPatternsAndFoldGreedily(getOperation(), frozenPatterns);
}

std::unique_ptr<mlir::Pass> circt::firrtl::createVectorizationPass() {
  return std::make_unique<VectorizationPass>();
}
