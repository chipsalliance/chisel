//===- RemoveCombGroups.cpp - Remove Comb Groups Pass -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Remove Comb Groups pass.
//
//===----------------------------------------------------------------------===//

/// Transforms combinational groups, which have a constant done condition,
/// into proper groups by registering the values read from the ports of cells
/// used within the combinational group.
///
/// It also transforms (invoke,if,while)-with into semantically equivalent
/// control programs that first enable a group that calculates and registers the
/// ports defined by the combinational group execute the respective cond group
/// and then execute the control operator.
///
/// # Example
/// ```
/// group comb_cond<"static"=0> {
///     lt.right = 32'd10;
///     lt.left = 32'd1;
///     eq.right = r.out;
///     eq.left = x.out;
///     comb_cond[done] = 1'd1;
/// }
/// control {
///     invoke comp(left = lt.out, ..)(..) with comb_cond;
///     if lt.out with comb_cond {
///         ...
///     }
///     while eq.out with comb_cond {
///         ...
///     }
/// }
/// ```
/// into:
/// ```
/// group comb_cond<"static"=1> {
///     lt.right = 32'd10;
///     lt.left = 32'd1;
///     eq.right = r.out;
///     eq.left = x.out;
///     lt_reg.in = lt.out
///     lt_reg.write_en = 1'd1;
///     eq_reg.in = eq.out;
///     eq_reg.write_en = 1'd1;
///     comb_cond[done] = lt_reg.done & eq_reg.done ? 1'd1;
/// }
/// control {
///     seq {
///       comb_cond;
///       invoke comp(left = lt_reg.out, ..)(..);
///     }
///     seq {
///       comb_cond;
///       if lt_reg.out {
///           ...
///       }
///     }
///     seq {
///       comb_cond;
///       while eq_reg.out {
///           ...
///           comb_cond;
///       }
///     }
/// }
/// ```

#include "PassDetails.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace calyx;
using namespace mlir;

namespace {

static calyx::RegisterOp createReg(ComponentOp component,
                                   PatternRewriter &rewriter, Location loc,
                                   Twine prefix, size_t width) {
  IRRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(component.getBodyBlock());
  return rewriter.create<calyx::RegisterOp>(loc, (prefix + "_reg").str(),
                                            width);
}

// Wraps the provided 'op' inside a newly created TOp operation, and
// returns the TOp operation.
template <typename TOp>
static TOp wrapInsideOp(OpBuilder &builder, Operation *op) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(op);
  auto newOp = builder.create<TOp>(op->getLoc());
  op->moveBefore(newOp.getBodyBlock(), newOp.getBodyBlock()->begin());
  return newOp;
}

using CombResRegMapping = DenseMap<Value, RegisterOp>;

struct RemoveCombGroupsPattern : public OpRewritePattern<calyx::CombGroupOp> {
  using OpRewritePattern::OpRewritePattern;

  RemoveCombGroupsPattern(MLIRContext *ctx, CombResRegMapping *mapping)
      : OpRewritePattern(ctx), combResRegMapping(mapping) {}

  LogicalResult matchAndRewrite(calyx::CombGroupOp combGroup,
                                PatternRewriter &rewriter) const override {

    auto component = combGroup->getParentOfType<ComponentOp>();
    auto group = rewriter.replaceOpWithNewOp<calyx::GroupOp>(
        combGroup, combGroup.getName());
    rewriter.mergeBlocks(combGroup.getBodyBlock(), group.getBodyBlock());

    // Determine which cell results are read from the control schedule.
    SetVector<Operation *> cellsAssigned;
    for (auto op : group.getOps<calyx::AssignOp>()) {
      auto defOp = dyn_cast<CellInterface>(op.getDest().getDefiningOp());
      assert(defOp && "expected some assignment to a cell");
      cellsAssigned.insert(defOp);
    }

    rewriter.setInsertionPointToStart(group.getBodyBlock());
    auto oneConstant = rewriter.create<hw::ConstantOp>(
        group.getLoc(), APInt(1, 1, /*isSigned=*/true));

    // Maintain the set of cell results which have already been assigned to
    // its register within this group.
    SetVector<Value> alreadyAssignedResults;

    // Collect register done signals. These are needed for generating the
    // GroupDone signal.
    SetVector<Value> registerDoneSigs;

    // 1. Iterate over the cells assigned within the combinational group.
    // 2. For any use of a cell result within the controls schedule.
    // 3.  Ensure that the cell result has a register.
    // 4.  Ensure that the cell result has been written to its register in this
    //     group.
    // We do not replace uses of the combinational results now, since the
    // following code relies on a checking cell result value use in the
    // control schedule, which needs to remain even when two combinational
    // groups assign to the same cell.
    for (auto *cellOp : cellsAssigned) {
      auto cell = dyn_cast<CellInterface>(cellOp);
      for (auto combRes : cell.getOutputPorts()) {
        for (auto &use : llvm::make_early_inc_range(combRes.getUses())) {
          if (use.getOwner()->getParentOfType<calyx::ControlOp>()) {
            auto combResReg = combResRegMapping->find(combRes);
            if (combResReg == combResRegMapping->end()) {
              // First time a registered variant of this result is needed.
              auto reg = createReg(component, rewriter, combRes.getLoc(),
                                   cell.instanceName(),
                                   combRes.getType().getIntOrFloatBitWidth());
              auto it = combResRegMapping->insert({combRes, reg});
              combResReg = it.first;
            }

            // Assign the cell result register - a register should only be
            // assigned once within a group.
            if (!alreadyAssignedResults.contains(combRes)) {
              rewriter.create<AssignOp>(combRes.getLoc(),
                                        combResReg->second.getIn(), combRes);
              rewriter.create<AssignOp>(combRes.getLoc(),
                                        combResReg->second.getWriteEn(),
                                        oneConstant);
              alreadyAssignedResults.insert(combRes);
            }

            registerDoneSigs.insert(combResReg->second.getDone());
          }
        }
      }
    }

    // Create a group done op with the complex &[regDone] expression as a
    // guard.
    assert(!registerDoneSigs.empty() &&
           "No registers assigned in the combinational group?");
    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    rewriter.create<calyx::GroupDoneOp>(
        group.getLoc(),
        rewriter.create<hw::ConstantOp>(group.getLoc(), APInt(1, 1)),
        rewriter.create<comb::AndOp>(combGroup.getLoc(), rewriter.getI1Type(),
                                     registerDoneSigs.takeVector()));

    return success();
  }

  mutable CombResRegMapping *combResRegMapping;
};

struct RemoveCombGroupsPass
    : public RemoveCombGroupsBase<RemoveCombGroupsPass> {
  void runOnOperation() override;

  /// Removes 'with' groups from an operation and instead schedules the group
  /// right before the oop.
  void rewriteIfWithCombGroup(OpBuilder &builder) {
    OpBuilder::InsertionGuard guard(builder);
    getOperation().walk([&](IfOp ifOp) {
      if (!ifOp.getGroupName())
        return;
      auto groupName = ifOp.getGroupName();
      // Ensure that we're inside a sequential control composition.
      wrapInsideOp<SeqOp>(builder, ifOp);
      builder.setInsertionPoint(ifOp);
      builder.create<EnableOp>(ifOp.getLoc(), groupName.value());
      ifOp.removeGroupNameAttr();
    });
  }

  void rewriteWhileWithCombGroup(OpBuilder &builder) {
    OpBuilder::InsertionGuard guard(builder);
    getOperation().walk([&](WhileOp whileOp) {
      if (!whileOp.getGroupName())
        return;
      auto groupName = whileOp.getGroupName().value();
      // Ensure that we're inside a sequential control composition.
      wrapInsideOp<SeqOp>(builder, whileOp);
      builder.setInsertionPoint(whileOp);
      builder.create<EnableOp>(whileOp.getLoc(), groupName);
      whileOp.removeGroupNameAttr();

      // Also schedule the group at the end of the while body.
      auto &curWhileBodyOp = whileOp.getBodyBlock()->front();
      builder.setInsertionPointToStart(whileOp.getBodyBlock());
      auto newSeqBody = builder.create<SeqOp>(curWhileBodyOp.getLoc());
      builder.setInsertionPointToStart(newSeqBody.getBodyBlock());
      auto condEnable =
          builder.create<EnableOp>(curWhileBodyOp.getLoc(), groupName);
      curWhileBodyOp.moveBefore(condEnable);
    });
  }

  void rewriteCellResults() {
    for (auto &&[res, reg] : combResToReg) {
      for (auto &use : llvm::make_early_inc_range(res.getUses())) {
        if (use.getOwner()->getParentOfType<calyx::ControlOp>()) {
          use.set(reg.getOut());
        }
      }
    }
  }

  CombResRegMapping combResToReg;
};

} // end anonymous namespace

void RemoveCombGroupsPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<calyx::CalyxDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addIllegalOp<calyx::CombGroupOp>();

  RewritePatternSet patterns(&getContext());

  // Maintain a mapping from combinational result SSA values to the registered
  // version of that combinational unit. This is used to avoid duplicating
  // registers when cells are used across different groups.
  patterns.add<RemoveCombGroupsPattern>(&getContext(), &combResToReg);

  if (applyPartialConversion(getOperation(), target, std::move(patterns))
          .failed())
    signalPassFailure();

  // Rewrite uses of the cell results to their registered variants.
  rewriteCellResults();

  // Rewrite 'with' uses of the previously combinational groups.
  OpBuilder builder(&getContext());
  rewriteIfWithCombGroup(builder);
  rewriteWhileWithCombGroup(builder);
}

std::unique_ptr<mlir::Pass> circt::calyx::createRemoveCombGroupsPass() {
  return std::make_unique<RemoveCombGroupsPass>();
}
