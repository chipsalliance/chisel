//===- RegisterOptimizer.cpp - Register Optimizer ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass optimized registers as allowed by historic firrtl register
// behaviors.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-register-optimizer"

using namespace circt;
using namespace firrtl;

// Instantiated for RegOp and RegResetOp
template <typename T>
static bool canErase(T op) {
  return !(hasDontTouch(op.getResult()) || op.isForceable() ||
           (op.getAnnotationsAttr() && !op.getAnnotationsAttr().empty()));
}

namespace {

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

struct RegisterOptimizerPass
    : public RegisterOptimizerBase<RegisterOptimizerPass> {
  void runOnOperation() override;
  void checkRegReset(mlir::DominanceInfo &dom,
                     SmallVector<Operation *> &toErase, RegResetOp reg);
  void checkReg(mlir::DominanceInfo &dom, SmallVector<Operation *> &toErase,
                RegOp reg);
};

} // namespace

void RegisterOptimizerPass::checkReg(mlir::DominanceInfo &dom,
                                     SmallVector<Operation *> &toErase,
                                     RegOp reg) {
  if (!canErase(reg))
    return;
  auto con = getSingleConnectUserOf(reg.getResult());
  if (!con)
    return;

  // Register is only written by itself, replace with invalid.
  if (con.getSrc() == reg.getResult()) {
    auto inv = OpBuilder(reg).create<InvalidValueOp>(reg.getLoc(),
                                                     reg.getResult().getType());
    reg.getResult().replaceAllUsesWith(inv.getResult());
    toErase.push_back(reg);
    toErase.push_back(con);
    return;
  }
  // Register is only written by a constant
  if (isConstant(con.getSrc())) {
    // constant may not dominate the register.  But it might be the next
    // operation, so we can't just move it.  Straight constants can be
    // rematerialized.  Derived constants are piped through wires.

    if (auto cst = con.getSrc().getDefiningOp<ConstantOp>()) {
      // Simple constants we can move safely
      auto *fmodb = con->getParentOfType<FModuleOp>().getBodyBlock();
      cst->moveBefore(fmodb, fmodb->begin());
      reg.getResult().replaceAllUsesWith(cst.getResult());
      toErase.push_back(con);
    } else {
      bool dominatesAll = true;
      for (auto *use : reg->getUsers()) {
        if (use == con)
          continue;
        if (!dom.dominates(con.getSrc(), use)) {
          dominatesAll = false;
          break;
        }
      }
      if (dominatesAll) {
        // Dominance is fine, just replace the op.
        reg.getResult().replaceAllUsesWith(con.getSrc());
        toErase.push_back(con);
      } else {
        auto bounce = OpBuilder(reg).create<WireOp>(reg.getLoc(),
                                                    reg.getResult().getType());
        reg.replaceAllUsesWith(bounce);
      }
    }
    toErase.push_back(reg);
    return;
  }
}

void RegisterOptimizerPass::checkRegReset(mlir::DominanceInfo &dom,
                                          SmallVector<Operation *> &toErase,
                                          RegResetOp reg) {
  if (!canErase(reg))
    return;
  auto con = getSingleConnectUserOf(reg.getResult());
  if (!con)
    return;

  // Register is only written by itself, and reset with a constant.
  if (reg.getResetValue().getType() == reg.getResult().getType()) {
    if (con.getSrc() == reg.getResult() && isConstant(reg.getResetValue())) {
      // constant obviously dominates the register.
      reg.getResult().replaceAllUsesWith(reg.getResetValue());
      toErase.push_back(reg);
      toErase.push_back(con);
      return;
    }
    // Register is only written by a constant, and reset with the same constant.
    if (con.getSrc() == reg.getResetValue() &&
        isConstant(reg.getResetValue())) {
      // constant obviously dominates the register.
      reg.getResult().replaceAllUsesWith(reg.getResetValue());
      toErase.push_back(reg);
      toErase.push_back(con);
      return;
    }
  }
}

void RegisterOptimizerPass::runOnOperation() {
  auto mod = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "===----- Running RegisterOptimizer "
                             "--------------------------------------===\n"
                          << "Module: '" << mod.getName() << "'\n";);

  SmallVector<Operation *> toErase;
  mlir::DominanceInfo dom(mod);

  for (auto &op : *mod.getBodyBlock()) {
    if (auto reg = dyn_cast<RegResetOp>(&op))
      checkRegReset(dom, toErase, reg);
    else if (auto reg = dyn_cast<RegOp>(&op))
      checkReg(dom, toErase, reg);
  }
  for (auto *op : toErase)
    op->erase();

  if (!toErase.empty())
    return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createRegisterOptimizerPass() {
  return std::make_unique<RegisterOptimizerPass>();
}
