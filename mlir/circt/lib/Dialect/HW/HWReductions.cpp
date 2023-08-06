//===- HWReductions.cpp - Reduction patterns for the HW dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWReductions.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Reduce/ReductionUtils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hw-reductions"

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Utility to track the transitive size of modules.
struct ModuleSizeCache {
  void clear() { moduleSizes.clear(); }

  uint64_t getModuleSize(HWModuleLike module,
                         InstanceGraphBase &instanceGraph) {
    if (auto it = moduleSizes.find(module); it != moduleSizes.end())
      return it->second;
    uint64_t size = 1;
    module->walk([&](Operation *op) {
      size += 1;
      if (auto instOp = dyn_cast<HWInstanceLike>(op))
        if (auto instModule = instanceGraph.getReferencedModule(instOp))
          size += getModuleSize(instModule, instanceGraph);
    });
    moduleSizes.insert({module, size});
    return size;
  }

private:
  llvm::DenseMap<Operation *, uint64_t> moduleSizes;
};

//===----------------------------------------------------------------------===//
// Reduction patterns
//===----------------------------------------------------------------------===//

/// A sample reduction pattern that maps `hw.module` to `hw.module.extern`.
struct ModuleExternalizer : public OpReduction<HWModuleOp> {
  void beforeReduction(mlir::ModuleOp op) override {
    instanceGraph = std::make_unique<InstanceGraph>(op);
    moduleSizes.clear();
  }

  uint64_t match(HWModuleOp op) override {
    return moduleSizes.getModuleSize(op, *instanceGraph);
  }

  LogicalResult rewrite(HWModuleOp op) override {
    OpBuilder builder(op);
    builder.create<HWModuleExternOp>(op->getLoc(), op.getModuleNameAttr(),
                                     op.getPortList(), StringRef(),
                                     op.getParameters());
    op->erase();
    return success();
  }

  std::string getName() const override { return "hw-module-externalizer"; }

  std::unique_ptr<InstanceGraph> instanceGraph;
  ModuleSizeCache moduleSizes;
};

/// A sample reduction pattern that replaces all uses of an operation with one
/// of its operands. This can help pruning large parts of the expression tree
/// rapidly.
template <unsigned OpNum>
struct HWOperandForwarder : public Reduction {
  uint64_t match(Operation *op) override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 2 ||
        OpNum >= op->getNumOperands())
      return 0;
    auto resultTy = op->getResult(0).getType().dyn_cast<IntegerType>();
    auto opTy = op->getOperand(OpNum).getType().dyn_cast<IntegerType>();
    return resultTy && opTy && resultTy == opTy &&
           op->getResult(0) != op->getOperand(OpNum);
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    ImplicitLocOpBuilder builder(op->getLoc(), op);
    auto result = op->getResult(0);
    auto operand = op->getOperand(OpNum);
    LLVM_DEBUG(llvm::dbgs()
               << "Forwarding " << operand << " in " << *op << "\n");
    result.replaceAllUsesWith(operand);
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override {
    return ("hw-operand" + Twine(OpNum) + "-forwarder").str();
  }
};

/// A sample reduction pattern that replaces integer operations with a constant
/// zero of their type.
struct HWConstantifier : public Reduction {
  uint64_t match(Operation *op) override {
    if (op->getNumResults() == 0 || op->getNumOperands() == 0)
      return 0;
    return llvm::all_of(op->getResults(), [](Value result) {
      return result.getType().isa<IntegerType>();
    });
  }
  LogicalResult rewrite(Operation *op) override {
    assert(match(op));
    OpBuilder builder(op);
    for (auto result : op->getResults()) {
      auto type = result.getType().cast<IntegerType>();
      auto newOp = builder.create<hw::ConstantOp>(op->getLoc(), type, 0);
      result.replaceAllUsesWith(newOp);
    }
    reduce::pruneUnusedOps(op, *this);
    return success();
  }
  std::string getName() const override { return "hw-constantifier"; }
};

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

void HWReducePatternDialectInterface::populateReducePatterns(
    circt::ReducePatternSet &patterns) const {
  // Gather a list of reduction patterns that we should try. Ideally these are
  // assigned reasonable benefit indicators (higher benefit patterns are
  // prioritized). For example, things that can knock out entire modules while
  // being cheap should be tried first (and thus have higher benefit), before
  // trying to tweak operands of individual arithmetic ops.
  patterns.add<ModuleExternalizer, 6>();
  patterns.add<HWConstantifier, 5>();
  patterns.add<HWOperandForwarder<0>, 4>();
  patterns.add<HWOperandForwarder<1>, 3>();
  patterns.add<HWOperandForwarder<2>, 2>();
}

void hw::registerReducePatternDialectInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HWDialect *dialect) {
    dialect->addInterfaces<HWReducePatternDialectInterface>();
  });
}
