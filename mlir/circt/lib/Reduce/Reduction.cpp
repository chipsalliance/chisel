//===- Reduction.cpp - Reductions for circt-reduce ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines datastructures to handle reduction patterns.
//
//===----------------------------------------------------------------------===//

#include "circt/Reduce/Reduction.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "circt-reduce"

using namespace circt;

//===----------------------------------------------------------------------===//
// Reduction
//===----------------------------------------------------------------------===//

Reduction::~Reduction() = default;

//===----------------------------------------------------------------------===//
// Pass Reduction
//===----------------------------------------------------------------------===//

PassReduction::PassReduction(MLIRContext *context, std::unique_ptr<Pass> pass,
                             bool canIncreaseSize, bool oneShot)
    : context(context), canIncreaseSize(canIncreaseSize), oneShot(oneShot) {
  passName = pass->getArgument();
  if (passName.empty())
    passName = pass->getName();

  pm = std::make_unique<mlir::PassManager>(
      context, "builtin.module", mlir::OpPassManager::Nesting::Explicit);
  auto opName = pass->getOpName();
  if (opName && opName->equals("firrtl.circuit"))
    pm->nest<firrtl::CircuitOp>().addPass(std::move(pass));
  else if (opName && opName->equals("firrtl.module"))
    pm->nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        std::move(pass));
  else if (opName && opName->equals("hw.module"))
    pm->nest<hw::HWModuleOp>().addPass(std::move(pass));
  else
    pm->addPass(std::move(pass));
}

uint64_t PassReduction::match(Operation *op) {
  return op->getName() == pm->getOpName(*context);
}

LogicalResult PassReduction::rewrite(Operation *op) { return pm->run(op); }

std::string PassReduction::getName() const { return passName.str(); }

//===----------------------------------------------------------------------===//
// ReducePatternSet
//===----------------------------------------------------------------------===//

void ReducePatternSet::filter(
    const std::function<bool(const Reduction &)> &pred) {
  for (auto *iter = reducePatternsWithBenefit.begin();
       iter != reducePatternsWithBenefit.end(); ++iter) {
    if (!pred(*iter->first))
      reducePatternsWithBenefit.erase(iter--);
  }
}

void ReducePatternSet::sortByBenefit() {
  llvm::stable_sort(reducePatternsWithBenefit,
                    [](const auto &pairA, const auto &pairB) {
                      return pairA.second > pairB.second;
                    });
}

size_t ReducePatternSet::size() const {
  return reducePatternsWithBenefit.size();
}

Reduction &ReducePatternSet::operator[](size_t idx) const {
  return *reducePatternsWithBenefit[idx].first;
}

//===----------------------------------------------------------------------===//
// ReducePatternInterfaceCollection
//===----------------------------------------------------------------------===//

void ReducePatternInterfaceCollection::populateReducePatterns(
    ReducePatternSet &patterns) const {
  for (const ReducePatternDialectInterface &interface : *this)
    interface.populateReducePatterns(patterns);
}
