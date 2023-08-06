//===- RuntimeCostEstimateInterfaceImpl.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the RuntimeCostEstimateDialectInterface for various dialects and
// exposes them using registration functions.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcInterfaces.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace comb;
using namespace hw;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static uint32_t sumNonConstantOperands(Operation *op, uint32_t perOperand) {
  uint32_t count = 0;
  for (auto operand : op->getOperands())
    if (!operand.template getDefiningOp<ConstantOp>())
      count += perOperand;
  return count;
}

//===----------------------------------------------------------------------===//
// Dialect interface implementations
//===----------------------------------------------------------------------===//

namespace {

class CombRuntimeCostEstimateDialectInterface
    : public RuntimeCostEstimateDialectInterface {
  using RuntimeCostEstimateDialectInterface::
      RuntimeCostEstimateDialectInterface;

  uint32_t getCostEstimate(mlir::Operation *op) const final {
    assert(isa<CombDialect>(op->getDialect()));

    return TypeSwitch<Operation *, uint32_t>(op)
        // ExtractOp is either lowered to shift+AND or only an AND operation.
        // Due to the high throughput of these simple ops, the real cost is
        // likely lower than the 1 or 2 cycles.
        .Case<ExtractOp>([](auto op) { return (op.getLowBit() > 0) * 4 + 4; })
        // TODO: improve this measure as it might lower to a sext or a mul
        .Case<ReplicateOp>([](auto op) { return 20; })
        .Case<MuxOp, ShlOp, ShrUOp, ShrSOp, SubOp, ICmpOp>(
            [](auto op) { return 10; })
        // NOTE: provided that the ISA has a popcount operation that takes 1
        // cycle
        .Case<ParityOp>([](auto op) { return 20; })
        // NOTE: processor performance varies a lot for these operations
        .Case<DivUOp, DivSOp, ModUOp, ModSOp>([](auto op) { return 100; })
        .Case<MulOp>([](auto op) { return (op->getNumOperands() - 1) * 30; })
        .Case<AddOp, AndOp, OrOp, XorOp>(
            [](auto op) { return (op->getNumOperands() - 1) * 10; })
        .Case<ConcatOp>(
            std::bind(&sumNonConstantOperands, std::placeholders::_1, 20));
  }
};

class HWRuntimeCostEstimateDialectInterface
    : public RuntimeCostEstimateDialectInterface {
  using RuntimeCostEstimateDialectInterface::
      RuntimeCostEstimateDialectInterface;

  uint32_t getCostEstimate(mlir::Operation *op) const final {
    assert(circt::isa<HWDialect>(op->getDialect()));

    return llvm::TypeSwitch<mlir::Operation *, uint32_t>(op)
        .Case<ConstantOp, EnumConstantOp, BitcastOp, AggregateConstantOp>(
            [](auto op) { return 0; })
        .Case<ArrayGetOp, StructExtractOp, StructInjectOp, UnionExtractOp>(
            [](auto op) { return 10; })
        .Case<ArrayCreateOp, StructCreateOp, StructExplodeOp, UnionCreateOp>(
            std::bind(&sumNonConstantOperands, std::placeholders::_1, 10));
    // TODO: ArraySliceOp, ArrayConcatOp
  }
};

class SCFRuntimeCostEstimateDialectInterface
    : public RuntimeCostEstimateDialectInterface {
  using RuntimeCostEstimateDialectInterface::
      RuntimeCostEstimateDialectInterface;

  uint32_t getCostEstimate(mlir::Operation *op) const final {
    assert(isa<scf::SCFDialect>(op->getDialect()));

    return llvm::TypeSwitch<mlir::Operation *, uint32_t>(op)
        .Case<scf::YieldOp>([](auto op) { return 0; })
        // TODO: this is chosen quite arbitrarily right now
        .Case<scf::IfOp>([](auto op) { return 20; });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration functions
//===----------------------------------------------------------------------===//

void arc::registerCombRuntimeCostEstimateInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, comb::CombDialect *dialect) {
    dialect->addInterfaces<CombRuntimeCostEstimateDialectInterface>();
  });
}

void arc::registerHWRuntimeCostEstimateInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, hw::HWDialect *dialect) {
    dialect->addInterfaces<HWRuntimeCostEstimateDialectInterface>();
  });
}

void arc::registerSCFRuntimeCostEstimateInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    dialect->addInterfaces<SCFRuntimeCostEstimateDialectInterface>();
  });
}
