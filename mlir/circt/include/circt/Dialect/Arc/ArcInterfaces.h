//===- ArcInterfaces.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides registration functions for all external interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCINTERFACES_H
#define CIRCT_DIALECT_ARC_ARCINTERFACES_H

#include "mlir/IR/DialectInterface.h"

// Forward declarations.
namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace circt {
namespace arc {

void registerCombRuntimeCostEstimateInterface(mlir::DialectRegistry &registry);
void registerHWRuntimeCostEstimateInterface(mlir::DialectRegistry &registry);
void registerSCFRuntimeCostEstimateInterface(mlir::DialectRegistry &registry);

inline void initAllExternalInterfaces(mlir::DialectRegistry &registry) {
  registerCombRuntimeCostEstimateInterface(registry);
  registerHWRuntimeCostEstimateInterface(registry);
  registerSCFRuntimeCostEstimateInterface(registry);
}

/// A dialect interface to get runtime cost estimates of MLIR operations. This
/// is useful for implementing heuristics in optimization passes.
class RuntimeCostEstimateDialectInterface
    : public mlir::DialectInterface::Base<RuntimeCostEstimateDialectInterface> {
public:
  RuntimeCostEstimateDialectInterface(mlir::Dialect *dialect) : Base(dialect) {}

  /// Returns a number indicating the expected number of cycles the given
  /// operation will take to execute on hardware times 10 (to allow a bit more
  /// fine tuning for high-throughput operations)
  virtual uint32_t getCostEstimate(mlir::Operation *op) const = 0;
};

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_ARCINTERFACES_H
