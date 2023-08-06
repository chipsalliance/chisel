//===- DependenceAnalysis.h - memory dependence analyses ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform analysis
// involving memory access dependences.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_DEPENDENCE_ANALYSIS_H
#define CIRCT_ANALYSIS_DEPENDENCE_ANALYSIS_H

#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include <utility>

namespace mlir {
namespace affine {
struct DependenceComponent;
} // namespace affine
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace circt {
namespace analysis {

/// MemoryDependence captures a dependence from one memory operation to another.
/// It represents the destination of the dependence edge, the type of the
/// dependence, and the components associated with each enclosing loop.
struct MemoryDependence {
  MemoryDependence(
      Operation *source,
      mlir::affine::DependenceResult::ResultEnum dependenceType,
      ArrayRef<mlir::affine::DependenceComponent> dependenceComponents)
      : source(source), dependenceType(dependenceType),
        dependenceComponents(dependenceComponents.begin(),
                             dependenceComponents.end()) {}

  // The source Operation where this dependence originates.
  Operation *source;

  // The dependence type denotes whether or not there is a dependence.
  mlir::affine::DependenceResult::ResultEnum dependenceType;

  // The dependence components include lower and upper bounds for each loop.
  SmallVector<mlir::affine::DependenceComponent> dependenceComponents;
};

/// MemoryDependenceResult captures a set of memory dependences. The map key is
/// the operation to which the dependences exist, and the map value is zero or
/// more MemoryDependences for that operation.
using MemoryDependenceResult =
    DenseMap<Operation *, SmallVector<MemoryDependence>>;

/// MemoryDependenceAnalysis traverses any AffineForOps in the FuncOp body and
/// checks for affine memory access dependences. Non-affine memory dependences
/// are currently not supported. Results are captured in a
/// MemoryDependenceResult, and an API is exposed to query dependences of a
/// given Operation.
/// TODO(mikeurbach): consider upstreaming this to MLIR's AffineAnalysis.
struct MemoryDependenceAnalysis {
  // Construct the analysis from a FuncOp.
  MemoryDependenceAnalysis(Operation *funcOp);

  // Returns the dependences, if any, that the given Operation depends on.
  ArrayRef<MemoryDependence> getDependences(Operation *);

  // Replaces the dependences, if any, from the oldOp to the newOp.
  void replaceOp(Operation *oldOp, Operation *newOp);

private:
  // Store dependence results.
  MemoryDependenceResult results;
};

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_DEPENDENCE_ANALYSIS_H
