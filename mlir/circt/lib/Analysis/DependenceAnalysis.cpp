//===- DependenceAnalysis.cpp - memory dependence analyses ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that perform analysis involving memory access
// dependences.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::affine;
using namespace circt::analysis;

/// Helper to iterate through memory operation pairs and check for dependences
/// at a given loop nesting depth.
static void checkMemrefDependence(SmallVectorImpl<Operation *> &memoryOps,
                                  unsigned depth,
                                  MemoryDependenceResult &results) {
  for (auto *source : memoryOps) {
    for (auto *destination : memoryOps) {
      if (source == destination)
        continue;

      // Initialize the dependence list for this destination.
      if (results.count(destination) == 0)
        results[destination] = SmallVector<MemoryDependence>();

      // Look for inter-iteration dependences on the same memory location.
      MemRefAccess src(source);
      MemRefAccess dst(destination);
      FlatAffineValueConstraints dependenceConstraints;
      SmallVector<DependenceComponent, 2> depComps;

      // Requested depth might not be a valid comparison if they do not belong
      // to the same loop nest
      if (depth > getInnermostCommonLoopDepth({source, destination}))
        continue;

      DependenceResult result = checkMemrefAccessDependence(
          src, dst, depth, &dependenceConstraints, &depComps, true);

      results[destination].emplace_back(source, result.value, depComps);

      // Also consider intra-iteration dependences on the same memory location.
      // This currently does not consider aliasing.
      if (src != dst)
        continue;

      // Collect surrounding loops to use in dependence components. Only proceed
      // if we are in the innermost loop.
      SmallVector<AffineForOp> enclosingLoops;
      getAffineForIVs(*destination, &enclosingLoops);
      if (enclosingLoops.size() != depth)
        continue;

      // Look for the common parent that src and dst share. If there is none,
      // there is nothing more to do.
      SmallVector<Operation *> srcParents;
      getEnclosingAffineOps(*source, &srcParents);
      SmallVector<Operation *> dstParents;
      getEnclosingAffineOps(*destination, &dstParents);

      Operation *commonParent = nullptr;
      for (auto *srcParent : llvm::reverse(srcParents)) {
        for (auto *dstParent : llvm::reverse(dstParents)) {
          if (srcParent == dstParent)
            commonParent = srcParent;
          if (commonParent != nullptr)
            break;
        }
        if (commonParent != nullptr)
          break;
      }

      if (commonParent == nullptr)
        continue;

      // Check the common parent's regions.
      for (auto &commonRegion : commonParent->getRegions()) {
        if (commonRegion.empty())
          continue;

        // Only support structured constructs with single-block regions for now.
        assert(commonRegion.hasOneBlock() &&
               "only single-block regions are supported");

        Block &commonBlock = commonRegion.front();

        // Find the src and dst ancestor in the common block, if any.
        Operation *srcOrAncestor = commonBlock.findAncestorOpInBlock(*source);
        Operation *dstOrAncestor =
            commonBlock.findAncestorOpInBlock(*destination);
        if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
          continue;

        // Check if the src or its ancestor is before the dst or its ancestor.
        if (srcOrAncestor->isBeforeInBlock(dstOrAncestor)) {
          // Build dependence components for each loop depth.
          SmallVector<DependenceComponent> intraDeps;
          for (size_t i = 0; i < depth; ++i) {
            DependenceComponent depComp;
            depComp.op = enclosingLoops[i];
            depComp.lb = 0;
            depComp.ub = 0;
            intraDeps.push_back(depComp);
          }

          results[destination].emplace_back(
              source, DependenceResult::HasDependence, intraDeps);
        }
      }
    }
  }
}

/// MemoryDependenceAnalysis traverses any AffineForOps in the FuncOp body and
/// checks for memory access dependences. Results are captured in a
/// MemoryDependenceResult, which can by queried by Operation.
circt::analysis::MemoryDependenceAnalysis::MemoryDependenceAnalysis(
    Operation *op) {
  auto funcOp = cast<func::FuncOp>(op);

  // Collect affine loops grouped by nesting depth.
  std::vector<SmallVector<AffineForOp, 2>> depthToLoops;
  mlir::affine::gatherLoops(funcOp, depthToLoops);

  // Collect load and store operations to check.
  SmallVector<Operation *> memoryOps;
  funcOp.walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      memoryOps.push_back(op);
  });

  // For each depth, check memref accesses.
  for (unsigned depth = 1, e = depthToLoops.size(); depth <= e; ++depth)
    checkMemrefDependence(memoryOps, depth, results);
}

/// Returns the dependences, if any, that the given Operation depends on.
ArrayRef<MemoryDependence>
circt::analysis::MemoryDependenceAnalysis::getDependences(Operation *op) {
  return results[op];
}

/// Replaces the dependences, if any, from the oldOp to the newOp.
void circt::analysis::MemoryDependenceAnalysis::replaceOp(Operation *oldOp,
                                                          Operation *newOp) {
  // If oldOp had any dependences.
  auto it = results.find(oldOp);
  if (it != results.end())
    // Move the dependences to newOp.
    it->first = newOp;

  // Find any dependences originating from oldOp and make newOp the source.
  // TODO(mikeurbach): consider adding an inverted index to avoid this scan.
  for (auto &it : results)
    for (auto &dep : it.second)
      if (dep.source == oldOp)
        dep.source = newOp;
}
