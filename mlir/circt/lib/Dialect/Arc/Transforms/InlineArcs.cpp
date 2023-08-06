//===- InlineArcs.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-inline"

using namespace circt;
using namespace arc;
using mlir::InlinerInterface;

namespace {

struct InlineArcsPass : public InlineArcsBase<InlineArcsPass> {
  InlineArcsPass() = default;
  InlineArcsPass(bool intoArcsOnly, unsigned maxNonTrivialOpsInBody)
      : InlineArcsPass() {
    this->intoArcsOnly.setInitialValue(intoArcsOnly);
    this->maxNonTrivialOpsInBody.setInitialValue(maxNonTrivialOpsInBody);
  }
  InlineArcsPass(const InlineArcsPass &pass) : InlineArcsPass() {}

  void runOnOperation() override;
  bool shouldInline(DefineOp defOp, ArrayRef<mlir::CallOpInterface> users);
};

/// A simple implementation of the `InlinerInterface` that marks all inlining as
/// legal since we know that we only ever attempt to inline `DefineOp` bodies
/// at `StateOp` sites.
struct ArcInliner : public InlinerInterface {
  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return isa<DefineOp>(src->getParentOp());
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return op->getParentOfType<DefineOp>();
  }
};

} // namespace

void InlineArcsPass::runOnOperation() {
  auto module = getOperation();

  // Store the call hierarcy manually since we need to keep it updated and the
  // provided datastructures do not seem to support this
  DenseMap<DefineOp, DenseSet<DefineOp>> arcsCalledInBody;
  DenseMap<DefineOp, DenseSet<mlir::CallOpInterface>> usersPerArc;

  SymbolTableCollection symbolTable;
  ArcInliner inliner(&getContext());

  // Compute Arc call hierarchy
  module->walk([&](Operation *op) {
    if (auto defOp = dyn_cast<DefineOp>(op)) {
      arcsCalledInBody.insert({defOp, {}});
      usersPerArc.insert({defOp, {}});
    }

    if (auto callOp = dyn_cast<mlir::CallOpInterface>(op)) {
      if (auto defOp =
              dyn_cast<DefineOp>(callOp.resolveCallable(&symbolTable))) {
        usersPerArc[defOp].insert(callOp);
        if (auto parentOp = op->getParentOfType<DefineOp>())
          arcsCalledInBody[parentOp].insert(defOp);
      }
    }
  });

  // Remove all unused arcs
  SmallVector<DefineOp> removeList(module.getOps<DefineOp>());
  while (!removeList.empty()) {
    auto defOp = removeList.pop_back_val();
    if (!defOp)
      continue;

    if (usersPerArc[defOp].empty()) {
      usersPerArc.erase(defOp);
      for (auto callee : arcsCalledInBody[defOp]) {
        defOp->walk([&](mlir::CallOpInterface call) {
          usersPerArc[callee].erase(call);
        });
        if (usersPerArc[callee].empty())
          removeList.push_back(callee);
      }
      arcsCalledInBody.erase(defOp);
      defOp.erase();
      ++numRemovedArcs;
    }
  }

  // It's a bit ugly that we have an intermediate datastructure here, but some
  // attempts in using a datastructure with deterministic iteration over
  // arcsCalledInBody have shown worse performance.
  SmallVector<DefineOp> arcsWithoutFurtherCalls;
  for (const auto &[defOp, list] : arcsCalledInBody)
    if (defOp && list.empty())
      arcsWithoutFurtherCalls.push_back(defOp);

  // We need to sort here because we iterated over a hashmap above.
  std::sort(arcsWithoutFurtherCalls.begin(), arcsWithoutFurtherCalls.end());
  SetVector<DefineOp> worklist(arcsWithoutFurtherCalls.begin(),
                               arcsWithoutFurtherCalls.end());

  while (!worklist.empty()) {
    auto defOp = worklist.pop_back_val();

    // TODO: copying here and sorting is not ideal, we should use another
    // datastructure than a DenseSet. We need to unique the initial set of
    // elements, deterministic iteration and fast element deletion, so maybe a
    // linked list?
    SmallVector<mlir::CallOpInterface> users(usersPerArc[defOp].begin(),
                                             usersPerArc[defOp].end());
    std::sort(users.begin(), users.end());

    // Update the worklist
    for (auto caller : users) {
      if (auto parentOp = caller->getParentOfType<DefineOp>()) {
        arcsCalledInBody[parentOp].erase(defOp);
        if (arcsCalledInBody[parentOp].empty())
          worklist.insert(parentOp);
      }
    }

    // Check if we should inline the arc.
    if (!shouldInline(defOp, users))
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Inlining " << defOp.getSymName() << "\n");

    // Inline all uses of the arc. Currently we inline all of them but in the
    // future we may decide per use site whether to inline or not.
    unsigned numUsersLeft = users.size();
    for (auto user : users) {
      if (!user->getParentOfType<DefineOp>() && intoArcsOnly)
        continue;

      // Recursive calls of arcs are not allowed, but since we cannot verify
      // that in the op verifier, we need to check check here just to be sure
      if (defOp == user->getParentOfType<DefineOp>())
        continue;

      if (succeeded(mlir::inlineCall(inliner, user, defOp, &defOp.getBody(),
                                     numUsersLeft > 1))) {
        usersPerArc[defOp].erase(user);
        user.erase();
        --numUsersLeft;
        ++numInlinedArcs;
      }

      if (numUsersLeft == 0) {
        arcsCalledInBody.erase(defOp);
        usersPerArc.erase(defOp);
        defOp.erase();
        ++numRemovedArcs;
      }
    }
  }
}

bool InlineArcsPass::shouldInline(DefineOp defOp,
                                  ArrayRef<mlir::CallOpInterface> users) {
  // Count the number of non-trivial ops in the arc. If there are only a few,
  // inline the arc.
  unsigned numNonTrivialOps = 0;
  defOp.getBodyBlock().walk([&](Operation *op) {
    if (!op->hasTrait<OpTrait::ConstantLike>() && !isa<OutputOp>(op))
      ++numNonTrivialOps;
  });
  if (numNonTrivialOps <= maxNonTrivialOpsInBody) {
    ++numTrivialArcs;
    return true;
  }
  LLVM_DEBUG(llvm::dbgs() << "Arc " << defOp.getSymName() << " has "
                          << numNonTrivialOps << " non-trivial ops\n");

  // Check if the arc is only ever used once.
  if (users.size() == 1) {
    ++numSingleUseArcs;
    return true;
  }

  return false;
}

std::unique_ptr<Pass> arc::createInlineArcsPass() {
  return std::make_unique<InlineArcsPass>();
}
