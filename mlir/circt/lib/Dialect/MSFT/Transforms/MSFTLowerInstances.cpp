//===- MSFTLowerInstances.cpp - Instace lowering pass -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Support/Namespace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Lower dynamic instances to global refs.
//===----------------------------------------------------------------------===//

namespace {
struct LowerInstancesPass : public LowerInstancesBase<LowerInstancesPass> {
  void runOnOperation() override;

  LogicalResult lower(DynamicInstanceOp inst, InstanceHierarchyOp hier,
                      OpBuilder &b);

  // Aggregation of the global ref attributes populated as a side-effect of the
  // conversion.
  DenseMap<Operation *, SmallVector<hw::GlobalRefAttr, 0>> globalRefsToApply;

  // Cache the top-level symbols. Insert the new ones we're creating for new
  // global ref ops.
  SymbolCache topSyms;

  // In order to be efficient, cache the "symbols" in each module.
  DenseMap<MSFTModuleOp, SymbolCache> perModSyms;
  // Accessor for `perModSyms` which lazily constructs each cache.
  const SymbolCache &getSyms(MSFTModuleOp mod);
};
} // anonymous namespace

const SymbolCache &LowerInstancesPass::getSyms(MSFTModuleOp mod) {
  auto symsFound = perModSyms.find(mod);
  if (symsFound != perModSyms.end())
    return symsFound->getSecond();

  // Build the cache.
  SymbolCache &syms = perModSyms[mod];
  hw::InnerSymbolTable::walkSymbols(
      mod, [&](StringAttr symName, hw::InnerSymTarget target) {
        if (!target.isPort())
          syms.addDefinition(symName, target.getOp());
      });
  return syms;
}

LogicalResult LowerInstancesPass::lower(DynamicInstanceOp inst,
                                        InstanceHierarchyOp hier,
                                        OpBuilder &b) {

  hw::GlobalRefOp ref = nullptr;

  // If 'inst' doesn't contain any ops which use a global ref op, don't create
  // one.
  if (llvm::any_of(inst.getOps(), [](Operation &op) {
        return isa<DynInstDataOpInterface>(op);
      })) {

    // Come up with a unique symbol name.
    auto refSym = StringAttr::get(&getContext(), "instref");
    auto origRefSym = refSym;
    unsigned ctr = 0;
    while (topSyms.getDefinition(refSym))
      refSym = StringAttr::get(&getContext(),
                               origRefSym.getValue() + "_" + Twine(++ctr));

    // Create a global ref to replace us.
    ArrayAttr globalRefPath = inst.globalRefPath();
    ref = b.create<hw::GlobalRefOp>(inst.getLoc(), refSym, globalRefPath);
    auto refAttr = hw::GlobalRefAttr::get(ref);

    // Add the new symbol to the symbol cache.
    topSyms.addDefinition(refSym, ref);

    // For each level of `globalRef`, find the static operation which needs a
    // back reference to the global ref which is replacing us.
    bool symNotFound = false;
    for (auto innerRef : globalRefPath.getAsRange<hw::InnerRefAttr>()) {
      MSFTModuleOp mod =
          cast<MSFTModuleOp>(topSyms.getDefinition(innerRef.getModule()));
      const SymbolCache &modSyms = getSyms(mod);
      Operation *tgtOp = modSyms.getDefinition(innerRef.getName());
      if (!tgtOp) {
        symNotFound = true;
        inst.emitOpError("Could not find ")
            << innerRef.getName() << " in module " << innerRef.getModule();
        continue;
      }
      // Add the backref to the list of attributes to apply.
      globalRefsToApply[tgtOp].push_back(refAttr);

      // Since GlobalRefOp uses the `inner_sym` attribute, assign the
      // 'inner_sym' attribute if it's not already assigned.
      if (!tgtOp->hasAttr("inner_sym")) {
        tgtOp->setAttr("inner_sym", hw::InnerSymAttr::get(innerRef.getName()));
      }
    }
    if (symNotFound)
      return inst.emitOpError(
          "Could not find operation corresponding to instance reference");
  }

  // Relocate all my children.
  OpBuilder hierBlock(&hier.getBody().front().front());
  for (Operation &op : llvm::make_early_inc_range(inst.getOps())) {
    // Child instances should have been lowered already.
    assert(!isa<DynamicInstanceOp>(op));
    op.remove();
    hierBlock.insert(&op);

    // Assign a ref for ops which need it.
    if (auto specOp = dyn_cast<DynInstDataOpInterface>(op)) {
      assert(ref);
      specOp.setGlobalRef(ref);
    }
  }

  inst.erase();
  return success();
}
void LowerInstancesPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();

  // Populate the top level symbol cache.
  topSyms.addDefinitions(top);

  size_t numFailed = 0;
  OpBuilder builder(ctxt);

  // Find all of the InstanceHierarchyOps.
  for (Operation &op : llvm::make_early_inc_range(top.getOps())) {
    auto instHierOp = dyn_cast<InstanceHierarchyOp>(op);
    if (!instHierOp)
      continue;
    builder.setInsertionPoint(&op);
    // Walk the child dynamic instances in _post-order_ so we lower and delete
    // the children first.
    instHierOp->walk<mlir::WalkOrder::PostOrder>([&](DynamicInstanceOp inst) {
      if (failed(lower(inst, instHierOp, builder)))
        ++numFailed;
    });
  }
  if (numFailed)
    signalPassFailure();

  // Since applying a large number of attributes is very expensive in MLIR (both
  // in terms of time and memory), bulk-apply the attributes necessary for
  // `hw.globalref`s.
  for (auto opRefPair : globalRefsToApply) {
    ArrayRef<hw::GlobalRefAttr> refArr = opRefPair.getSecond();
    SmallVector<Attribute> newGlobalRefs(
        llvm::map_range(refArr, [](hw::GlobalRefAttr ref) { return ref; }));
    Operation *op = opRefPair.getFirst();
    if (auto refArr =
            op->getAttrOfType<ArrayAttr>(hw::GlobalRefAttr::DialectAttrName))
      newGlobalRefs.append(refArr.getValue().begin(), refArr.getValue().end());
    op->setAttr(hw::GlobalRefAttr::DialectAttrName,
                ArrayAttr::get(ctxt, newGlobalRefs));
  }
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createLowerInstancesPass() {
  return std::make_unique<LowerInstancesPass>();
}
} // namespace msft
} // namespace circt
