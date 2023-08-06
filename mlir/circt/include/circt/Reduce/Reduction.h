//===- Reduction.h - Reduction datastructure decl. for circt-reduce -------===//
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

#ifndef CIRCT_REDUCE_REDUCTION_H
#define CIRCT_REDUCE_REDUCTION_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {

/// An abstract reduction pattern.
struct Reduction {
  virtual ~Reduction();

  /// Called before the reduction is applied to a new subset of operations.
  /// Reductions may use this callback to collect information such as symbol
  /// tables about the module upfront.
  virtual void beforeReduction(mlir::ModuleOp) {}

  /// Called after the reduction has been applied to a subset of operations.
  /// Reductions may use this callback to perform post-processing of the
  /// reductions before the resulting module is tried for interestingness.
  virtual void afterReduction(mlir::ModuleOp) {}

  /// Check if the reduction can apply to a specific operation. Returns a
  /// benefit measure where a higher number means that applying the pattern
  /// leads to a bigger reduction and zero means that the patten does not
  /// match and thus cannot be applied at all.
  virtual uint64_t match(Operation *op) = 0;

  /// Apply the reduction to a specific operation. If the returned result
  /// indicates that the application failed, the resulting module is treated the
  /// same as if the tester marked it as uninteresting.
  virtual LogicalResult rewrite(Operation *op) = 0;

  /// Return a human-readable name for this reduction pattern.
  virtual std::string getName() const = 0;

  /// Return true if the tool should accept the transformation this reduction
  /// performs on the module even if the overall size of the output increases.
  /// This can be handy for patterns that reduce the complexity of the IR at the
  /// cost of some verbosity.
  virtual bool acceptSizeIncrease() const { return false; }

  /// Return true if the tool should not try to reapply this reduction after it
  /// has been successful. This is useful for reductions whose `match()`
  /// function keeps returning true even after the reduction has reached a
  /// fixed-point and no longer performs any change. An example of this are
  /// reductions that apply a lowering pass which always applies but may leave
  /// the input unmodified.
  ///
  /// This is mainly useful in conjunction with returning true from
  /// `acceptSizeIncrease()`. For reductions that don't accept an increase, the
  /// module size has to decrease for them to be considered useful, which
  /// prevents the tool from getting stuck at a local point where the reduction
  /// applies but produces no change in the input. However, reductions that *do*
  /// accept a size increase can get stuck in this local fixed-point as they
  /// keep applying to the same operations and the tool keeps accepting the
  /// unmodified input as an improvement.
  virtual bool isOneShot() const { return false; }

  /// An optional callback for reductions to communicate removal of operations.
  std::function<void(Operation *)> notifyOpErasedCallback = nullptr;

  void notifyOpErased(Operation *op) {
    if (notifyOpErasedCallback)
      notifyOpErasedCallback(op);
  }
};

template <typename OpTy>
struct OpReduction : public Reduction {
  uint64_t match(Operation *op) override {
    if (auto concreteOp = dyn_cast<OpTy>(op))
      return match(concreteOp);
    return 0;
  }
  LogicalResult rewrite(Operation *op) override {
    return rewrite(cast<OpTy>(op));
  }

  virtual uint64_t match(OpTy op) { return 1; }
  virtual LogicalResult rewrite(OpTy op) = 0;
};

/// A reduction pattern that applies an `mlir::Pass`.
struct PassReduction : public Reduction {
  PassReduction(MLIRContext *context, std::unique_ptr<Pass> pass,
                bool canIncreaseSize = false, bool oneShot = false);
  uint64_t match(Operation *op) override;
  LogicalResult rewrite(Operation *op) override;
  std::string getName() const override;
  bool acceptSizeIncrease() const override { return canIncreaseSize; }
  bool isOneShot() const override { return oneShot; }

protected:
  MLIRContext *const context;
  std::unique_ptr<mlir::PassManager> pm;
  StringRef passName;
  bool canIncreaseSize;
  bool oneShot;
};

class ReducePatternSet {
public:
  template <typename R, unsigned Benefit, typename... Args>
  void add(Args &&...args) {
    reducePatternsWithBenefit.push_back(
        {std::make_unique<R>(std::forward<Args>(args)...), Benefit});
  }

  void filter(const std::function<bool(const Reduction &)> &pred);
  void sortByBenefit();
  size_t size() const;

  Reduction &operator[](size_t idx) const;

private:
  SmallVector<std::pair<std::unique_ptr<Reduction>, unsigned>>
      reducePatternsWithBenefit;
};

/// A dialect interface to provide reduction patterns to a reducer tool.
struct ReducePatternDialectInterface
    : public mlir::DialectInterface::Base<ReducePatternDialectInterface> {
  ReducePatternDialectInterface(Dialect *dialect) : Base(dialect) {}

  virtual void populateReducePatterns(ReducePatternSet &patterns) const = 0;
};

struct ReducePatternInterfaceCollection
    : public mlir::DialectInterfaceCollection<ReducePatternDialectInterface> {
  using Base::Base;

  // Collect the reduce patterns defined by each dialect.
  void populateReducePatterns(ReducePatternSet &patterns) const;
};

} // namespace circt

#endif // CIRCT_REDUCE_REDUCTION_H
