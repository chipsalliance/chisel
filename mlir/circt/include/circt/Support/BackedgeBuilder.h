//===- BackedgeBuilder.h - Support for building backedges -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Backedges are operations/values which have to exist as operands before
// they are produced in a result. Since it isn't clear how to build backedges
// in MLIR, these helper classes set up a canonical way to do so.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_BACKEDGEBUILDER_H
#define CIRCT_SUPPORT_BACKEDGEBUILDER_H

#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class OpBuilder;
class PatternRewriter;
class Operation;
} // namespace mlir

namespace circt {

class Backedge;

/// Instantiate one of these and use it to build typed backedges. Backedges
/// which get used as operands must be assigned to with the actual value before
/// this class is destructed, usually at the end of a scope. It will check that
/// invariant then erase all the backedge ops during destruction.
///
/// Example use:
/// ```
///   circt::BackedgeBuilder back(rewriter, loc);
///   circt::Backedge ready = back.get(rewriter.getI1Type());
///   // Use `ready` as a `Value`.
///   auto addOp = rewriter.create<addOp>(loc, ready);
///   // When the actual value is available,
///   ready.set(anotherOp.getResult(0));
/// ```
class BackedgeBuilder {
  friend class Backedge;

public:
  /// To build a backedge op and manipulate it, we need a `PatternRewriter` and
  /// a `Location`. Store them during construct of this instance and use them
  /// when building.
  BackedgeBuilder(mlir::OpBuilder &builder, mlir::Location loc);
  BackedgeBuilder(mlir::PatternRewriter &rewriter, mlir::Location loc);
  ~BackedgeBuilder();

  /// Create a typed backedge. If no location is provided, the one passed to the
  /// constructor will be used.
  Backedge get(mlir::Type resultType, mlir::LocationAttr optionalLoc = {});

  /// Clear the backedges, erasing any remaining cursor ops. Returns `failure`
  /// and emits diagnostic messages if a backedge is still active.
  mlir::LogicalResult clearOrEmitError();

  /// Abandon the backedges, suppressing any diagnostics if they are still
  /// active upon destruction of the backedge builder. Also, any currently
  /// existing cursor ops will be abandoned.
  void abandon();

private:
  mlir::OpBuilder &builder;
  mlir::PatternRewriter *rewriter;
  mlir::Location loc;
  llvm::SmallVector<mlir::Operation *, 16> edges;
};

/// `Backedge` is a wrapper class around a `Value`. When assigned another
/// `Value`, it replaces all uses of itself with the new `Value` then become a
/// wrapper around the new `Value`.
class Backedge {
  friend class BackedgeBuilder;

  /// `Backedge` is constructed exclusively by `BackedgeBuilder`.
  Backedge(mlir::Operation *op);

public:
  Backedge() {}

  explicit operator bool() const { return !!value; }
  operator mlir::Value() const { return value; }
  void setValue(mlir::Value);

private:
  mlir::Value value;
  bool set = false;
};

} // namespace circt

#endif // CIRCT_SUPPORT_BACKEDGEBUILDER_H
