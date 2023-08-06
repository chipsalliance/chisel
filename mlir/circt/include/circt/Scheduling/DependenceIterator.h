//===- DependenceIterator.h - Uniform handling of dependences ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities to let algorithms iterate over different flavors
// of dependences in a uniform way.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_DEPENDENCEITERATOR_H
#define CIRCT_SCHEDULING_DEPENDENCEITERATOR_H

#include "circt/Support/LLVM.h"

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/iterator.h"

#include <optional>

namespace circt {
namespace scheduling {

class Problem;

namespace detail {

/// A wrapper class to uniformly handle def-use and auxiliary dependence edges.
/// Should be small enough (two pointers) to be passed around by value.
class Dependence {
public:
  /// The "expanded" representation of a dependence, intended as the key for
  /// comparisons and hashing.
  using TupleRepr =
      std::tuple<Operation *, Operation *, std::optional<unsigned>,
                 std::optional<unsigned>>;

  /// Wrap a def-use dependence, which is uniquely identified in the SSA graph
  /// by an `OpOperand`.
  Dependence(OpOperand *defUseDep) : auxSrc(nullptr), defUse(defUseDep) {}
  /// Wrap an auxiliary dependence, identified by the pair of its endpoints.
  Dependence(std::pair<Operation *, Operation *> auxDep)
      : auxSrc(std::get<0>(auxDep)), auxDst(std::get<1>(auxDep)) {}
  /// Wrap an auxiliary dependence between \p from and \p to.
  Dependence(Operation *from, Operation *to) : auxSrc(from), auxDst(to) {}
  /// Construct an invalid dependence.
  Dependence() : auxSrc(nullptr), auxDst(nullptr) {}

  /// Return true if this is a valid auxiliary dependence.
  bool isAuxiliary() const { return auxSrc && auxDst; }
  /// Return true if this is a valid def-use dependence.
  bool isDefUse() const { return !auxSrc && auxDst; }
  /// Return true if this is an invalid dependence.
  bool isInvalid() const { return !auxDst; }

  /// Return the source of the dependence.
  Operation *getSource() const;
  /// Return the destination of the dependence.
  Operation *getDestination() const;

  /// Return the source operation's result number, if applicable.
  std::optional<unsigned> getSourceIndex() const;
  /// Return the destination operation's operand number, if applicable.
  std::optional<unsigned> getDestinationIndex() const;

  /// Return the tuple representation of this dependence.
  TupleRepr getAsTuple() const;

  bool operator==(const Dependence &other) const;

private:
  Operation *auxSrc;
  union {
    Operation *auxDst;
    OpOperand *defUse;
  };
};

/// An iterator to transparently surface an operation's def-use dependences from
/// the SSA subgraph (induced by the registered operations), as well as
/// auxiliary, operation-to-operation dependences explicitly provided by the
/// client.
class DependenceIterator
    : public llvm::iterator_facade_base<DependenceIterator,
                                        std::forward_iterator_tag, Dependence> {
public:
  /// Construct an iterator over the \p op's def-use dependences (i.e. result
  /// values of other operations registered in the scheduling problem, which are
  /// used by one of \p op's operands), and over auxiliary dependences (i.e.
  /// from other operation to \p op).
  DependenceIterator(Problem &problem, Operation *op, bool end = false);

  bool operator==(const DependenceIterator &other) const {
    return dep == other.dep;
  }

  const Dependence &operator*() const { return dep; }

  DependenceIterator &operator++() {
    findNextDependence();
    return *this;
  }

private:
  void findNextDependence();

  Problem &problem;
  Operation *op;

  unsigned operandIdx;
  unsigned auxPredIdx;
  llvm::SmallSetVector<Operation *, 4> *auxPreds;

  Dependence dep;
};

} // namespace detail
} // namespace scheduling
} // namespace circt

namespace llvm {

using circt::scheduling::detail::Dependence;

template <>
struct DenseMapInfo<Dependence> {
  static inline Dependence getEmptyKey() {
    return Dependence(DenseMapInfo<mlir::Operation *>::getEmptyKey(), nullptr);
  }
  static inline Dependence getTombstoneKey() {
    return Dependence(DenseMapInfo<mlir::Operation *>::getTombstoneKey(),
                      nullptr);
  }
  static unsigned getHashValue(const Dependence &val) {
    return llvm::hash_value(val.getAsTuple());
  }
  static bool isEqual(const Dependence &lhs, const Dependence &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

#endif // CIRCT_SCHEDULING_DEPENDENCEITERATOR_H
