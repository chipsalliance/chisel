//===- FieldRef.h -  Field References ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines FieldRefs and helpers for them.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_FIELDREF_H
#define CIRCT_SUPPORT_FIELDREF_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace circt {

/// This class represents a reference to a specific field or element of an
/// aggregate value.  Typically, the user will assign a unique field ID to each
/// field in an aggregate type by visiting them in a depth-first pre-order.
///
/// This can be used as the key in a hashtable to store field specific
/// information.
class FieldRef {
public:
  /// Get a null FieldRef.
  FieldRef() {}

  /// Get a FieldRef location for the specified value.
  FieldRef(Value value, unsigned id) : value(value), id(id) {}

  /// Get the Value which created this location.
  Value getValue() const { return value; }

  /// Get the operation which defines this field.  If the field is a block
  /// argument it will return the operation which owns the block.
  Operation *getDefiningOp() const;

  /// Get the operation which defines this field and cast it to the OpTy.
  /// Returns null if the defining operation is of a different type.
  template <typename OpTy>
  OpTy getDefiningOp() const {
    return llvm::dyn_cast<OpTy>(getDefiningOp());
  }

  template <typename... Any>
  bool isa() const {
    auto *op = getDefiningOp();
    assert(op && "isa<> used on a null type.");
    return ::llvm::isa<Any...>(op);
  }

  /// Get the field ID of this FieldRef, which is a unique identifier mapped to
  /// a specific field in a bundle.
  unsigned getFieldID() const { return id; }

  /// Get a reference to a subfield.
  FieldRef getSubField(unsigned subFieldID) const {
    return FieldRef(value, id + subFieldID);
  }

  /// Get the location associated with the value of this field ref.
  Location getLoc() const { return getValue().getLoc(); }

  bool operator==(const FieldRef &other) const {
    return value == other.value && id == other.id;
  }

  bool operator<(const FieldRef &other) const {
    if (value.getImpl() < other.value.getImpl())
      return true;
    if (value.getImpl() > other.value.getImpl())
      return false;
    return id < other.id;
  }

  operator bool() const { return bool(value); }

private:
  /// A pointer to the value which created this.
  Value value;

  /// A unique field ID.
  unsigned id = 0;
};

/// Get a hash code for a FieldRef.
inline ::llvm::hash_code hash_value(const FieldRef &fieldRef) {
  return llvm::hash_combine(fieldRef.getValue(), fieldRef.getFieldID());
}

} // namespace circt

namespace llvm {
/// Allow using FieldRef with DenseMaps.  This hash is based on the Value
/// identity and field ID.
template <>
struct DenseMapInfo<circt::FieldRef> {
  static inline circt::FieldRef getEmptyKey() {
    return circt::FieldRef(DenseMapInfo<mlir::Value>::getEmptyKey(), 0);
  }
  static inline circt::FieldRef getTombstoneKey() {
    return circt::FieldRef(DenseMapInfo<mlir::Value>::getTombstoneKey(), 0);
  }
  static unsigned getHashValue(const circt::FieldRef &val) {
    return circt::hash_value(val);
  }
  static bool isEqual(const circt::FieldRef &lhs, const circt::FieldRef &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // CIRCT_SUPPORT_FIELDREF_H
