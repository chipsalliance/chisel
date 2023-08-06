//===- InnerSymbolTable.h - Inner Symbol Table -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the InnerSymbolTable and related classes, used for
// managing and tracking "inner symbols".
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_INNERSYMBOLTABLE_H
#define CIRCT_DIALECT_HW_INNERSYMBOLTABLE_H

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace hw {

/// The target of an inner symbol, the entity the symbol is a handle for.
class InnerSymTarget {
public:
  /// Default constructor, invalid.
  /* implicit */ InnerSymTarget() { assert(!*this); }

  /// Target an operation, and optionally a field (=0 means the op itself).
  explicit InnerSymTarget(Operation *op, size_t fieldID = 0)
      : op(op), portIdx(invalidPort), fieldID(fieldID) {}

  /// Target a port, and optionally a field (=0 means the port itself).
  /// Operation should be an FModuleLike.
  explicit InnerSymTarget(size_t portIdx, Operation *op, size_t fieldID = 0)
      : op(op), portIdx(portIdx), fieldID(fieldID) {}

  InnerSymTarget(const InnerSymTarget &) = default;
  InnerSymTarget(InnerSymTarget &&) = default;

  // Accessors:

  /// Return the target's fieldID.
  auto getField() const { return fieldID; }

  /// Return the target's base operation.  For ports, this is the module.
  Operation *getOp() const { return op; }

  /// Return the target's port, if valid.  Check "isPort()".
  auto getPort() const {
    assert(isPort());
    return portIdx;
  }

  // Classification:

  /// Return if this targets a field (nonzero fieldID).
  bool isField() const { return fieldID != 0; }

  /// Return if this targets a port.
  bool isPort() const { return portIdx != invalidPort; }

  /// Returns if this targets an operation only (not port or field).
  bool isOpOnly() const { return !isPort() && !isField(); }

  /// Return a target to the specified field within the given base.
  /// FieldID is relative to the specified base target.
  static InnerSymTarget getTargetForSubfield(const InnerSymTarget &base,
                                             size_t fieldID) {
    if (base.isPort())
      return InnerSymTarget(base.portIdx, base.op, base.fieldID + fieldID);
    return InnerSymTarget(base.op, base.fieldID + fieldID);
  }

private:
  auto asTuple() const { return std::tie(op, portIdx, fieldID); }
  Operation *op = nullptr;
  size_t portIdx = 0;
  size_t fieldID = 0;
  static constexpr size_t invalidPort = ~size_t{0};

public:
  // Operators are defined below.

  // Comparison operators:
  bool operator==(const InnerSymTarget &rhs) const {
    return asTuple() == rhs.asTuple();
  }

  // Assignment operators:
  InnerSymTarget &operator=(InnerSymTarget &&) = default;
  InnerSymTarget &operator=(const InnerSymTarget &) = default;

  /// Check if this target is valid.
  operator bool() const { return op; }
};

/// A table of inner symbols and their resolutions.
class InnerSymbolTable {
public:
  /// Build an inner symbol table for the given operation.  The operation must
  /// have the InnerSymbolTable trait.
  explicit InnerSymbolTable(Operation *op);

  /// Non-copyable
  InnerSymbolTable(const InnerSymbolTable &) = delete;
  InnerSymbolTable &operator=(const InnerSymbolTable &) = delete;

  // Moveable
  InnerSymbolTable(InnerSymbolTable &&) = default;
  InnerSymbolTable &operator=(InnerSymbolTable &&) = default;

  /// Look up a symbol with the specified name, returning empty InnerSymTarget
  /// if no such name exists. Names never include the @ on them.
  InnerSymTarget lookup(StringRef name) const;
  InnerSymTarget lookup(StringAttr name) const;

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists or doesn't target just an operation.
  Operation *lookupOp(StringRef name) const;
  template <typename T>
  T lookupOp(StringRef name) const {
    return dyn_cast_or_null<T>(lookupOp(name));
  }

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists or doesn't target just an operation.
  Operation *lookupOp(StringAttr name) const;
  template <typename T>
  T lookupOp(StringAttr name) const {
    return dyn_cast_or_null<T>(lookupOp(name));
  }

  /// Get InnerSymbol for an operation.
  static StringAttr getInnerSymbol(Operation *op);

  /// Get InnerSymbol for a target.
  static StringAttr getInnerSymbol(const InnerSymTarget &target);

  /// Return the name of the attribute used for inner symbol names.
  static StringRef getInnerSymbolAttrName() { return "inner_sym"; }

  /// Construct an InnerSymbolTable, checking for verification failure.
  /// Emits diagnostics describing encountered issues.
  static FailureOr<InnerSymbolTable> get(Operation *op);

  using InnerSymCallbackFn =
      llvm::function_ref<LogicalResult(StringAttr, const InnerSymTarget &)>;

  /// Walk the given IST operation and invoke the callback for all encountered
  /// inner symbols.
  /// This variant allows callbacks that return LogicalResult OR void,
  /// and wraps the underlying implementation.
  template <typename FuncTy, typename RetTy = typename std::invoke_result_t<
                                 FuncTy, StringAttr, const InnerSymTarget &>>
  static RetTy walkSymbols(Operation *op, FuncTy &&callback) {
    if constexpr (std::is_void_v<RetTy>)
      return (void)walkSymbols(
          op, InnerSymCallbackFn(
                  [&](StringAttr name, const InnerSymTarget &target) {
                    std::invoke(std::forward<FuncTy>(callback), name, target);
                    return success();
                  }));
    else
      return walkSymbols(
          op, InnerSymCallbackFn([&](StringAttr name,
                                     const InnerSymTarget &target) {
            return std::invoke(std::forward<FuncTy>(callback), name, target);
          }));
  }

  /// Walk the given IST operation and invoke the callback for all encountered
  /// inner symbols.
  /// This variant is the underlying implementation.
  /// If callback returns failure, the walk is aborted and failure is returned.
  /// A successful walk with no failures returns success.
  static LogicalResult walkSymbols(Operation *op, InnerSymCallbackFn callback);

private:
  using TableTy = DenseMap<StringAttr, InnerSymTarget>;
  /// Construct an inner symbol table for the given operation,
  /// with pre-populated table contents.
  explicit InnerSymbolTable(Operation *op, TableTy &&table)
      : innerSymTblOp(op), symbolTable(table){};

  /// This is the operation this table is constructed for, which must have the
  /// InnerSymbolTable trait.
  Operation *innerSymTblOp;

  /// This maps inner symbol names to their targets.
  TableTy symbolTable;
};

/// This class represents a collection of InnerSymbolTable's.
class InnerSymbolTableCollection {
public:
  /// Get or create the InnerSymbolTable for the specified operation.
  InnerSymbolTable &getInnerSymbolTable(Operation *op);

  /// Populate tables in parallel for all InnerSymbolTable operations in the
  /// given InnerRefNamespace operation, verifying each and returning
  /// the verification result.
  LogicalResult populateAndVerifyTables(Operation *innerRefNSOp);

  explicit InnerSymbolTableCollection() = default;
  explicit InnerSymbolTableCollection(Operation *innerRefNSOp) {
    // Caller is not interested in verification, no way to report it upwards.
    auto result = populateAndVerifyTables(innerRefNSOp);
    (void)result;
    assert(succeeded(result));
  }
  InnerSymbolTableCollection(const InnerSymbolTableCollection &) = delete;
  InnerSymbolTableCollection &
  operator=(const InnerSymbolTableCollection &) = delete;

private:
  /// This maps Operations to their InnnerSymbolTable's.
  DenseMap<Operation *, std::unique_ptr<InnerSymbolTable>> symbolTables;
};

/// This class represents the namespace in which InnerRef's can be resolved.
struct InnerRefNamespace {
  SymbolTable &symTable;
  InnerSymbolTableCollection &innerSymTables;

  /// Resolve the InnerRef to its target within this namespace, returning empty
  /// target if no such name exists.
  InnerSymTarget lookup(hw::InnerRefAttr inner);

  /// Resolve the InnerRef to its target within this namespace, returning
  /// empty target if no such name exists or it's not an operation.
  /// Template type can be used to limit results to specified op type.
  Operation *lookupOp(hw::InnerRefAttr inner);
  template <typename T>
  T lookupOp(hw::InnerRefAttr inner) {
    return dyn_cast_or_null<T>(lookupOp(inner));
  }
};

/// Printing InnerSymTarget's.
template <typename OS>
OS &operator<<(OS &os, const InnerSymTarget &target) {
  if (!target)
    return os << "<invalid target>";

  if (target.isField())
    os << "field " << target.getField() << " of ";

  if (target.isPort())
    os << "<port " << target.getPort() << " on @"
       << SymbolTable::getSymbolName(target.getOp()) << ">";
  else
    os << "<op " << *target.getOp() << ">";

  return os;
}

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_INNERSYMBOLTABLE_H
