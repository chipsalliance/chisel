//===- EmissionPattern.h - Emission Pattern Base and Utility --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This declares the emission pattern base and utility classes.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_TARGET_EXPORTSYSTEMC_EMISSIONPATTERN_H
#define CIRCT_TARGET_EXPORTSYSTEMC_EMISSIONPATTERN_H

#include "EmissionPatternSupport.h"
#include "mlir/IR/Operation.h"
#include <any>

namespace circt {
namespace ExportSystemC {
// Forward declarations.
class EmissionPrinter;

//===----------------------------------------------------------------------===//
// Inline emission utilities.
//===----------------------------------------------------------------------===//

/// This enum encodes the precedence of C++ expressions. A lower number means
/// higher precedence. Source:
/// https://en.cppreference.com/w/cpp/language/operator_precedence
enum class Precedence {
  LIT = 0,
  VAR = 0,
  SCOPE_RESOLUTION = 1,
  POSTFIX_INC = 2,
  POSTFIX_DEC = 2,
  FUNCTIONAL_CAST = 2,
  FUNCTION_CALL = 2,
  SUBSCRIPT = 2,
  MEMBER_ACCESS = 2,
  PREFIX_INC = 3,
  PREFIX_DEC = 3,
  NOT = 3,
  CAST = 3,
  DEREFERENCE = 3,
  ADDRESS_OF = 3,
  SIZEOF = 3,
  NEW = 3,
  DELETE = 3,
  POINTER_TO_MEMBER = 4,
  MUL = 5,
  DIV = 5,
  MOD = 5,
  ADD = 6,
  SUB = 6,
  SHL = 7,
  SHR = 7,
  RELATIONAL = 9,
  EQUALITY = 10,
  BITWISE_AND = 11,
  BITWISE_XOR = 12,
  BITWISE_OR = 13,
  LOGICAL_AND = 14,
  LOGICAL_OR = 15,
  TERNARY = 16,
  THROW = 16,
  ASSIGN = 16,
  COMMA = 17
};

/// This class allows a pattern's match function for inlining to pass its
/// result's precedence to the pattern that requested the expression.
class MatchResult {
public:
  MatchResult() = default;
  MatchResult(Precedence precedence)
      : isFailure(false), precedence(precedence) {}

  bool failed() const { return isFailure; }
  Precedence getPrecedence() const { return precedence; }

private:
  bool isFailure = true;
  Precedence precedence;
};

//===----------------------------------------------------------------------===//
// Emission pattern base classes.
//===----------------------------------------------------------------------===//

/// This is indented to be the base class for all emission patterns.
class PatternBase {
public:
  explicit PatternBase(const void *rootValue) : rootValue(rootValue) {}

  template <typename E, typename... Args>
  static std::unique_ptr<E> create(Args &&...args) {
    std::unique_ptr<E> pattern =
        std::make_unique<E>(std::forward<Args>(args)...);
    return pattern;
  }

  /// Get a unique identifier for the C++ type the pattern is matching on. This
  /// could be a specific MLIR type or operation.
  const void *getRootValue() const { return rootValue; }

private:
  const void *rootValue;
};

/// This is intended to be the base class for all emission patterns matching on
/// operations.
struct OpEmissionPatternBase : PatternBase {
  OpEmissionPatternBase(StringRef operationName, MLIRContext *context)
      : PatternBase(
            OperationName(operationName, context).getAsOpaquePointer()) {}
  virtual ~OpEmissionPatternBase() = default;

  /// Checks if this pattern is applicable to the given value to emit an
  /// inlinable expression. Additionally returns information such as the
  /// precedence to the pattern where this pattern's result is to be inlined.
  virtual MatchResult matchInlinable(Value value) = 0;

  /// Checks if this pattern is applicable to the given operation for statement
  /// emission.
  virtual bool matchStatement(mlir::Operation *op) = 0;

  /// Emit the expression for the given value.
  virtual void emitInlined(mlir::Value value, EmissionPrinter &p) = 0;

  /// Emit zero or more statements for the given operation.
  virtual void emitStatement(mlir::Operation *op, EmissionPrinter &p) = 0;
};

/// This is intended to be the base class for all emission patterns matching on
/// types.
struct TypeEmissionPatternBase : PatternBase {
  explicit TypeEmissionPatternBase(TypeID typeId)
      : PatternBase(typeId.getAsOpaquePointer()) {}
  virtual ~TypeEmissionPatternBase() = default;

  /// Checks if this pattern is applicable to the given type.
  virtual bool match(Type type) = 0;

  /// Emit the given type to the emission printer.
  virtual void emitType(Type type, EmissionPrinter &p) = 0;
};

/// This is intended to be the base class for all emission patterns matching on
/// attributes.
struct AttrEmissionPatternBase : PatternBase {
  explicit AttrEmissionPatternBase(TypeID typeId)
      : PatternBase(typeId.getAsOpaquePointer()) {}
  virtual ~AttrEmissionPatternBase() = default;

  /// Checks if this pattern is applicable to the given attribute.
  virtual bool match(Attribute attr) = 0;

  /// Emit the given attribute to the emission printer.
  virtual void emitAttr(Attribute attr, EmissionPrinter &p) = 0;
};

/// This is a convenience class providing default implementations for operation
/// emission patterns.
template <typename Op>
struct OpEmissionPattern : OpEmissionPatternBase {
  explicit OpEmissionPattern(MLIRContext *context)
      : OpEmissionPatternBase(Op::getOperationName(), context) {}

  void emitStatement(mlir::Operation *op, EmissionPrinter &p) final {
    return emitStatement(cast<Op>(op), p);
  }

  /// Checks if this pattern is applicable to the given value to emit an
  /// inlinable expression. Additionally returns information such as the
  /// precedence to the pattern where this pattern's result is to be inlined.
  /// Defaults to never match.
  MatchResult matchInlinable(Value value) override { return MatchResult(); }

  /// Checks if this pattern is applicable to the given operation for statement
  /// emission. When not overriden this matches on all operations of the type
  /// given as template parameter and emits nothing.
  bool matchStatement(mlir::Operation *op) override { return isa<Op>(op); }

  /// Emit the expression for the given value. This has to be overriden whenever
  /// the 'matchInlinable' function is overriden and emit a valid expression.
  void emitInlined(mlir::Value value, EmissionPrinter &p) override {}

  /// Emit zero (default) or more statements for the given operation.
  virtual void emitStatement(Op op, EmissionPrinter &p) {}
};

/// This is a convenience class providing default implementations for type
/// emission patterns.
template <typename Ty>
struct TypeEmissionPattern : TypeEmissionPatternBase {
  TypeEmissionPattern() : TypeEmissionPatternBase(TypeID::get<Ty>()) {}

  void emitType(Type type, EmissionPrinter &p) final {
    emitType(type.cast<Ty>(), p);
  }

  /// Checks if this pattern is applicable to the given type. Matches to the
  /// type given as template argument by default.
  bool match(Type type) override { return type.isa<Ty>(); }

  /// Emit the given type to the emission printer.
  virtual void emitType(Ty type, EmissionPrinter &p) = 0;
};

/// This is a convenience class providing default implementations for attribute
/// emission patterns.
template <typename A>
struct AttrEmissionPattern : AttrEmissionPatternBase {
  AttrEmissionPattern() : AttrEmissionPatternBase(TypeID::get<A>()) {}

  void emitAttr(Attribute attr, EmissionPrinter &p) final {
    emitAttr(attr.cast<A>(), p);
  }

  /// Checks if this pattern is applicable to the given attribute. Matches to
  /// the attribute given as template argument by default.
  bool match(Attribute attr) override { return attr.isa<A>(); }

  /// Emit the given attribute to the emission printer.
  virtual void emitAttr(A attr, EmissionPrinter &p) = 0;
};

//===----------------------------------------------------------------------===//
// Emission pattern sets.
//===----------------------------------------------------------------------===//

/// This class collects a set of emission patterns with base type 'PatternTy'.
template <typename PatternTy>
class EmissionPatternSet {
public:
  /// Add a new emission pattern that requires additional constructor arguments
  /// to this set.
  template <typename... Es, typename ConstructorArg,
            typename... ConstructorArgs,
            typename = std::enable_if_t<sizeof...(Es) != 0>>
  void add(ConstructorArg &&arg, ConstructorArgs &&...args) {
    (void)std::initializer_list<int>{
        0, (addImpl<Es>(std::forward<ConstructorArg>(arg),
                        std::forward<ConstructorArgs>(args)...),
            0)...};
  }

  /// Add a new emission pattern to the set.
  template <typename... Es, typename = std::enable_if_t<sizeof...(Es) != 0>>
  void add() {
    (void)std::initializer_list<int>{0, (addImpl<Es>(), 0)...};
  }

  /// Get all the emission patterns added to this set.
  std::vector<std::unique_ptr<PatternTy>> &getNativePatterns() {
    return patterns;
  }

private:
  template <typename E, typename... Args>
  std::enable_if_t<std::is_base_of<PatternTy, E>::value>
  addImpl(Args &&...args) {
    std::unique_ptr<E> pattern =
        PatternBase::create<E>(std::forward<Args>(args)...);
    patterns.emplace_back(std::move(pattern));
  }

private:
  std::vector<std::unique_ptr<PatternTy>> patterns;
};

/// This class intends to collect a set of emission patterns in a way to provide
/// fast lookups, but does not allow to add more patterns after construction.
template <typename PatternTy, typename KeyTy>
class FrozenEmissionPatternSet {
  using NativePatternListT = std::vector<std::unique_ptr<PatternTy>>;

public:
  /// A map of type specific native patterns.
  using OpSpecificNativePatternListT =
      DenseMap<KeyTy, std::vector<PatternTy *>>;

  FrozenEmissionPatternSet() : impl(std::make_shared<Impl>()) {}

  /// Freeze the patterns held in `patterns`, and take ownership.
  FrozenEmissionPatternSet(EmissionPatternSet<PatternTy> &&patterns)
      : impl(std::make_shared<Impl>()) {
    for (std::unique_ptr<PatternTy> &pat : patterns.getNativePatterns()) {
      impl->nativeOpSpecificPatternMap[KeyTy::getFromOpaquePointer(
                                           pat->getRootValue())]
          .push_back(pat.get());
      impl->nativeOpSpecificPatternList.push_back(std::move(pat));
    }
  }

  /// Return the native patterns held by this set.
  const OpSpecificNativePatternListT &getSpecificNativePatterns() const {
    return impl->nativeOpSpecificPatternMap;
  }

private:
  /// The internal implementation of the frozen pattern set.
  struct Impl {
    /// The set of emission patterns that are matched to specific kinds.
    OpSpecificNativePatternListT nativeOpSpecificPatternMap;

    /// The full native rewrite list. This allows for the map above
    /// to contain duplicate patterns, e.g. for interfaces and traits.
    NativePatternListT nativeOpSpecificPatternList;
  };

  /// A pointer to the internal pattern list. This uses a shared_ptr to avoid
  /// the need to compile the same pattern list multiple times. For example,
  /// during multi-threaded pass execution, all copies of a pass can share the
  /// same pattern list.
  std::shared_ptr<Impl> impl;
};

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_EMISSIONPATTERN_H
