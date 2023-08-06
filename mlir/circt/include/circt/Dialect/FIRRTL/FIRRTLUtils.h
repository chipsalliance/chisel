//===- FIRRTLUtils.h - FIRRTL IR Utilities ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various utilties to help generate and process FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Parallel.h"

namespace circt {
namespace firrtl {
/// Emit a connect between two values.
void emitConnect(OpBuilder &builder, Location loc, Value lhs, Value rhs);
void emitConnect(ImplicitLocOpBuilder &builder, Value lhs, Value rhs);

/// Utiility for generating a constant attribute.
IntegerAttr getIntAttr(Type type, const APInt &value);

/// Utility for generating a constant zero attribute.
IntegerAttr getIntZerosAttr(Type type);

/// Utility for generating a constant all ones attribute.
IntegerAttr getIntOnesAttr(Type type);

/// Return the single assignment to a Property value.
PropAssignOp getPropertyAssignment(FIRRTLPropertyValue value);

/// Return the module-scoped driver of a value only looking through one connect.
Value getDriverFromConnect(Value val);

/// Return the value that drives another FIRRTL value within module scope.  This
/// is parameterized by looking through or not through certain constructs.
Value getValueSource(Value val, bool lookThroughWires, bool lookThroughNodes,
                     bool lookThroughCasts);

/// Return the value that drives another FIRRTL value within module scope.  This
/// is parameterized by looking through or not through certain constructs.  This
/// assumes a single driver and should only be run after `ExpandWhens`.
Value getModuleScopedDriver(Value val, bool lookThroughWires,
                            bool lookThroughNodes, bool lookThroughCasts);

/// Return true if a value is module-scoped driven by a value of a specific
/// type.
template <typename A, typename... B>
static bool isModuleScopedDrivenBy(Value val, bool lookThroughWires,
                                   bool lookThroughNodes,
                                   bool lookThroughCasts) {
  val = getModuleScopedDriver(val, lookThroughWires, lookThroughNodes,
                              lookThroughCasts);

  if (!val)
    return false;

  auto *op = val.getDefiningOp();
  if (!op)
    return false;

  return isa<A, B...>(op);
}

/// Walk all the drivers of a value, passing in the connect operations drive the
/// value. If the value is an aggregate it will find connects to subfields. If
/// the callback returns false, this function will stop walking.  Returns false
/// if walking was broken, and true otherwise.
using WalkDriverCallback =
    llvm::function_ref<bool(const FieldRef &dst, const FieldRef &src)>;
bool walkDrivers(FIRRTLBaseValue value, bool lookThroughWires,
                 bool lookThroughNodes, bool lookThroughCasts,
                 WalkDriverCallback callback);

/// Get the FieldRef from a value.  This will travel backwards to through the
/// IR, following Subfield and Subindex to find the op which declares the
/// location.
FieldRef getFieldRefFromValue(Value value);

/// Get a string identifier representing the FieldRef.  Return this string and a
/// boolean indicating if a valid "root" for the identifier was found.  If
/// nameSafe is true, this will generate a string that is better suited for
/// naming something in the IR.  E.g., if the fieldRef is a subfield of a
/// subindex, without name safe the output would be:
///
///   foo[42].bar
///
/// With nameSafe, this would be:
///
///   foo_42_bar
std::pair<std::string, bool> getFieldName(const FieldRef &fieldRef,
                                          bool nameSafe = false);

Value getValueByFieldID(ImplicitLocOpBuilder builder, Value value,
                        unsigned fieldID);

/// Walk leaf ground types in the `firrtlType` and apply the function `fn`.
/// The first argument of `fn` is field ID, and the second argument is a
/// leaf ground type.
void walkGroundTypes(FIRRTLType firrtlType,
                     llvm::function_ref<void(uint64_t, FIRRTLBaseType)> fn);
//===----------------------------------------------------------------------===//
// Inner symbol and InnerRef helpers.
//===----------------------------------------------------------------------===//

/// Returns an inner symbol identifier for the specified target (op or port),
/// adding one if necessary.
StringAttr
getOrAddInnerSym(const hw::InnerSymTarget &target,
                 llvm::function_ref<ModuleNamespace &(FModuleOp)> getNamespace);

/// Obtain an inner reference to the target (operation or port),
/// adding an inner symbol as necessary.
hw::InnerRefAttr
getInnerRefTo(const hw::InnerSymTarget &target,
              llvm::function_ref<ModuleNamespace &(FModuleOp)> getNamespace);

/// Returns an inner symbol identifier for the specified operation, adding one
/// if necessary.
static inline StringAttr getOrAddInnerSym(
    Operation *op,
    llvm::function_ref<ModuleNamespace &(FModuleOp)> getNamespace) {
  return getOrAddInnerSym(hw::InnerSymTarget(op), getNamespace);
}
/// Returns an inner symbol identifier for the specified operation's field
/// adding one if necessary.
static inline StringAttr getOrAddInnerSym(
    Operation *op, uint64_t fieldID,
    llvm::function_ref<ModuleNamespace &(FModuleOp)> getNamespace) {
  return getOrAddInnerSym(hw::InnerSymTarget(op, fieldID), getNamespace);
}

/// Obtain an inner reference to an operation, possibly adding an inner symbol.
static inline hw::InnerRefAttr
getInnerRefTo(Operation *op,
              llvm::function_ref<ModuleNamespace &(FModuleOp)> getNamespace) {
  return getInnerRefTo(hw::InnerSymTarget(op), getNamespace);
}

/// Obtain an inner reference to an operation's field, possibly adding an inner
/// symbol.
static inline hw::InnerRefAttr
getInnerRefTo(Operation *op, uint64_t fieldID,
              llvm::function_ref<ModuleNamespace &(FModuleOp)> getNamespace) {
  return getInnerRefTo(hw::InnerSymTarget(op, fieldID), getNamespace);
}

/// Returns an inner symbol identifier for the specified port, adding one if
/// necessary.
static inline StringAttr getOrAddInnerSym(
    FModuleLike mod, size_t portIdx,
    llvm::function_ref<ModuleNamespace &(FModuleLike)> getNamespace) {
  return getOrAddInnerSym(hw::InnerSymTarget(portIdx, mod), getNamespace);
}

/// Returns an inner symbol identifier for the specified port's field, adding
/// one if necessary.
static inline StringAttr getOrAddInnerSym(
    FModuleLike mod, size_t portIdx, uint64_t fieldID,
    llvm::function_ref<ModuleNamespace &(FModuleLike)> getNamespace) {
  return getOrAddInnerSym(hw::InnerSymTarget(portIdx, mod, fieldID),
                          getNamespace);
}

/// Obtain an inner reference to a port, possibly adding an inner symbol.
static inline hw::InnerRefAttr
getInnerRefTo(FModuleLike mod, size_t portIdx,
              llvm::function_ref<ModuleNamespace &(FModuleLike)> getNamespace) {
  return getInnerRefTo(hw::InnerSymTarget(portIdx, mod), getNamespace);
}

/// Obtain an inner reference to a port's field, possibly adding an inner
/// symbol.
static inline hw::InnerRefAttr
getInnerRefTo(FModuleLike mod, size_t portIdx, uint64_t fieldID,
              llvm::function_ref<ModuleNamespace &(FModuleLike)> getNamespace) {
  return getInnerRefTo(hw::InnerSymTarget(portIdx, mod, fieldID), getNamespace);
}

//===----------------------------------------------------------------------===//
// Type utilities
//===----------------------------------------------------------------------===//

/// If it is a base type, return it as is. If reftype, return wrapped base type.
/// Otherwise, return null.
inline FIRRTLBaseType getBaseType(Type type) {
  return TypeSwitch<Type, FIRRTLBaseType>(type)
      .Case<FIRRTLBaseType>([](auto base) { return base; })
      .Case<RefType>([](auto ref) { return ref.getType(); })
      .Default([](Type type) { return nullptr; });
}

/// Get base type if isa<> the requested type, else null.
template <typename T>
inline T getBaseOfType(Type type) {
  return dyn_cast_or_null<T>(getBaseType(type));
}

/// Return a FIRRTLType with its base type component mutated by the given
/// function. (i.e., ref<T> -> ref<f(T)> and T -> f(T)).
inline FIRRTLType mapBaseType(FIRRTLType type,
                              function_ref<FIRRTLBaseType(FIRRTLBaseType)> fn) {
  return TypeSwitch<FIRRTLType, FIRRTLType>(type)
      .Case<FIRRTLBaseType>([&](auto base) { return fn(base); })
      .Case<RefType>([&](auto ref) {
        return RefType::get(fn(ref.getType()), ref.getForceable());
      });
}

/// Return a FIRRTLType with its base type component mutated by the given
/// function. Return null when the function returns null.
/// (i.e., ref<T> -> ref<f(T)> if f(T) != null else null, and T -> f(T)).
inline FIRRTLType
mapBaseTypeNullable(FIRRTLType type,
                    function_ref<FIRRTLBaseType(FIRRTLBaseType)> fn) {
  return TypeSwitch<FIRRTLType, FIRRTLType>(type)
      .Case<FIRRTLBaseType>([&](auto base) { return fn(base); })
      .Case<RefType>([&](auto ref) -> FIRRTLType {
        auto result = fn(ref.getType());
        if (!result)
          return {};
        return RefType::get(result, ref.getForceable());
      });
}

/// Given a type, return the corresponding lowered type for the HW dialect.
/// Non-FIRRTL types are simply passed through. This returns a null type if it
/// cannot be lowered. The optional function is required to specify how to lower
/// AliasTypes.
Type lowerType(
    Type type, std::optional<Location> loc = {},
    llvm::function_ref<hw::TypeAliasType(Type, BaseTypeAliasType, Location)>
        getTypeDeclFn = {});

//===----------------------------------------------------------------------===//
// Parser-related utilities
//
// These cannot always be relegated to the parser and sometimes need to be
// available for passes.  This has specifically come up for Annotation lowering
// where there is FIRRTL stuff that needs to be parsed out of an annotation.
//===----------------------------------------------------------------------===//

/// Parse a string that may encode a FIRRTL location into a LocationAttr.
std::pair<bool, std::optional<mlir::LocationAttr>> maybeStringToLocation(
    StringRef spelling, bool skipParsing, StringAttr &locatorFilenameCache,
    FileLineColLoc &fileLineColLocCache, MLIRContext *context);

//===----------------------------------------------------------------------===//
// Parallel utilities
//===----------------------------------------------------------------------===//

/// Wrapper for llvm::parallelTransformReduce that performs the transform_reduce
/// serially when MLIR multi-threading is disabled.
/// Does not add a ParallelDiagnosticHandler like mlir::parallelFor.
template <class IterTy, class ResultTy, class ReduceFuncTy,
          class TransformFuncTy>
static ResultTy transformReduce(MLIRContext *context, IterTy begin, IterTy end,
                                ResultTy init, ReduceFuncTy reduce,
                                TransformFuncTy transform) {
  // Parallel when enabled
  if (context->isMultithreadingEnabled())
    return llvm::parallelTransformReduce(begin, end, init, reduce, transform);

  // Serial fallback (from llvm::parallelTransformReduce)
  for (IterTy i = begin; i != end; ++i)
    init = reduce(std::move(init), transform(*i));
  return std::move(init);
}

/// Range wrapper
template <class RangeTy, class ResultTy, class ReduceFuncTy,
          class TransformFuncTy>
static ResultTy transformReduce(MLIRContext *context, RangeTy &&r,
                                ResultTy init, ReduceFuncTy reduce,
                                TransformFuncTy transform) {
  return transformReduce(context, std::begin(r), std::end(r), init, reduce,
                         transform);
}

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLUTILS_H
