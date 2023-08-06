//===- HWTypes.h - Types for the HW dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types for the HW dialect are mostly in tablegen. This file should contain
// C++ types used in MLIR type parameters.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_TYPES_H
#define CIRCT_DIALECT_HW_TYPES_H

#include "circt/Dialect/HW/HWDialect.h"

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace circt {
namespace hw {

struct ModulePort {
  enum Direction { Input, Output, InOut };
  mlir::StringAttr name;
  mlir::Type type;
  Direction dir;
};

class HWSymbolCache;
class ParamDeclAttr;
class TypedeclOp;
class ModuleType;

namespace detail {

ModuleType fnToMod(Operation *op, ArrayAttr inputNames, ArrayAttr outputNames);
ModuleType fnToMod(FunctionType fn, ArrayAttr inputNames,
                   ArrayAttr outputNames);

/// Struct defining a field. Used in structs.
struct FieldInfo {
  mlir::StringAttr name;
  mlir::Type type;
};

/// Struct defining a field with an offset. Used in unions.
struct OffsetFieldInfo {
  StringAttr name;
  Type type;
  size_t offset;
};
} // namespace detail
} // namespace hw
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/HW/HWTypes.h.inc"

namespace circt {
namespace hw {

// Returns the canonical type of a HW type (inner type of a type alias).
mlir::Type getCanonicalType(mlir::Type type);

/// Return true if the specified type is a value HW Integer type.  This checks
/// that it is a signless standard dialect type.
bool isHWIntegerType(mlir::Type type);

/// Return true if the specified type is a HW Enum type.
bool isHWEnumType(mlir::Type type);

/// Return true if the specified type can be used as an HW value type, that is
/// the set of types that can be composed together to represent synthesized,
/// hardware but not marker types like InOutType or unknown types from other
/// dialects.
bool isHWValueType(mlir::Type type);

/// Return the hardware bit width of a type. Does not reflect any encoding,
/// padding, or storage scheme, just the bit (and wire width) of a
/// statically-size type. Reflects the number of wires needed to transmit a
/// value of this type. Returns -1 if the type is not known or cannot be
/// statically computed.
int64_t getBitWidth(mlir::Type type);

/// Return true if the specified type contains known marker types like
/// InOutType.  Unlike isHWValueType, this is not conservative, it only returns
/// false on known InOut types, rather than any unknown types.
bool hasHWInOutType(mlir::Type type);

template <typename... BaseTy>
bool type_isa(Type type) {
  // First check if the type is the requested type.
  if (type.isa<BaseTy...>())
    return true;

  // Then check if it is a type alias wrapping the requested type.
  if (auto alias = type.dyn_cast<TypeAliasType>())
    return type_isa<BaseTy...>(alias.getInnerType());

  return false;
}

// type_isa for a nullable argument.
template <typename... BaseTy>
bool type_isa_and_nonnull(Type type) { // NOLINT(readability-identifier-naming)
  if (!type)
    return false;
  return type_isa<BaseTy...>(type);
}

template <typename BaseTy>
BaseTy type_cast(Type type) {
  assert(type_isa<BaseTy>(type) && "type must convert to requested type");

  // If the type is the requested type, return it.
  if (type.isa<BaseTy>())
    return type.cast<BaseTy>();

  // Otherwise, it must be a type alias wrapping the requested type.
  return type_cast<BaseTy>(type.cast<TypeAliasType>().getInnerType());
}

template <typename BaseTy>
BaseTy type_dyn_cast(Type type) {
  if (!type_isa<BaseTy>(type))
    return BaseTy();

  return type_cast<BaseTy>(type);
}

/// Utility type that wraps a type that may be one of several possible Types.
/// This is similar to std::variant but is implemented for mlir::Type, and it
/// understands how to handle type aliases.
template <typename... Types>
class TypeVariant
    : public ::mlir::Type::TypeBase<TypeVariant<Types...>, mlir::Type,
                                    mlir::TypeStorage> {
  using mlir::Type::TypeBase<TypeVariant<Types...>, mlir::Type,
                             mlir::TypeStorage>::Base::Base;

public:
  // Support LLVM isa/cast/dyn_cast to one of the possible types.
  static bool classof(Type other) { return type_isa<Types...>(other); }
};

template <typename BaseTy>
class TypeAliasOr
    : public ::mlir::Type::TypeBase<TypeAliasOr<BaseTy>, mlir::Type,
                                    mlir::TypeStorage> {
  using mlir::Type::TypeBase<TypeAliasOr<BaseTy>, mlir::Type,
                             mlir::TypeStorage>::Base::Base;

public:
  // Support LLVM isa/cast/dyn_cast to BaseTy.
  static bool classof(Type other) { return type_isa<BaseTy>(other); }

  // Support C++ implicit conversions to BaseTy.
  operator BaseTy() const { return type_cast<BaseTy>(*this); }
};

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_TYPES_H
