//===- FIRRTLTypes.h - FIRRTL Type System -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type system for the FIRRTL Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_TYPES_H
#define CIRCT_DIALECT_FIRRTL_TYPES_H

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace firrtl {
namespace detail {
struct FIRRTLBaseTypeStorage;
struct WidthTypeStorage;
struct BundleTypeStorage;
struct FVectorTypeStorage;
struct FEnumTypeStorage;
struct CMemoryTypeStorage;
struct RefTypeStorage;
struct BaseTypeAliasStorage;
struct OpenBundleTypeStorage;
struct OpenVectorTypeStorage;
} // namespace detail.

class ClassType;
class ClockType;
class ResetType;
class AsyncResetType;
class SIntType;
class UIntType;
class AnalogType;
class BundleType;
class OpenBundleType;
class OpenVectorType;
class FVectorType;
class FEnumType;
class RefType;
class PropertyType;
class StringType;
class FIntegerType;
class ListType;
class MapType;
class PathType;
class BaseTypeAliasType;

/// A collection of bits indicating the recursive properties of a type.
struct RecursiveTypeProperties {
  /// Whether the type only contains passive elements.
  bool isPassive : 1;
  /// Whether the type contains a reference type.
  bool containsReference : 1;
  /// Whether the type contains an analog type.
  bool containsAnalog : 1;
  /// Whether the type contains a const type.
  bool containsConst : 1;
  /// Whether the type contains a type alias.
  bool containsTypeAlias : 1;
  /// Whether the type has any uninferred bit widths.
  bool hasUninferredWidth : 1;
  /// Whether the type has any uninferred reset.
  bool hasUninferredReset : 1;
};

// This is a common base class for all FIRRTL types.
class FIRRTLType : public Type {
public:
  /// Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return llvm::isa<FIRRTLDialect>(type.getDialect());
  }

  /// Return the recursive properties of the type, containing the `isPassive`,
  /// `containsAnalog`, and `hasUninferredWidth` bits, among others.
  RecursiveTypeProperties getRecursiveTypeProperties() const;

  //===--------------------------------------------------------------------===//
  // Convenience methods for accessing recursive type properties
  //===--------------------------------------------------------------------===//

  /// Returns true if this is or contains a 'const' type.
  bool containsConst() { return getRecursiveTypeProperties().containsConst; }

  /// Return true if this is or contains an Analog type.
  bool containsAnalog() { return getRecursiveTypeProperties().containsAnalog; }

  /// Return true if this is or contains a Reference type.
  bool containsReference() {
    return getRecursiveTypeProperties().containsReference;
  }

  /// Return true if this is an anonymous type (no type alias).
  bool containsTypeAlias() {
    return getRecursiveTypeProperties().containsTypeAlias;
  }

  /// Return true if this type contains an uninferred bit width.
  bool hasUninferredWidth() {
    return getRecursiveTypeProperties().hasUninferredWidth;
  }

  /// Return true if this type contains an uninferred bit reset.
  bool hasUninferredReset() {
    return getRecursiveTypeProperties().hasUninferredReset;
  }

  //===--------------------------------------------------------------------===//
  // Type classifications
  //===--------------------------------------------------------------------===//

  /// Return true if this is a 'ground' type, aka a non-aggregate type.
  bool isGround();

  /// Returns true if this is a 'const' type that can only hold compile-time
  /// constant values
  bool isConst();

protected:
  using Type::Type;
};

// Common base class for all base FIRRTL types.
class FIRRTLBaseType
    : public FIRRTLType::TypeBase<FIRRTLBaseType, FIRRTLType,
                                  detail::FIRRTLBaseTypeStorage> {
public:
  using Base::Base;

  /// Returns true if this is a 'const' type that can only hold compile-time
  /// constant values
  bool isConst();

  /// Return true if this is a "passive" type - one that contains no "flip"
  /// types recursively within itself.
  bool isPassive() const { return getRecursiveTypeProperties().isPassive; }

  /// Return this type with any flip types recursively removed from itself.
  FIRRTLBaseType getPassiveType();

  /// Return this type with any type alias types recursively removed from
  /// itself.
  FIRRTLBaseType getAnonymousType();

  /// Return a 'const' or non-'const' version of this type.
  FIRRTLBaseType getConstType(bool isConst);

  /// Return this type with a 'const' modifiers dropped
  FIRRTLBaseType getAllConstDroppedType();

  /// Return this type with all ground types replaced with UInt<1>.  This is
  /// used for `mem` operations.
  FIRRTLBaseType getMaskType();

  /// Return this type with widths of all ground types removed. This
  /// enables two types to be compared by structure and name ignoring
  /// widths.
  FIRRTLBaseType getWidthlessType();

  /// If this is an IntType, AnalogType, or sugar type for a single bit (Clock,
  /// Reset, etc) then return the bitwidth.  Return -1 if the is one of these
  /// types but without a specified bitwidth.  Return -2 if this isn't a simple
  /// type.
  int32_t getBitWidthOrSentinel();

  /// Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return llvm::isa<FIRRTLDialect>(type.getDialect()) &&
           !llvm::isa<PropertyType, RefType, OpenBundleType, OpenVectorType>(
               type);
  }

  /// Returns true if this is a non-const "passive" that which is not analog.
  bool isRegisterType() {
    return isPassive() && !containsAnalog() && !containsConst();
  }

  /// Return true if this is a valid "reset" type.
  bool isResetType();

  //===--------------------------------------------------------------------===//
  // hw::FieldIDTypeInterface
  //===--------------------------------------------------------------------===//

  /// Get the maximum field ID of this type.  For integers and other ground
  /// types, there are no subfields and the maximum field ID is 0.  For bundle
  /// types and vector types, each field is assigned a field ID in a depth-first
  /// walk order. This function is used to calculate field IDs when this type is
  /// nested under another type.
  uint64_t getMaxFieldID();

  /// Get the sub-type of a type for a field ID, and the subfield's ID. Strip
  /// off a single layer of this type and return the sub-type and a field ID
  /// targeting the same field, but rebased on the sub-type.
  std::pair<circt::hw::FieldIDTypeInterface, uint64_t>
  getSubTypeByFieldID(uint64_t fieldID);

  /// Return the final type targeted by this field ID by recursively walking all
  /// nested aggregate types. This is the identity function for ground types.
  circt::hw::FieldIDTypeInterface getFinalTypeByFieldID(uint64_t fieldID);

  /// Returns the effective field id when treating the index field as the
  /// root of the type.  Essentially maps a fieldID to a fieldID after a
  /// subfield op. Returns the new id and whether the id is in the given
  /// child.
  std::pair<uint64_t, bool> rootChildFieldID(uint64_t fieldID, uint64_t index);
};

/// Returns true if this is a 'const' type whose value is guaranteed to be
/// unchanging at circuit execution time
bool isConst(Type type);

/// Returns true if the type is or contains a 'const' type whose value is
/// guaranteed to be unchanging at circuit execution time
bool containsConst(Type type);

/// Returns whether the two types are equivalent.  This implements the exact
/// definition of type equivalence in the FIRRTL spec.  If the types being
/// compared have any outer flips that encode FIRRTL module directions (input or
/// output), these should be stripped before using this method.
bool areTypesEquivalent(FIRRTLType destType, FIRRTLType srcType,
                        bool destOuterTypeIsConst = false,
                        bool srcOuterTypeIsConst = false,
                        bool requireSameWidths = false);

/// Returns true if two types are weakly equivalent.  See the FIRRTL spec,
/// Section 4.6, for a full definition of this.  Roughly, the oriented types
/// (the types with any flips pushed to the leaves) must match.  This allows for
/// types with flips in different positions to be equivalent.
bool areTypesWeaklyEquivalent(FIRRTLType destType, FIRRTLType srcType,
                              bool destFlip = false, bool srcFlip = false,
                              bool destOuterTypeIsConst = false,
                              bool srcOuterTypeIsConst = false);

/// Returns whether the srcType can be const-casted to the destType.
bool areTypesConstCastable(FIRRTLType destType, FIRRTLType srcType,
                           bool srcOuterTypeIsConst = false);

/// Return true if destination ref type can be cast from source ref type,
/// per FIRRTL spec rules they must be identical or destination has
/// more general versions of the corresponding type in the source.
bool areTypesRefCastable(Type dstType, Type srcType);

/// Returns true if the destination is at least as wide as a source.  The source
/// and destination types must be equivalent non-analog types.  The types are
/// recursively connected to ensure that the destination is larger than the
/// source: ground types are compared on width, vector types are checked
/// recursively based on their elements and bundles are compared
/// field-by-field.  Types with unresolved widths are assumed to fit into or
/// hold their counterparts.
bool isTypeLarger(FIRRTLBaseType dstType, FIRRTLBaseType srcType);

/// Return true if anonymous types of given arguments are equivalent by pointer
/// comparison.
bool areAnonymousTypesEquivalent(FIRRTLBaseType lhs, FIRRTLBaseType rhs);
bool areAnonymousTypesEquivalent(mlir::Type lhs, mlir::Type rhs);

mlir::Type getPassiveType(mlir::Type anyBaseFIRRTLType);

/// Returns true if the given type has some flipped (aka unaligned) dataflow.
/// This will be true if the port contains either bi-directional signals or
/// analog types. Non-HW types (e.g., ref types) are never considered InOut.
bool isTypeInOut(mlir::Type type);

//===----------------------------------------------------------------------===//
// Width Qualified Ground Types
//===----------------------------------------------------------------------===//

/// Trait for types which have a width.
/// Users must implement:
/// ```c++
/// /// Return the width if known, or -1 if unknown.
/// int32_t getWidthOrSentinel();
/// ```
template <typename ConcreteType>
class WidthQualifiedTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, WidthQualifiedTypeTrait> {
public:
  /// Return an optional containing the width, if the width is known (or empty
  /// if width is unknown).
  std::optional<int32_t> getWidth() {
    auto width = static_cast<ConcreteType *>(this)->getWidthOrSentinel();
    if (width < 0)
      return std::nullopt;
    return width;
  }

  /// Return true if this integer type has a known width.
  bool hasWidth() {
    return 0 <= static_cast<ConcreteType *>(this)->getWidthOrSentinel();
  }
};

//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

class SIntType;
class UIntType;

/// This is the common base class between SIntType and UIntType.
class IntType : public FIRRTLBaseType, public WidthQualifiedTypeTrait<IntType> {
public:
  using FIRRTLBaseType::FIRRTLBaseType;

  /// Return an SIntType or UIntType with the specified signedness, width, and
  /// constness.
  static IntType get(MLIRContext *context, bool isSigned,
                     int32_t widthOrSentinel = -1, bool isConst = false);

  bool isSigned() { return isa<SIntType>(); }
  bool isUnsigned() { return isa<UIntType>(); }

  /// Return the width of this type, or -1 if it has none specified.
  int32_t getWidthOrSentinel();

  /// Return a 'const' or non-'const' version of this type.
  IntType getConstType(bool isConst);

  static bool classof(Type type) { return llvm::isa<SIntType, UIntType>(type); }
};

//===----------------------------------------------------------------------===//
// PropertyTypes
//===----------------------------------------------------------------------===//

class PropertyType : public FIRRTLType {
public:
  /// Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return llvm::isa<ClassType, StringType, FIntegerType, ListType, MapType,
                     PathType>(type);
  }

protected:
  using FIRRTLType::FIRRTLType;
};

//===----------------------------------------------------------------------===//
// ClassElement
//===----------------------------------------------------------------------===//

struct ClassElement {
  ClassElement(StringAttr name, Type type, Direction direction)
      : name(name), type(type), direction(direction) {}

  StringAttr name;
  Type type;
  Direction direction;

  StringRef getName() const { return name.getValue(); }

  /// Return true if this is a simple output-only element.  If you want the
  /// direction of the port, use the \p direction field directly.
  bool isInput() const { return direction == Direction::In && !isInOut(); }

  /// Return true if this is a simple input-only element.  If you want the
  /// direction of the port, use the \p direction field directly.
  bool isOutput() const { return direction == Direction::Out && !isInOut(); }

  /// Return true if this is an inout port.  This will be true if the port
  /// contains either bi-directional signals or analog types.
  /// Non-HW types (e.g., ref types) are never considered InOut.
  bool isInOut() const { return isTypeInOut(type); }

  bool operator==(const ClassElement &rhs) const {
    return name == rhs.name && type == rhs.type;
  }

  bool operator!=(const ClassElement &rhs) const { return !(*this == rhs); }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline llvm::hash_code hash_value(const ClassElement &element) {
  return llvm::hash_combine(element.name, element.type, element.direction);
}

//===----------------------------------------------------------------------===//
// Type helpers
//===----------------------------------------------------------------------===//

// Get the bit width for this type, return None  if unknown. Unlike
// getBitWidthOrSentinel(), this can recursively compute the bitwidth of
// aggregate types. For bundle and vectors, recursively get the width of each
// field element and return the total bit width of the aggregate type. This
// returns None, if any of the bundle fields is a flip type, or ground type with
// unknown bit width.
std::optional<int64_t> getBitWidth(FIRRTLBaseType type,
                                   bool ignoreFlip = false);

// Parse a FIRRTL type without a leading `!firrtl.` dialect tag.
ParseResult parseNestedType(FIRRTLType &result, AsmParser &parser);
ParseResult parseNestedBaseType(FIRRTLBaseType &result, AsmParser &parser);
ParseResult parseNestedPropertyType(PropertyType &result, AsmParser &parser);

// Print a FIRRTL type without a leading `!firrtl.` dialect tag.
void printNestedType(Type type, AsmPrinter &os);

using FIRRTLValue = mlir::TypedValue<FIRRTLType>;
using FIRRTLBaseValue = mlir::TypedValue<FIRRTLBaseType>;
using FIRRTLPropertyValue = mlir::TypedValue<PropertyType>;

} // namespace firrtl
} // namespace circt

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h.inc"

namespace llvm {

// Type hash just like pointers.
template <>
struct DenseMapInfo<circt::firrtl::FIRRTLType> {
  using FIRRTLType = circt::firrtl::FIRRTLType;
  static FIRRTLType getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return FIRRTLType(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static FIRRTLType getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return FIRRTLType(static_cast<mlir::Type::ImplType *>(pointer));
  }
  static unsigned getHashValue(FIRRTLType val) { return mlir::hash_value(val); }
  static bool isEqual(FIRRTLType LHS, FIRRTLType RHS) { return LHS == RHS; }
};

} // namespace llvm

namespace circt {
namespace firrtl {
//===--------------------------------------------------------------------===//
// Utility for type aliases
//===--------------------------------------------------------------------===//

/// A struct to check if there is a type derived from FIRRTLBaseType.
/// `ContainAliasableTypes<BaseTy>::value` returns true if `BaseTy` is derived
/// from `FIRRTLBaseType` and not `FIRRTLBaseType` itself, or is not FIRRTL type
/// to cover type interfaces.
template <typename head, typename... tail>
class ContainAliasableTypes {
public:
  static constexpr bool value = ContainAliasableTypes<head>::value ||
                                ContainAliasableTypes<tail...>::value;
};

template <typename BaseTy>
class ContainAliasableTypes<BaseTy> {
  static constexpr bool isFIRRTLBaseType =
      std::is_base_of<FIRRTLBaseType, BaseTy>::value &&
      !std::is_same_v<FIRRTLBaseType, BaseTy>;
  static constexpr bool isFIRRTLType =
      std::is_base_of<FIRRTLType, BaseTy>::value;

public:
  static constexpr bool value = isFIRRTLBaseType || !isFIRRTLType;
};

template <typename... BaseTy>
bool type_isa(Type type) { // NOLINT(readability-identifier-naming)
  // First check if the type is the requested type.
  if (isa<BaseTy...>(type))
    return true;

  // If the requested type is a subtype of FIRRTLBaseType, then check if it is a
  // type alias wrapping the requested type.
  if constexpr (ContainAliasableTypes<BaseTy...>::value) {
    if (auto alias = dyn_cast<BaseTypeAliasType>(type))
      return type_isa<BaseTy...>(alias.getInnerType());
  }

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
BaseTy type_cast(Type type) { // NOLINT(readability-identifier-naming)
  assert(type_isa<BaseTy>(type) && "type must convert to requested type");

  // If the type is the requested type, return it.
  if (isa<BaseTy>(type))
    return cast<BaseTy>(type);

  // Otherwise, it must be a type alias wrapping the requested type.
  if constexpr (ContainAliasableTypes<BaseTy>::value) {
    if (auto alias = dyn_cast<BaseTypeAliasType>(type))
      return type_cast<BaseTy>(alias.getInnerType());
  }

  // Otherwise, it should fail. `cast` should cause a better assertion failure,
  // so just use it.
  return cast<BaseTy>(type);
}

template <typename BaseTy>
BaseTy type_dyn_cast(Type type) { // NOLINT(readability-identifier-naming)
  if (type_isa<BaseTy>(type))
    return type_cast<BaseTy>(type);
  return {};
}

template <typename BaseTy>
BaseTy
type_dyn_cast_or_null(Type type) { // NOLINT(readability-identifier-naming)
  if (type_isa_and_nonnull<BaseTy>(type))
    return type_cast<BaseTy>(type);
  return {};
}

//===--------------------------------------------------------------------===//
// Type alias aware TypeSwitch.
//===--------------------------------------------------------------------===//

/// This class implements the same functionality as TypeSwitch except that
/// it uses firrtl::type_dyn_cast for dynamic cast. llvm::TypeSwitch is not
/// customizable so this class currently duplicates the code.
template <typename T, typename ResultT = void>
class FIRRTLTypeSwitch
    : public llvm::detail::TypeSwitchBase<FIRRTLTypeSwitch<T, ResultT>, T> {
public:
  using BaseT = llvm::detail::TypeSwitchBase<FIRRTLTypeSwitch<T, ResultT>, T>;
  using BaseT::BaseT;
  using BaseT::Case;
  FIRRTLTypeSwitch(FIRRTLTypeSwitch &&other) = default;

  /// Add a case on the given type.
  template <typename CaseT, typename CallableT>
  FIRRTLTypeSwitch<T, ResultT> &
  Case(CallableT &&caseFn) { // NOLINT(readability-identifier-naming)
    if (result)
      return *this;

    // Check to see if CaseT applies to 'value'. Use `type_dyn_cast` here.
    if (auto caseValue = circt::firrtl::type_dyn_cast<CaseT>(this->value))
      result.emplace(caseFn(caseValue));
    return *this;
  }

  /// As a default, invoke the given callable within the root value.
  template <typename CallableT>
  [[nodiscard]] ResultT
  Default(CallableT &&defaultFn) { // NOLINT(readability-identifier-naming)
    if (result)
      return std::move(*result);
    return defaultFn(this->value);
  }

  /// As a default, return the given value.
  [[nodiscard]] ResultT
  Default(ResultT defaultResult) { // NOLINT(readability-identifier-naming)
    if (result)
      return std::move(*result);
    return defaultResult;
  }

  [[nodiscard]] operator ResultT() {
    assert(result && "Fell off the end of a type-switch");
    return std::move(*result);
  }

private:
  /// The pointer to the result of this switch statement, once known,
  /// null before that.
  std::optional<ResultT> result;
};

/// Specialization of FIRRTLTypeSwitch for void returning callables.
template <typename T>
class FIRRTLTypeSwitch<T, void>
    : public llvm::detail::TypeSwitchBase<FIRRTLTypeSwitch<T, void>, T> {
public:
  using BaseT = llvm::detail::TypeSwitchBase<FIRRTLTypeSwitch<T, void>, T>;
  using BaseT::BaseT;
  using BaseT::Case;
  FIRRTLTypeSwitch(FIRRTLTypeSwitch &&other) = default;

  /// Add a case on the given type.
  template <typename CaseT, typename CallableT>
  FIRRTLTypeSwitch<T, void> &
  Case(CallableT &&caseFn) { // NOLINT(readability-identifier-naming)
    if (foundMatch)
      return *this;

    // Check to see if any of the types apply to 'value'.
    if (auto caseValue = circt::firrtl::type_dyn_cast<CaseT>(this->value)) {
      caseFn(caseValue);
      foundMatch = true;
    }
    return *this;
  }

  /// As a default, invoke the given callable within the root value.
  template <typename CallableT>
  void Default(CallableT &&defaultFn) { // NOLINT(readability-identifier-naming)
    if (!foundMatch)
      defaultFn(this->value);
  }

private:
  /// A flag detailing if we have already found a match.
  bool foundMatch = false;
};

template <typename BaseTy>
class BaseTypeAliasOr
    : public ::mlir::Type::TypeBase<BaseTypeAliasOr<BaseTy>,
                                    firrtl::FIRRTLBaseType,
                                    detail::FIRRTLBaseTypeStorage> {

public:
  using mlir::Type::TypeBase<BaseTypeAliasOr<BaseTy>, firrtl::FIRRTLBaseType,
                             detail::FIRRTLBaseTypeStorage>::Base::Base;
  // Support LLVM isa/cast/dyn_cast to BaseTy.
  static bool classof(Type other) { return type_isa<BaseTy>(other); }

  // Support C++ implicit conversions to BaseTy.
  operator BaseTy() const { return circt::firrtl::type_cast<BaseTy>(*this); }

  BaseTy get() const { return circt::firrtl::type_cast<BaseTy>(*this); }
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_TYPES_H
