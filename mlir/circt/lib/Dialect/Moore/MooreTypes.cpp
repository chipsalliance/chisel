//===- MooreTypes.cpp - Implement the Moore types -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the Moore dialect type system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreTypes.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::moore;
using mlir::DialectAsmParser;
using mlir::DialectAsmPrinter;
using mlir::LocationAttr;
using mlir::OptionalParseResult;
using mlir::StringSwitch;
using mlir::TypeStorage;
using mlir::TypeStorageAllocator;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Moore/MooreTypes.cpp.inc"

void MooreDialect::registerTypes() {
  addTypes<VoidType, StringType, ChandleType, EventType, IntType, RealType,
           PackedNamedType, PackedRefType, UnpackedNamedType, UnpackedRefType,
           PackedUnsizedDim, PackedRangeDim, UnpackedUnsizedDim,
           UnpackedArrayDim, UnpackedRangeDim, UnpackedAssocDim,
           UnpackedQueueDim, EnumType, PackedStructType, UnpackedStructType,
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Moore/MooreTypes.cpp.inc"
           >();
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

StringRef moore::getKeywordFromSign(const Sign &sign) {
  switch (sign) {
  case Sign::Unsigned:
    return "unsigned";
  case Sign::Signed:
    return "signed";
  }
  llvm_unreachable("all signs should be handled");
}

std::optional<Sign> moore::getSignFromKeyword(StringRef keyword) {
  return StringSwitch<std::optional<Sign>>(keyword)
      .Case("unsigned", Sign::Unsigned)
      .Case("signed", Sign::Signed)
      .Default({});
}

//===----------------------------------------------------------------------===//
// Simple Bit Vector Type
//===----------------------------------------------------------------------===//

PackedType SimpleBitVectorType::getType(MLIRContext *context) const {
  if (!*this)
    return {};
  std::optional<Sign> maybeSign;
  if (explicitSign)
    maybeSign = sign;

  // If the type originally used an integer atom, try to reconstruct that.
  if (usedAtom)
    if (auto kind = IntType::getKindFromDomainAndSize(domain, size))
      return IntType::get(context, *kind, maybeSign);

  // Build the core integer bit type.
  auto kind = domain == Domain::TwoValued ? IntType::Bit : IntType::Logic;
  auto intType = IntType::get(context, kind, maybeSign);

  // If the vector is wider than a single bit, or the dimension was explicit in
  // the original type, add a dimension around the bit type.
  if (size > 1 || explicitSize)
    return PackedRangeDim::get(intType, size);
  return intType;
}

//===----------------------------------------------------------------------===//
// Unpacked Type
//===----------------------------------------------------------------------===//

UnpackedType UnpackedType::resolved() const {
  return TypeSwitch<UnpackedType, UnpackedType>(*this)
      .Case<PackedType, UnpackedIndirectType, UnpackedDim>(
          [&](auto type) { return type.resolved(); })
      .Default([](auto type) { return type; });
}

UnpackedType UnpackedType::fullyResolved() const {
  return TypeSwitch<UnpackedType, UnpackedType>(*this)
      .Case<PackedType, UnpackedIndirectType, UnpackedDim>(
          [&](auto type) { return type.fullyResolved(); })
      .Default([](auto type) { return type; });
}

Domain UnpackedType::getDomain() const {
  return TypeSwitch<UnpackedType, Domain>(*this)
      .Case<PackedType>([](auto type) { return type.getDomain(); })
      .Case<UnpackedIndirectType, UnpackedDim>(
          [&](auto type) { return type.getInner().getDomain(); })
      .Case<UnpackedStructType>(
          [](auto type) { return type.getStruct().domain; })
      .Default([](auto) { return Domain::TwoValued; });
}

Sign UnpackedType::getSign() const {
  return TypeSwitch<UnpackedType, Sign>(*this)
      .Case<PackedType>([](auto type) { return type.getSign(); })
      .Case<UnpackedIndirectType, UnpackedDim>(
          [&](auto type) { return type.getInner().getSign(); })
      .Default([](auto) { return Sign::Unsigned; });
}

std::optional<unsigned> UnpackedType::getBitSize() const {
  return TypeSwitch<UnpackedType, std::optional<unsigned>>(*this)
      .Case<PackedType, RealType>([](auto type) { return type.getBitSize(); })
      .Case<UnpackedUnsizedDim>([](auto) { return std::nullopt; })
      .Case<UnpackedArrayDim>([](auto type) -> std::optional<unsigned> {
        if (auto size = type.getInner().getBitSize())
          return (*size) * type.getSize();
        return {};
      })
      .Case<UnpackedRangeDim>([](auto type) -> std::optional<unsigned> {
        if (auto size = type.getInner().getBitSize())
          return (*size) * type.getRange().size;
        return {};
      })
      .Case<UnpackedIndirectType>(
          [](auto type) { return type.getInner().getBitSize(); })
      .Case<UnpackedStructType>(
          [](auto type) { return type.getStruct().bitSize; })
      .Default([](auto) { return std::nullopt; });
}

/// Map an `IntType` to the corresponding SBVT. Never returns a null type.
static SimpleBitVectorType getSimpleBitVectorFromIntType(IntType type) {
  auto bitSize = type.getBitSize();
  bool usedAtom = bitSize > 1;
  return SimpleBitVectorType(type.getDomain(), type.getSign(), bitSize,
                             usedAtom, type.isSignExplicit(), false);
}

SimpleBitVectorType UnpackedType::getSimpleBitVectorOrNull() const {
  return TypeSwitch<UnpackedType, SimpleBitVectorType>(fullyResolved())
      .Case<IntType>([](auto type) {
        // Integer types trivially map to SBVTs.
        return getSimpleBitVectorFromIntType(type);
      })
      .Case<PackedRangeDim>([](auto rangeType) {
        // Inner type must be an integer.
        auto innerType =
            rangeType.getInner().fullyResolved().template dyn_cast<IntType>();
        if (!innerType)
          return SimpleBitVectorType{};

        // Inner type must be a single-bit integer. Cannot have integer atom
        // vectors like `int [31:0]`.
        auto sbv = getSimpleBitVectorFromIntType(innerType);
        if (sbv.usedAtom)
          return SimpleBitVectorType{};

        // Range must be have non-zero size, and go downwards to zero.
        auto range = rangeType.getRange();
        if (range.size == 0 || range.offset != 0 || range.dir != RangeDir::Down)
          return SimpleBitVectorType{};
        sbv.explicitSize = true;
        sbv.size = range.size;
        return sbv;
      })
      .Default([](auto) { return SimpleBitVectorType{}; });
}

SimpleBitVectorType UnpackedType::castToSimpleBitVectorOrNull() const {
  // If the type is already a valid SBVT, return that immediately without
  // casting.
  if (auto sbv = getSimpleBitVectorOrNull())
    return sbv;

  // All packed types with a known size (i.e., with no `[]` dimensions) can be
  // cast to an SBVT.
  auto packed = fullyResolved().dyn_cast<PackedType>();
  if (!packed)
    return {};
  auto bitSize = packed.getBitSize();
  if (!bitSize || *bitSize == 0)
    return {};

  return SimpleBitVectorType(packed.getDomain(), packed.getSign(), *bitSize,
                             /*usedAtom=*/false, /*explicitSign=*/false,
                             /*explicitSize=*/false);
}

void UnpackedType::format(
    llvm::raw_ostream &os,
    llvm::function_ref<void(llvm::raw_ostream &os)> around) const {
  TypeSwitch<UnpackedType>(*this)
      .Case<StringType>([&](auto) { os << "string"; })
      .Case<ChandleType>([&](auto) { os << "chandle"; })
      .Case<EventType>([&](auto) { os << "event"; })
      .Case<RealType>([&](auto type) { os << type.getKeyword(); })
      .Case<PackedType, UnpackedStructType>([&](auto type) { type.format(os); })
      .Case<UnpackedDim>([&](auto type) { type.format(os, around); })
      .Case<UnpackedNamedType>(
          [&](auto type) { os << type.getName().getValue(); })
      .Case<UnpackedRefType>(
          [&](auto type) { os << "type(" << type.getInner() << ")"; })
      .Default([](auto) { llvm_unreachable("all types should be handled"); });

  // In case there were no unpacked dimensions, the `around` function was never
  // called. However, callers expect us to be able to format things like `bit
  // [7:0] fieldName`, where `fieldName` would be printed by `around`. So in
  // case `around` is non-null, but no unpacked dimension had a chance to print
  // it, simply print it now.
  if (!isa<UnpackedDim>() && around) {
    os << " ";
    around(os);
  }
}

//===----------------------------------------------------------------------===//
// Packed Type
//===----------------------------------------------------------------------===//

PackedType PackedType::resolved() const {
  return TypeSwitch<PackedType, PackedType>(*this)
      .Case<PackedIndirectType, PackedDim>(
          [&](auto type) { return type.resolved(); })
      .Default([](auto type) { return type; });
}

PackedType PackedType::fullyResolved() const {
  return TypeSwitch<PackedType, PackedType>(*this)
      .Case<PackedIndirectType, PackedDim>(
          [&](auto type) { return type.fullyResolved(); })
      .Default([](auto type) { return type; });
}

Domain PackedType::getDomain() const {
  return TypeSwitch<PackedType, Domain>(*this)
      .Case<VoidType>([](auto) { return Domain::TwoValued; })
      .Case<IntType>([&](auto type) { return type.getDomain(); })
      .Case<PackedIndirectType, PackedDim>(
          [&](auto type) { return type.getInner().getDomain(); })
      .Case<EnumType>([](auto type) { return type.getBase().getDomain(); })
      .Case<PackedStructType>(
          [](auto type) { return type.getStruct().domain; });
}

Sign PackedType::getSign() const {
  return TypeSwitch<PackedType, Sign>(*this)
      .Case<VoidType>([](auto) { return Sign::Unsigned; })
      .Case<IntType, PackedStructType>(
          [&](auto type) { return type.getSign(); })
      .Case<PackedIndirectType, PackedDim>(
          [&](auto type) { return type.getInner().getSign(); })
      .Case<EnumType>([](auto type) { return type.getBase().getSign(); });
}

std::optional<unsigned> PackedType::getBitSize() const {
  return TypeSwitch<PackedType, std::optional<unsigned>>(*this)
      .Case<VoidType>([](auto) { return 0; })
      .Case<IntType>([](auto type) { return type.getBitSize(); })
      .Case<PackedUnsizedDim>([](auto) { return std::nullopt; })
      .Case<PackedRangeDim>([](auto type) -> std::optional<unsigned> {
        if (auto size = type.getInner().getBitSize())
          return (*size) * type.getRange().size;
        return {};
      })
      .Case<PackedIndirectType>(
          [](auto type) { return type.getInner().getBitSize(); })
      .Case<EnumType>([](auto type) { return type.getBase().getBitSize(); })
      .Case<PackedStructType>(
          [](auto type) { return type.getStruct().bitSize; });
}

void PackedType::format(llvm::raw_ostream &os) const {
  TypeSwitch<PackedType>(*this)
      .Case<VoidType>([&](auto) { os << "void"; })
      .Case<IntType, PackedRangeDim, PackedUnsizedDim, EnumType,
            PackedStructType>([&](auto type) { type.format(os); })
      .Case<PackedNamedType>(
          [&](auto type) { os << type.getName().getValue(); })
      .Case<PackedRefType>(
          [&](auto type) { os << "type(" << type.getInner() << ")"; })
      .Default([](auto) { llvm_unreachable("all types should be handled"); });
}

//===----------------------------------------------------------------------===//
// Unit Types
//===----------------------------------------------------------------------===//

VoidType VoidType::get(MLIRContext *context) { return Base::get(context); }

StringType StringType::get(MLIRContext *context) { return Base::get(context); }

ChandleType ChandleType::get(MLIRContext *context) {
  return Base::get(context);
}

EventType EventType::get(MLIRContext *context) { return Base::get(context); }

//===----------------------------------------------------------------------===//
// Packed Integers
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {
struct IntTypeStorage : TypeStorage {
  using KeyTy = unsigned;
  using Kind = IntType::Kind;

  IntTypeStorage(KeyTy key)
      : kind(static_cast<Kind>((key >> 16) & 0xFF)),
        sign(static_cast<Sign>((key >> 8) & 0xFF)), explicitSign(key & 1) {}
  static KeyTy pack(Kind kind, Sign sign, bool explicitSign) {
    return static_cast<unsigned>(kind) << 16 |
           static_cast<unsigned>(sign) << 8 | explicitSign;
  }
  bool operator==(const KeyTy &key) const {
    return pack(kind, sign, explicitSign) == key;
  }
  static IntTypeStorage *construct(TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<IntTypeStorage>()) IntTypeStorage(key);
  }

  Kind kind;
  Sign sign;
  bool explicitSign;
};
} // namespace detail
} // namespace moore
} // namespace circt

std::optional<IntType::Kind> IntType::getKindFromKeyword(StringRef keyword) {
  return StringSwitch<std::optional<Kind>>(keyword)
      .Case("bit", IntType::Bit)
      .Case("logic", IntType::Logic)
      .Case("reg", IntType::Reg)
      .Case("byte", IntType::Byte)
      .Case("shortint", IntType::ShortInt)
      .Case("int", IntType::Int)
      .Case("longint", IntType::LongInt)
      .Case("integer", IntType::Integer)
      .Case("time", IntType::Time)
      .Default({});
}

StringRef IntType::getKeyword(Kind kind) {
  switch (kind) {
  case IntType::Bit:
    return "bit";
  case IntType::Logic:
    return "logic";
  case IntType::Reg:
    return "reg";
  case IntType::Byte:
    return "byte";
  case IntType::ShortInt:
    return "shortint";
  case IntType::Int:
    return "int";
  case IntType::LongInt:
    return "longint";
  case IntType::Integer:
    return "integer";
  case IntType::Time:
    return "time";
  }
  llvm_unreachable("all kinds should be handled");
}

Sign IntType::getDefaultSign(Kind kind) {
  switch (kind) {
  case IntType::Bit:
  case IntType::Logic:
  case IntType::Reg:
  case IntType::Time:
    return Sign::Unsigned;
  case IntType::Byte:
  case IntType::ShortInt:
  case IntType::Int:
  case IntType::LongInt:
  case IntType::Integer:
    return Sign::Signed;
  }
  llvm_unreachable("all kinds should be handled");
}

Domain IntType::getDomain(Kind kind) {
  switch (kind) {
  case IntType::Bit:
  case IntType::Byte:
  case IntType::ShortInt:
  case IntType::Int:
  case IntType::LongInt:
  case IntType::Time:
    return Domain::TwoValued;
  case IntType::Logic:
  case IntType::Reg:
  case IntType::Integer:
    return Domain::FourValued;
  }
  llvm_unreachable("all kinds should be handled");
}

unsigned IntType::getBitSize(Kind kind) {
  switch (kind) {
  case IntType::Bit:
  case IntType::Logic:
  case IntType::Reg:
    return 1;
  case IntType::Byte:
    return 8;
  case IntType::ShortInt:
    return 16;
  case IntType::Int:
    return 32;
  case IntType::LongInt:
    return 64;
  case IntType::Integer:
    return 32;
  case IntType::Time:
    return 64;
  }
  llvm_unreachable("all kinds should be handled");
}

std::optional<IntType::Kind> IntType::getKindFromDomainAndSize(Domain domain,
                                                               unsigned size) {
  switch (domain) {
  case Domain::TwoValued:
    switch (size) {
    case 1:
      return IntType::Bit;
    case 8:
      return IntType::Byte;
    case 16:
      return IntType::ShortInt;
    case 32:
      return IntType::Int;
    case 64:
      return IntType::LongInt;
    default:
      return {};
    }
  case Domain::FourValued:
    switch (size) {
    case 1:
      return IntType::Logic;
    case 32:
      return IntType::Integer;
    default:
      return {};
    }
  }
  llvm_unreachable("all domains should be handled");
}

IntType IntType::get(MLIRContext *context, Kind kind,
                     std::optional<Sign> sign) {
  return Base::get(context, detail::IntTypeStorage::pack(
                                kind, sign.value_or(getDefaultSign(kind)),
                                sign.has_value()));
}

IntType::Kind IntType::getKind() const { return getImpl()->kind; }

Sign IntType::getSign() const { return getImpl()->sign; }

bool IntType::isSignExplicit() const { return getImpl()->explicitSign; }

void IntType::format(llvm::raw_ostream &os) const {
  os << getKeyword();
  auto sign = getSign();
  if (isSignExplicit() || sign != getDefaultSign())
    os << " " << sign;
}

//===----------------------------------------------------------------------===//
// Unpacked Reals
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {
struct RealTypeStorage : TypeStorage {
  using KeyTy = unsigned;
  using Kind = RealType::Kind;

  RealTypeStorage(KeyTy key) : kind(static_cast<Kind>(key)) {}
  bool operator==(const KeyTy &key) const {
    return kind == static_cast<Kind>(key);
  }
  static RealTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<RealTypeStorage>()) RealTypeStorage(key);
  }

  Kind kind;
};
} // namespace detail
} // namespace moore
} // namespace circt

std::optional<RealType::Kind> RealType::getKindFromKeyword(StringRef keyword) {
  return StringSwitch<std::optional<Kind>>(keyword)
      .Case("shortreal", ShortReal)
      .Case("real", Real)
      .Case("realtime", RealTime)
      .Default({});
}

StringRef RealType::getKeyword(Kind kind) {
  switch (kind) {
  case ShortReal:
    return "shortreal";
  case Real:
    return "real";
  case RealTime:
    return "realtime";
  }
  llvm_unreachable("all kinds should be handled");
}

unsigned RealType::getBitSize(Kind kind) {
  switch (kind) {
  case ShortReal:
    return 32;
  case Real:
    return 64;
  case RealTime:
    return 64;
  }
  llvm_unreachable("all kinds should be handled");
}

RealType RealType::get(MLIRContext *context, Kind kind) {
  return Base::get(context, static_cast<unsigned>(kind));
}

RealType::Kind RealType::getKind() const { return getImpl()->kind; }

//===----------------------------------------------------------------------===//
// Packed Type Indirections
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {

struct IndirectTypeStorage : TypeStorage {
  using KeyTy = std::tuple<UnpackedType, StringAttr, LocationAttr>;

  IndirectTypeStorage(KeyTy key)
      : IndirectTypeStorage(std::get<0>(key), std::get<1>(key),
                            std::get<2>(key)) {}
  IndirectTypeStorage(UnpackedType inner, StringAttr name, LocationAttr loc)
      : inner(inner), name(name), loc(loc) {}
  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == inner && std::get<1>(key) == name &&
           std::get<2>(key) == loc;
  }
  static IndirectTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<IndirectTypeStorage>())
        IndirectTypeStorage(key);
  }

  UnpackedType inner;
  StringAttr name;
  LocationAttr loc;
};

UnpackedType getIndirectTypeInner(const TypeStorage *impl) {
  return static_cast<const IndirectTypeStorage *>(impl)->inner;
}

Location getIndirectTypeLoc(const TypeStorage *impl) {
  return static_cast<const IndirectTypeStorage *>(impl)->loc;
}

StringAttr getIndirectTypeName(const TypeStorage *impl) {
  return static_cast<const IndirectTypeStorage *>(impl)->name;
}

} // namespace detail
} // namespace moore
} // namespace circt

template <>
PackedNamedType NamedTypeBase<PackedNamedType, PackedIndirectType>::get(
    PackedType inner, StringAttr name, Location loc) {
  return Base::get(inner.getContext(), inner, name, loc);
}

template <>
UnpackedNamedType NamedTypeBase<UnpackedNamedType, UnpackedIndirectType>::get(
    UnpackedType inner, StringAttr name, Location loc) {
  return Base::get(inner.getContext(), inner, name, loc);
}

template <>
PackedRefType
RefTypeBase<PackedRefType, PackedIndirectType>::get(PackedType inner,
                                                    Location loc) {
  return Base::get(inner.getContext(), inner, StringAttr{}, loc);
}

template <>
UnpackedRefType
RefTypeBase<UnpackedRefType, UnpackedIndirectType>::get(UnpackedType inner,
                                                        Location loc) {
  return Base::get(inner.getContext(), inner, StringAttr{}, loc);
}

//===----------------------------------------------------------------------===//
// Packed Dimensions
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {

struct DimStorage : TypeStorage {
  using KeyTy = UnpackedType;

  DimStorage(KeyTy key) : inner(key) {}
  bool operator==(const KeyTy &key) const { return key == inner; }
  static DimStorage *construct(TypeStorageAllocator &allocator,
                               const KeyTy &key) {
    return new (allocator.allocate<DimStorage>()) DimStorage(key);
  }

  // Mutation function to late-initialize the resolved versions of the type.
  LogicalResult mutate(TypeStorageAllocator &allocator,
                       UnpackedType newResolved,
                       UnpackedType newFullyResolved) {
    // Cannot set change resolved types once they've been initialized.
    if (resolved && resolved != newResolved)
      return failure();
    if (fullyResolved && fullyResolved != newFullyResolved)
      return failure();

    // Update the resolved types.
    resolved = newResolved;
    fullyResolved = newFullyResolved;
    return success();
  }

  /// Each dimension type calls this function from its `get` method. The first
  /// argument, `dim`, is set to the type that was constructed by the call to
  /// `Base::get`. If that type has just been created, its `resolved` and
  /// `fullyResolved` fields are not yet set. If that is the case, the
  /// `finalize` method constructs the these resolved types by resolving the
  /// inner type appropriately and wrapping it in the dimension type. These
  /// wrapped types, which are equivalent to the `dim` itself but with the inner
  /// type resolved, are passed to `DimStorage::mutate` which fills in the
  /// `resolved` and `fullyResolved` fields behind a storage lock in the
  /// MLIRContext.
  ///
  /// This has been inspired by https://reviews.llvm.org/D84171.
  template <class ConcreteDim, typename... Args>
  void finalize(ConcreteDim dim, Args... args) const {
    if (resolved && fullyResolved)
      return;
    auto inner = dim.getInner();
    auto newResolved = dim;
    auto newFullyResolved = dim;
    if (inner != inner.resolved())
      newResolved = ConcreteDim::get(inner.resolved(), args...);
    if (inner != inner.fullyResolved())
      newFullyResolved = ConcreteDim::get(inner.fullyResolved(), args...);
    auto result = dim.mutate(newResolved, newFullyResolved);
    (void)result; // Supress warning
    assert(succeeded(result));
  }

  UnpackedType inner;
  UnpackedType resolved;
  UnpackedType fullyResolved;
};

struct UnsizedDimStorage : DimStorage {
  UnsizedDimStorage(KeyTy key) : DimStorage(key) {}
  static UnsizedDimStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<UnsizedDimStorage>()) UnsizedDimStorage(key);
  }
};

struct RangeDimStorage : DimStorage {
  using KeyTy = std::pair<UnpackedType, Range>;

  RangeDimStorage(KeyTy key) : DimStorage(key.first), range(key.second) {}
  bool operator==(const KeyTy &key) const {
    return key.first == inner && key.second == range;
  }
  static RangeDimStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<RangeDimStorage>()) RangeDimStorage(key);
  }

  Range range;
};

} // namespace detail
} // namespace moore
} // namespace circt

PackedType PackedDim::getInner() const {
  return getImpl()->inner.cast<PackedType>();
}

void PackedDim::format(llvm::raw_ostream &os) const {
  SmallVector<PackedDim> dims;
  dims.push_back(*this);
  for (;;) {
    PackedType inner = dims.back().getInner();
    if (auto dim = inner.dyn_cast<PackedDim>()) {
      dims.push_back(dim);
    } else {
      inner.format(os);
      break;
    }
  }
  os << " ";
  for (auto dim : dims) {
    dim.formatDim(os);
  }
}

void PackedDim::formatDim(llvm::raw_ostream &os) const {
  TypeSwitch<PackedDim>(*this)
      .Case<PackedRangeDim>(
          [&](auto dim) { os << "[" << dim.getRange() << "]"; })
      .Case<PackedUnsizedDim>([&](auto dim) { os << "[]"; })
      .Default([&](auto) { llvm_unreachable("unhandled dim type"); });
}

PackedType PackedDim::resolved() const {
  return getImpl()->resolved.cast<PackedType>();
}

PackedType PackedDim::fullyResolved() const {
  return getImpl()->fullyResolved.cast<PackedType>();
}

std::optional<Range> PackedDim::getRange() const {
  if (auto dim = dyn_cast<PackedRangeDim>())
    return dim.getRange();
  return {};
}

std::optional<unsigned> PackedDim::getSize() const {
  return llvm::transformOptional(getRange(), [](auto r) { return r.size; });
}

const detail::DimStorage *PackedDim::getImpl() const {
  return static_cast<detail::DimStorage *>(this->impl);
}

PackedUnsizedDim PackedUnsizedDim::get(PackedType inner) {
  auto type = Base::get(inner.getContext(), inner);
  type.getImpl()->finalize<PackedUnsizedDim>(type);
  return type;
}

PackedRangeDim PackedRangeDim::get(PackedType inner, Range range) {
  auto type = Base::get(inner.getContext(), inner, range);
  type.getImpl()->finalize<PackedRangeDim>(type, range);
  return type;
}

Range PackedRangeDim::getRange() const { return getImpl()->range; }

//===----------------------------------------------------------------------===//
// Unpacked Dimensions
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {

struct SizedDimStorage : DimStorage {
  using KeyTy = std::pair<UnpackedType, unsigned>;

  SizedDimStorage(KeyTy key) : DimStorage(key.first), size(key.second) {}
  bool operator==(const KeyTy &key) const {
    return key.first == inner && key.second == size;
  }
  static SizedDimStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<SizedDimStorage>()) SizedDimStorage(key);
  }

  unsigned size;
};

struct AssocDimStorage : DimStorage {
  using KeyTy = std::pair<UnpackedType, UnpackedType>;

  AssocDimStorage(KeyTy key) : DimStorage(key.first), indexType(key.second) {}
  bool operator==(const KeyTy &key) const {
    return key.first == inner && key.second == indexType;
  }
  static AssocDimStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<AssocDimStorage>()) AssocDimStorage(key);
  }

  UnpackedType indexType;
};

} // namespace detail
} // namespace moore
} // namespace circt

UnpackedType UnpackedDim::getInner() const { return getImpl()->inner; }

void UnpackedDim::format(
    llvm::raw_ostream &os,
    llvm::function_ref<void(llvm::raw_ostream &)> around) const {
  SmallVector<UnpackedDim> dims;
  dims.push_back(*this);
  for (;;) {
    UnpackedType inner = dims.back().getInner();
    if (auto dim = inner.dyn_cast<UnpackedDim>()) {
      dims.push_back(dim);
    } else {
      inner.format(os);
      break;
    }
  }
  os << " ";
  if (around)
    around(os);
  else
    os << "$";
  os << " ";
  for (auto dim : dims) {
    dim.formatDim(os);
  }
}

void UnpackedDim::formatDim(llvm::raw_ostream &os) const {
  TypeSwitch<UnpackedDim>(*this)
      .Case<UnpackedUnsizedDim>([&](auto dim) { os << "[]"; })
      .Case<UnpackedArrayDim>(
          [&](auto dim) { os << "[" << dim.getSize() << "]"; })
      .Case<UnpackedRangeDim>(
          [&](auto dim) { os << "[" << dim.getRange() << "]"; })
      .Case<UnpackedAssocDim>([&](auto dim) {
        os << "[";
        if (auto indexType = dim.getIndexType())
          indexType.format(os);
        else
          os << "*";
        os << "]";
      })
      .Case<UnpackedQueueDim>([&](auto dim) {
        os << "[$";
        if (auto bound = dim.getBound())
          os << ":" << *bound;
        os << "]";
      })
      .Default([&](auto) { llvm_unreachable("unhandled dim type"); });
}

UnpackedType UnpackedDim::resolved() const { return getImpl()->resolved; }

UnpackedType UnpackedDim::fullyResolved() const {
  return getImpl()->fullyResolved;
}

const detail::DimStorage *UnpackedDim::getImpl() const {
  return static_cast<detail::DimStorage *>(this->impl);
}

UnpackedUnsizedDim UnpackedUnsizedDim::get(UnpackedType inner) {
  auto type = Base::get(inner.getContext(), inner);
  type.getImpl()->finalize<UnpackedUnsizedDim>(type);
  return type;
}

UnpackedArrayDim UnpackedArrayDim::get(UnpackedType inner, unsigned size) {
  auto type = Base::get(inner.getContext(), inner, size);
  type.getImpl()->finalize<UnpackedArrayDim>(type, size);
  return type;
}

unsigned UnpackedArrayDim::getSize() const { return getImpl()->size; }

UnpackedRangeDim UnpackedRangeDim::get(UnpackedType inner, Range range) {
  auto type = Base::get(inner.getContext(), inner, range);
  type.getImpl()->finalize<UnpackedRangeDim>(type, range);
  return type;
}

Range UnpackedRangeDim::getRange() const { return getImpl()->range; }

UnpackedAssocDim UnpackedAssocDim::get(UnpackedType inner,
                                       UnpackedType indexType) {
  auto type = Base::get(inner.getContext(), inner, indexType);
  type.getImpl()->finalize<UnpackedAssocDim>(type, indexType);
  return type;
}

UnpackedType UnpackedAssocDim::getIndexType() const {
  return getImpl()->indexType;
}

UnpackedQueueDim UnpackedQueueDim::get(UnpackedType inner,
                                       std::optional<unsigned> bound) {
  auto type = Base::get(inner.getContext(), inner, bound.value_or(-1));
  type.getImpl()->finalize<UnpackedQueueDim>(type, bound);
  return type;
}

std::optional<unsigned> UnpackedQueueDim::getBound() const {
  unsigned bound = getImpl()->size;
  if (bound == static_cast<unsigned>(-1))
    return {};
  return bound;
}

//===----------------------------------------------------------------------===//
// Enumerations
//===----------------------------------------------------------------------===//

namespace circt {
namespace moore {
namespace detail {

struct EnumTypeStorage : TypeStorage {
  using KeyTy = std::tuple<StringAttr, Location, PackedType, char>;

  EnumTypeStorage(KeyTy key)
      : name(std::get<0>(key)), loc(std::get<1>(key)), base(std::get<2>(key)),
        explicitBase(std::get<3>(key)) {}
  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == name && std::get<1>(key) == loc &&
           std::get<2>(key) == base && std::get<3>(key) == explicitBase;
  }
  static EnumTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<EnumTypeStorage>()) EnumTypeStorage(key);
  }

  StringAttr name;
  Location loc;
  PackedType base;
  bool explicitBase;
};

} // namespace detail
} // namespace moore
} // namespace circt

EnumType EnumType::get(StringAttr name, Location loc, PackedType base) {
  return Base::get(loc.getContext(), name, loc,
                   base ? base : IntType::getInt(loc.getContext()), !!base);
}

PackedType EnumType::getBase() const { return getImpl()->base; }

bool EnumType::isBaseExplicit() const { return getImpl()->explicitBase; }

StringAttr EnumType::getName() const { return getImpl()->name; }

Location EnumType::getLoc() const { return getImpl()->loc; }

void EnumType::format(llvm::raw_ostream &os) const {
  os << "enum";

  // If the enum is part of a typedefm simply print it as `enum <name>`.
  if (auto name = getName()) {
    os << " " << name.getValue();
    return;
  }

  // Otherwise print `enum <base-type>` or just `enum`.
  if (isBaseExplicit())
    os << " " << getBase();
}

//===----------------------------------------------------------------------===//
// Packed and Unpacked Structs
//===----------------------------------------------------------------------===//

StringRef moore::getMnemonicFromStructKind(StructKind kind) {
  switch (kind) {
  case StructKind::Struct:
    return "struct";
  case StructKind::Union:
    return "union";
  case StructKind::TaggedUnion:
    return "tagged_union";
  }
  llvm_unreachable("all struct kinds should be handled");
}

std::optional<StructKind> moore::getStructKindFromMnemonic(StringRef mnemonic) {
  return StringSwitch<std::optional<StructKind>>(mnemonic)
      .Case("struct", StructKind::Struct)
      .Case("union", StructKind::Union)
      .Case("tagged_union", StructKind::TaggedUnion)
      .Default({});
}

Struct::Struct(StructKind kind, ArrayRef<StructMember> members, StringAttr name,
               Location loc)
    : kind(kind), members(members.begin(), members.end()), name(name),
      loc(loc) {
  // The struct's value domain is two-valued if all members are two-valued.
  // Otherwise it is four-valued.
  domain = llvm::all_of(members,
                        [](auto &member) {
                          return member.type.getDomain() == Domain::TwoValued;
                        })
               ? Domain::TwoValued
               : Domain::FourValued;

  // The bit size is the sum of all member bit sizes, or `None` if any of the
  // member bit sizes are `None`.
  bitSize = 0;
  for (const auto &member : members) {
    if (auto memberSize = member.type.getBitSize()) {
      *bitSize += *memberSize;
    } else {
      bitSize = std::nullopt;
      break;
    }
  }
}

void Struct::format(llvm::raw_ostream &os, bool packed,
                    std::optional<Sign> signing) const {
  os << kind;
  if (packed)
    os << " packed";
  if (signing)
    os << " " << *signing;

  // If the struct is part of a typedef, simply print it as `struct <name>`.
  if (name) {
    os << " " << name.getValue();
    return;
  }

  // Otherwise actually print the struct definition inline.
  os << " {";
  for (auto &member : members)
    os << " " << member.type << " " << member.name.getValue() << ";";
  if (!members.empty())
    os << " ";
  os << "}";
}

namespace circt {
namespace moore {
namespace detail {

struct StructTypeStorage : TypeStorage {
  using KeyTy =
      std::tuple<unsigned, ArrayRef<StructMember>, StringAttr, Location>;

  StructTypeStorage(KeyTy key)
      : strukt(static_cast<StructKind>((std::get<0>(key) >> 16) & 0xFF),
               std::get<1>(key), std::get<2>(key), std::get<3>(key)),
        sign(static_cast<Sign>((std::get<0>(key) >> 8) & 0xFF)),
        explicitSign((std::get<0>(key) >> 0) & 1) {}
  static unsigned pack(StructKind kind, Sign sign, bool explicitSign) {
    return static_cast<unsigned>(kind) << 16 |
           static_cast<unsigned>(sign) << 8 | explicitSign;
  }
  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == pack(strukt.kind, sign, explicitSign) &&
           std::get<1>(key) == ArrayRef<StructMember>(strukt.members) &&
           std::get<2>(key) == strukt.name && std::get<3>(key) == strukt.loc;
  }
  static StructTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<StructTypeStorage>()) StructTypeStorage(key);
  }

  Struct strukt;
  Sign sign;
  bool explicitSign;
};

} // namespace detail
} // namespace moore
} // namespace circt

PackedStructType PackedStructType::get(StructKind kind,
                                       ArrayRef<StructMember> members,
                                       StringAttr name, Location loc,
                                       std::optional<Sign> sign) {
  assert(llvm::all_of(members,
                      [](const StructMember &member) {
                        return member.type.isa<PackedType>();
                      }) &&
         "packed struct members must be packed");
  return Base::get(loc.getContext(),
                   detail::StructTypeStorage::pack(
                       kind, sign.value_or(Sign::Unsigned), sign.has_value()),
                   members, name, loc);
}

Sign PackedStructType::getSign() const { return getImpl()->sign; }

bool PackedStructType::isSignExplicit() const {
  return getImpl()->explicitSign;
}

const Struct &PackedStructType::getStruct() const { return getImpl()->strukt; }

UnpackedStructType UnpackedStructType::get(StructKind kind,
                                           ArrayRef<StructMember> members,
                                           StringAttr name, Location loc) {
  return Base::get(loc.getContext(),
                   detail::StructTypeStorage::pack(kind, Sign::Unsigned, false),
                   members, name, loc);
}

const Struct &UnpackedStructType::getStruct() const {
  return getImpl()->strukt;
}

//===----------------------------------------------------------------------===//
// Parsing and Printing
//===----------------------------------------------------------------------===//

struct Subset {
  enum { None, Unpacked, Packed } implied = None;
  bool allowUnpacked = true;
};

static ParseResult parseMooreType(DialectAsmParser &parser, Subset subset,
                                  Type &type);
static void printMooreType(Type type, DialectAsmPrinter &printer,
                           Subset subset);

/// Parse a type with custom syntax.
static OptionalParseResult customTypeParser(DialectAsmParser &parser,
                                            StringRef mnemonic, Subset subset,
                                            llvm::SMLoc loc, Type &type) {
  auto *context = parser.getContext();

  auto yieldPacked = [&](PackedType x) {
    type = x;
    return success();
  };
  auto yieldUnpacked = [&](UnpackedType x) {
    if (!subset.allowUnpacked) {
      parser.emitError(loc)
          << "unpacked type " << x << " where only packed types are allowed";
      return failure();
    }
    type = x;
    return success();
  };
  auto yieldImplied =
      [&](llvm::function_ref<PackedType()> ifPacked,
          llvm::function_ref<UnpackedType()> ifUnpacked) {
        if (subset.implied == Subset::Packed)
          return yieldPacked(ifPacked());
        if (subset.implied == Subset::Unpacked)
          return yieldUnpacked(ifUnpacked());
        parser.emitError(loc)
            << "ambiguous packing; wrap `" << mnemonic
            << "` in `packed<...>` or `unpacked<...>` to disambiguate";
        return failure();
      };

  // Explicit packing indicators, like `unpacked.named`.
  if (mnemonic == "unpacked") {
    UnpackedType inner;
    if (parser.parseLess() ||
        parseMooreType(parser, {Subset::Unpacked, true}, inner) ||
        parser.parseGreater())
      return failure();
    return yieldUnpacked(inner);
  }
  if (mnemonic == "packed") {
    PackedType inner;
    if (parser.parseLess() ||
        parseMooreType(parser, {Subset::Packed, false}, inner) ||
        parser.parseGreater())
      return failure();
    return yieldPacked(inner);
  }

  // Packed primary types.
  if (mnemonic == "void")
    return yieldPacked(VoidType::get(context));

  if (auto kind = IntType::getKindFromKeyword(mnemonic)) {
    std::optional<Sign> sign;
    if (succeeded(parser.parseOptionalLess())) {
      StringRef signKeyword;
      if (parser.parseKeyword(&signKeyword) || parser.parseGreater())
        return failure();
      sign = getSignFromKeyword(signKeyword);
      if (!sign) {
        parser.emitError(parser.getCurrentLocation())
            << "expected keyword `unsigned` or `signed`";
        return failure();
      }
    }
    return yieldPacked(IntType::get(context, *kind, sign));
  }

  // Unpacked primary types.
  if (mnemonic == "string")
    return yieldUnpacked(StringType::get(context));
  if (mnemonic == "chandle")
    return yieldUnpacked(ChandleType::get(context));
  if (mnemonic == "event")
    return yieldUnpacked(EventType::get(context));
  if (auto kind = RealType::getKindFromKeyword(mnemonic))
    return yieldUnpacked(RealType::get(context, *kind));

  // Enums
  if (mnemonic == "enum") {
    if (parser.parseLess())
      return failure();
    StringAttr name;
    auto result = parser.parseOptionalAttribute(name);
    if (result.has_value())
      if (*result || parser.parseComma())
        return failure();
    LocationAttr loc;
    PackedType base;
    result = parser.parseOptionalAttribute(loc);
    if (result.has_value()) {
      if (*result)
        return failure();
    } else {
      if (parseMooreType(parser, {Subset::Packed, false}, base) ||
          parser.parseComma() || parser.parseAttribute(loc))
        return failure();
    }
    if (parser.parseGreater())
      return failure();
    return yieldPacked(EnumType::get(name, loc, base));
  }

  // Everything that follows can be packed or unpacked. The packing is inferred
  // from the last `packed<...>` or `unpacked<...>` that we've seen. The
  // `yieldImplied` function will call the first lambda to construct a packed
  // type, or the second lambda to construct an unpacked type. If the
  // `subset.implied` field is not set, which means there hasn't been any prior
  // `packed` or `unpacked`, the function will emit an error properly.

  // Packed and unpacked type indirections.
  if (mnemonic == "named") {
    UnpackedType inner;
    StringAttr name;
    LocationAttr loc;
    if (parser.parseLess() || parser.parseAttribute(name) ||
        parser.parseComma() || parseMooreType(parser, subset, inner) ||
        parser.parseComma() || parser.parseAttribute(loc) ||
        parser.parseGreater())
      return failure();
    return yieldImplied(
        [&]() {
          return PackedNamedType::get(inner.cast<PackedType>(), name, loc);
        },
        [&]() { return UnpackedNamedType::get(inner, name, loc); });
  }
  if (mnemonic == "ref") {
    UnpackedType inner;
    LocationAttr loc;
    if (parser.parseLess() || parseMooreType(parser, subset, inner) ||
        parser.parseComma() || parser.parseAttribute(loc) ||
        parser.parseGreater())
      return failure();
    return yieldImplied(
        [&]() { return PackedRefType::get(inner.cast<PackedType>(), loc); },
        [&]() { return UnpackedRefType::get(inner, loc); });
  }

  // Packed and unpacked ranges.
  if (mnemonic == "unsized") {
    UnpackedType inner;
    if (parser.parseLess() || parseMooreType(parser, subset, inner) ||
        parser.parseGreater())
      return failure();
    return yieldImplied(
        [&]() { return PackedUnsizedDim::get(inner.cast<PackedType>()); },
        [&]() { return UnpackedUnsizedDim::get(inner); });
  }
  if (mnemonic == "range") {
    UnpackedType inner;
    int left, right;
    if (parser.parseLess() || parseMooreType(parser, subset, inner) ||
        parser.parseComma() || parser.parseInteger(left) ||
        parser.parseColon() || parser.parseInteger(right) ||
        parser.parseGreater())
      return failure();
    return yieldImplied(
        [&]() {
          return PackedRangeDim::get(inner.cast<PackedType>(), left, right);
        },
        [&]() { return UnpackedRangeDim::get(inner, left, right); });
  }
  if (mnemonic == "array") {
    UnpackedType inner;
    unsigned size;
    if (parser.parseLess() || parseMooreType(parser, subset, inner) ||
        parser.parseComma() || parser.parseInteger(size) ||
        parser.parseGreater())
      return failure();
    return yieldUnpacked(UnpackedArrayDim::get(inner, size));
  }
  if (mnemonic == "assoc") {
    UnpackedType inner;
    UnpackedType index;
    if (parser.parseLess() || parseMooreType(parser, subset, inner))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      if (parseMooreType(parser, {Subset::Unpacked, true}, index))
        return failure();
    }
    if (parser.parseGreater())
      return failure();
    return yieldUnpacked(UnpackedAssocDim::get(inner, index));
  }
  if (mnemonic == "queue") {
    UnpackedType inner;
    std::optional<unsigned> size;
    if (parser.parseLess() || parseMooreType(parser, subset, inner))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      unsigned tmpSize;
      if (parser.parseInteger(tmpSize))
        return failure();
      size = tmpSize;
    }
    if (parser.parseGreater())
      return failure();
    return yieldUnpacked(UnpackedQueueDim::get(inner, size));
  }

  // Structs
  if (auto kind = getStructKindFromMnemonic(mnemonic)) {
    if (parser.parseLess())
      return failure();

    StringAttr name;
    auto result = parser.parseOptionalAttribute(name);
    if (result.has_value())
      if (*result || parser.parseComma())
        return failure();

    std::optional<Sign> sign;
    StringRef keyword;
    if (succeeded(parser.parseOptionalKeyword(&keyword))) {
      sign = getSignFromKeyword(keyword);
      if (!sign) {
        parser.emitError(loc) << "expected keyword `unsigned` or `signed`";
        return failure();
      }
      if (subset.implied == Subset::Unpacked) {
        parser.emitError(loc) << "unpacked struct cannot have a sign";
        return failure();
      }
      if (parser.parseComma())
        return failure();
    }

    SmallVector<StructMember> members;
    auto result2 =
        parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Braces, [&]() {
          if (parser.parseKeyword(&keyword))
            return failure();
          UnpackedType type;
          LocationAttr loc;
          if (parser.parseColon() || parseMooreType(parser, subset, type) ||
              parser.parseAttribute(loc))
            return failure();
          members.push_back(
              {StringAttr::get(parser.getContext(), keyword), loc, type});
          return success();
        });
    if (result2)
      return failure();

    LocationAttr loc;
    if (parser.parseComma() || parser.parseAttribute(loc) ||
        parser.parseGreater())
      return failure();

    return yieldImplied(
        [&]() {
          return PackedStructType::get(*kind, members, name, loc, sign);
        },
        [&]() { return UnpackedStructType::get(*kind, members, name, loc); });
  }

  return {};
}

/// Print a type with custom syntax.
static LogicalResult customTypePrinter(Type type, DialectAsmPrinter &printer,
                                       Subset subset) {
  // If we are printing a type that may be both packed or unpacked, emit a
  // wrapping `packed<...>` or `unpacked<...>` accordingly if not done so
  // previously, in order to disambiguate between the two.
  if (type.isa<PackedDim>() || type.isa<UnpackedDim>() ||
      type.isa<PackedIndirectType>() || type.isa<UnpackedIndirectType>() ||
      type.isa<PackedStructType>() || type.isa<UnpackedStructType>()) {
    auto needed = type.isa<PackedType>() ? Subset::Packed : Subset::Unpacked;
    if (needed != subset.implied) {
      printer << (needed == Subset::Packed ? "packed" : "unpacked") << "<";
      printMooreType(type, printer, {needed, true});
      printer << ">";
      return success();
    }
  }

  return TypeSwitch<Type, LogicalResult>(type)
      // Unit types
      .Case<VoidType>([&](auto) {
        printer << "void";
        return success();
      })
      .Case<StringType>([&](auto) {
        printer << "string";
        return success();
      })
      .Case<ChandleType>([&](auto) {
        printer << "chandle";
        return success();
      })
      .Case<EventType>([&](auto) {
        printer << "event";
        return success();
      })

      // Integers and reals
      .Case<IntType>([&](auto type) {
        printer << type.getKeyword();
        auto sign = type.getSign();
        if (type.isSignExplicit())
          printer << "<" << getKeywordFromSign(sign) << ">";
        return success();
      })
      .Case<RealType>(
          [&](auto type) { return printer << type.getKeyword(), success(); })

      // Enums
      .Case<EnumType>([&](auto type) {
        printer << "enum<";
        if (type.getName())
          printer << type.getName() << ", ";
        if (type.isBaseExplicit()) {
          printMooreType(type.getBase(), printer, subset);
          printer << ", ";
        }
        printer << type.getLoc() << ">";
        return success();
      })

      // Type indirections
      .Case<PackedNamedType, UnpackedNamedType>([&](auto type) {
        printer << "named<" << type.getName() << ", ";
        printMooreType(type.getInner(), printer, subset);
        printer << ", " << type.getLoc() << ">";
        return success();
      })
      .Case<PackedRefType, UnpackedRefType>([&](auto type) {
        printer << "ref<";
        printMooreType(type.getInner(), printer, subset);
        printer << ", " << type.getLoc() << ">";
        return success();
      })

      // Packed and unpacked dimensions
      .Case<PackedUnsizedDim, UnpackedUnsizedDim>([&](auto type) {
        printer << "unsized<";
        printMooreType(type.getInner(), printer, subset);
        printer << ">";
        return success();
      })
      .Case<PackedRangeDim, UnpackedRangeDim>([&](auto type) {
        printer << "range<";
        printMooreType(type.getInner(), printer, subset);
        printer << ", " << type.getRange() << ">";
        return success();
      })
      .Case<UnpackedArrayDim>([&](auto type) {
        printer << "array<";
        printMooreType(type.getInner(), printer, subset);
        printer << ", " << type.getSize() << ">";
        return success();
      })
      .Case<UnpackedAssocDim>([&](auto type) {
        printer << "assoc<";
        printMooreType(type.getInner(), printer, subset);
        if (auto indexType = type.getIndexType()) {
          printer << ", ";
          printMooreType(indexType, printer, {Subset::Unpacked, true});
        }
        printer << ">";
        return success();
      })
      .Case<UnpackedQueueDim>([&](auto type) {
        printer << "queue<";
        printMooreType(type.getInner(), printer, subset);
        if (auto bound = type.getBound())
          printer << ", " << *bound;
        printer << ">";
        return success();
      })

      // Structs
      .Case<PackedStructType, UnpackedStructType>([&](auto type) {
        const auto &strukt = type.getStruct();
        printer << getMnemonicFromStructKind(strukt.kind) << "<";
        if (strukt.name)
          printer << strukt.name << ", ";
        auto packed = type.template dyn_cast<PackedStructType>();
        if (packed && packed.isSignExplicit())
          printer << packed.getSign() << ", ";
        printer << "{";
        llvm::interleaveComma(strukt.members, printer, [&](const auto &member) {
          printer << member.name.getValue() << ": ";
          printMooreType(member.type, printer, subset);
          printer << " " << member.loc;
        });
        printer << "}, ";
        printer << strukt.loc << ">";
        return success();
      })

      .Default([](auto) { return failure(); });
}

/// Parse a type registered with this dialect.
static ParseResult parseMooreType(DialectAsmParser &parser, Subset subset,
                                  Type &type) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  StringRef mnemonic;
  OptionalParseResult result = generatedTypeParser(parser, &mnemonic, type);
  if (result.has_value())
    return result.value();

  result = customTypeParser(parser, mnemonic, subset, loc, type);
  if (result.has_value())
    return result.value();

  parser.emitError(loc) << "unknown type `" << mnemonic
                        << "` in dialect `moore`";
  return failure();
}

/// Print a type registered with this dialect.
static void printMooreType(Type type, DialectAsmPrinter &printer,
                           Subset subset) {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  if (succeeded(customTypePrinter(type, printer, subset)))
    return;
  assert(false && "no printer for unknown `moore` dialect type");
}

/// Parse a type registered with this dialect.
Type MooreDialect::parseType(DialectAsmParser &parser) const {
  Type type;
  if (parseMooreType(parser, {Subset::None, true}, type))
    return {};
  return type;
}

/// Print a type registered with this dialect.
void MooreDialect::printType(Type type, DialectAsmPrinter &printer) const {
  printMooreType(type, printer, {Subset::None, true});
}
