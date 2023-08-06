//===- MooreTypes.h - Declare Moore dialect types ----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the Moore dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOORETYPES_H
#define CIRCT_DIALECT_MOORE_MOORETYPES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include <variant>

namespace circt {
namespace moore {

/// The number of values each bit of a type can assume.
enum class Domain {
  /// Two-valued types such as `bit` or `int`.
  TwoValued,
  /// Four-valued types such as `logic` or `integer`.
  FourValued,
};

/// Whether a type is signed or unsigned.
enum class Sign {
  /// An `unsigned` type.
  Unsigned,
  /// A `signed` type.
  Signed,
};

/// Map a `Sign` to the corresponding keyword.
StringRef getKeywordFromSign(const Sign &sign);
/// Map the keywords `unsigned` and `signed` to the corresponding `Sign`.
std::optional<Sign> getSignFromKeyword(StringRef keyword);

template <typename Os>
Os &operator<<(Os &os, const Sign &sign) {
  os << getKeywordFromSign(sign);
  return os;
}

/// Which side is greater in a range `[a:b]`.
enum class RangeDir {
  /// `a < b`
  Up,
  /// `a >= b`
  Down,
};

/// The `[a:b]` part in a vector/array type such as `logic [a:b]`.
struct Range {
  /// The total number of bits, given as `|a-b|+1`.
  unsigned size;
  /// The direction of the vector, i.e. whether `a > b` or `a < b`.
  RangeDir dir;
  /// The starting offset of the range.
  int offset;

  /// Construct a range `[size-1:0]`.
  explicit Range(unsigned size) : Range(size, RangeDir::Down, 0) {}

  /// Construct a range `[offset+size-1:offset]` if `dir` is `Down`, or
  /// `[offset:offset+size-1]` if `dir` is `Up`.
  Range(unsigned size, RangeDir dir, int offset)
      : size(size), dir(dir), offset(offset) {}

  /// Construct a range [left:right]`, with the direction inferred as `Down` if
  /// `left >= right`, or `Up` otherwise.
  Range(int left, int right) {
    if (left >= right) {
      size = left + 1 - right;
      dir = RangeDir::Down;
      offset = right;
    } else {
      size = right + 1 - left;
      dir = RangeDir::Up;
      offset = left;
    }
  }

  bool operator==(const Range &other) const {
    return size == other.size && dir == other.dir && offset == other.offset;
  }

  /// Get the `$left` dimension.
  int left() const { return dir == RangeDir::Up ? low() : high(); }
  /// Get the `$right` dimension.
  int right() const { return dir == RangeDir::Up ? high() : low(); }
  /// Get the `$low` dimension.
  int low() const { return offset; }
  /// Get the `$high` dimension.
  int high() const { return offset + size - 1; }
  /// Get the `$increment` size.
  int increment() const { return dir == RangeDir::Up ? 1 : -1; }

  /// Format this range as a string.
  std::string toString() const {
    std::string buffer;
    llvm::raw_string_ostream(buffer) << *this;
    return buffer;
  }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline llvm::hash_code hash_value(const Range &x) {
  return llvm::hash_combine(x.size, x.dir, x.offset);
}

template <typename Os>
Os &operator<<(Os &os, const Range &range) {
  os << range.left() << ":" << range.right();
  return os;
}

class PackedType;

/// A simple bit vector type.
///
/// The SystemVerilog standard somewhat loosely defines a "Simple Bit Vector"
/// type. In essence, this is a zero or one-dimensional integer type. For
/// example, `bit`, `logic [0:0]`, `reg [31:0]`, or `int` are SBVs, but `bit
/// [1:0][2:0]`, `int [4:0]`, `bit [5:2]`, or `bit []` are not.
struct SimpleBitVectorType {
  /// Create a null SBVT.
  SimpleBitVectorType() {}

  /// Create a new SBVT with the given domain, sign, and size. The resulting
  /// type will expand exactly to `bit signed? [size-1:0]`.
  SimpleBitVectorType(Domain domain, Sign sign, unsigned size,
                      bool usedAtom = false, bool explicitSign = false,
                      bool explicitSize = true)
      : size(size), domain(domain), sign(sign), usedAtom(usedAtom),
        explicitSign(explicitSign), explicitSize(explicitSize) {
    assert(size > 0 && "SBVT requires non-zero size");
  }

  /// Convert this SBVT to an actual type.
  PackedType getType(MLIRContext *context) const;

  /// Check whether the type is unsigned.
  bool isUnsigned() const { return sign == Sign::Unsigned; }

  /// Check whether the type is signed.
  bool isSigned() const { return sign == Sign::Signed; }

  /// Get the range of the type.
  Range getRange() const { return Range(size, RangeDir::Down, 0); }

  /// Check whether this type is equivalent to another.
  bool isEquivalent(const SimpleBitVectorType &other) const {
    return domain == other.domain && sign == other.sign && size == other.size;
  }

  bool operator==(const SimpleBitVectorType &other) const {
    if (size == 0 || other.size == 0)
      return size == other.size; // if either is null, the other has to be null
    return isEquivalent(other) && usedAtom == other.usedAtom &&
           explicitSign == other.explicitSign &&
           explicitSize == other.explicitSize;
  }

  /// Check whether this is a null type.
  operator bool() const { return size > 0; }

  /// Format this simple bit vector type as a string.
  std::string toString() const {
    std::string buffer;
    llvm::raw_string_ostream(buffer) << *this;
    return buffer;
  }

  /// The size of the vector.
  unsigned size = 0;
  /// The domain, which dictates whether this is a `bit` or `logic` vector.
  Domain domain : 8;
  /// The sign.
  Sign sign : 8;

  // The following flags ensure that converting a `PackedType` to an SBVT and
  // then back to a `PackedType` will yield exactly the original type. For
  // example, the packed type `int` maps to an SBVT `{32, TwoValued, Signed}`,
  // which should be converted back to `int` instead of `bit signed [31:0]`.

  /// Whether the type used an integer atom like `int` in the source text.
  bool usedAtom : 1;
  /// Whether the sign was explicit in the source text.
  bool explicitSign : 1;
  /// Whether the single-bit vector had an explicit range in the source text.
  /// Essentially whether it was `bit` or `bit[a:a]`.
  bool explicitSize : 1;
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline llvm::hash_code hash_value(const SimpleBitVectorType &x) {
  if (x)
    return llvm::hash_combine(x.size, x.domain, x.sign, x.usedAtom,
                              x.explicitSign, x.explicitSize);
  return {};
}

template <typename Os>
Os &operator<<(Os &os, const SimpleBitVectorType &type) {
  if (!type) {
    os << "<<<NULL SBVT>>>";
    return os;
  }
  os << (type.domain == Domain::TwoValued ? "bit" : "logic");
  if (type.sign != Sign::Unsigned || type.explicitSign)
    os << " " << type.sign;
  if (type.size > 1 || type.explicitSize)
    os << " [" << type.getRange() << "]";
  return os;
}

namespace detail {
struct RealTypeStorage;
struct IntTypeStorage;
struct IndirectTypeStorage;
struct DimStorage;
struct UnsizedDimStorage;
struct RangeDimStorage;
struct SizedDimStorage;
struct AssocDimStorage;
struct EnumTypeStorage;
struct StructTypeStorage;
} // namespace detail

/// Base class for all SystemVerilog types in the Moore dialect.
class SVType : public Type {
protected:
  using Type::Type;
};

//===----------------------------------------------------------------------===//
// Unpacked Type
//===----------------------------------------------------------------------===//

class PackedType;
class StringType;
class ChandleType;
class EventType;
class RealType;
class UnpackedIndirectType;
class UnpackedDim;
class UnpackedStructType;

/// An unpacked SystemVerilog type.
///
/// Unpacked types are a second level of types in SystemVerilog. They extend a
/// core unpacked type with a variety of unpacked dimensions, depending on which
/// syntactic construct generated the type (variable or otherwise). The core
/// unpacked types are:
///
/// - Packed types
/// - Non-integer types: `shortreal`, `real`, `realtime`
/// - Unpacked structs and unions
/// - `string`, `chandle`, `event`
/// - Virtual interfaces
/// - Class types
/// - Covergroups
/// - Unpacked named types
/// - Unpacked type references
///
/// The unpacked dimensions are:
///
/// - Unsized (`[]`)
/// - Arrays (`[x]`)
/// - Ranges (`[x:y]`)
/// - Associative (`[T]` or `[*]`)
/// - Queues (`[$]` or `[$:x]`)
class UnpackedType : public SVType {
public:
  static bool classof(Type type) {
    return type.isa<PackedType>() || type.isa<StringType>() ||
           type.isa<ChandleType>() || type.isa<EventType>() ||
           type.isa<RealType>() || type.isa<UnpackedIndirectType>() ||
           type.isa<UnpackedDim>() || type.isa<UnpackedStructType>();
  }

  /// Resolve one level of name or type reference indirection.
  ///
  /// For example, given `typedef int foo; typedef foo bar;`, resolves `bar`
  /// to `foo`.
  UnpackedType resolved() const;

  /// Resolve all name or type reference indirections.
  ///
  /// For example, given `typedef int foo; typedef foo bar;`, resolves `bar`
  /// to `int`.
  UnpackedType fullyResolved() const;

  /// Get the value domain of this type.
  Domain getDomain() const;

  /// Get the sign for this type.
  Sign getSign() const;

  /// Get the size of this type in bits.
  ///
  /// Returns `None` if any of the type's dimensions is unsized, associative, or
  /// a queue, or the core type itself has no known size.
  std::optional<unsigned> getBitSize() const;

  /// Get this type as a simple bit vector, if it is one. Returns a null type
  /// otherwise.
  SimpleBitVectorType getSimpleBitVectorOrNull() const;

  /// Check whether this is a simple bit vector type.
  bool isSimpleBitVector() const { return !!getSimpleBitVectorOrNull(); }

  /// Get this type as a simple bit vector. Aborts if it is no simple bit
  /// vector.
  SimpleBitVectorType getSimpleBitVector() const {
    auto sbv = getSimpleBitVectorOrNull();
    assert(sbv && "getSimpleBitVector called on type that is no SBV");
    return sbv;
  }

  /// Cast this type to a simple bit vector. Returns null if this type cannot be
  /// cast to a simple bit vector.
  SimpleBitVectorType castToSimpleBitVectorOrNull() const;

  /// Check whether this type can be cast to a simple bit vector type.
  bool isCastableToSimpleBitVector() const {
    return !!castToSimpleBitVectorOrNull();
  }

  /// Cast this type to a simple bit vector. Aborts if this type cannot be cast
  /// to a simple bit vector.
  SimpleBitVectorType castToSimpleBitVector() const {
    auto sbv = castToSimpleBitVectorOrNull();
    assert(
        sbv &&
        "castToSimpleBitVector called on type that cannot be cast to an SBV");
    return sbv;
  }

  /// Format this type in SystemVerilog syntax into an output stream. Useful to
  /// present the type back to the user in diagnostics.
  void
  format(llvm::raw_ostream &os,
         llvm::function_ref<void(llvm::raw_ostream &os)> around = {}) const;

  void format(llvm::raw_ostream &os, StringRef around) const {
    format(os, [&](llvm::raw_ostream &os) { os << around; });
  }

  /// Format this type in SystemVerilog syntax into a string. Useful to present
  /// the type back to the user in diagnostics. Prefer the `format` function if
  /// possible, as that does not need to allocate a string.
  template <typename... Args>
  std::string toString(Args... args) const {
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    format(os, args...);
    return buffer;
  }

protected:
  using SVType::SVType;
};

template <
    typename Ty,
    std::enable_if_t<std::is_base_of<UnpackedType, Ty>::value, bool> = true>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Ty type) {
  type.format(os);
  return os;
}

//===----------------------------------------------------------------------===//
// Packed Type
//===----------------------------------------------------------------------===//

class VoidType;
class IntType;
class PackedIndirectType;
class PackedDim;
class EnumType;
class PackedStructType;

/// A packed SystemVerilog type.
///
/// Packed types are the core types of SystemVerilog. They combine a core packed
/// type with an optional sign and zero or more packed dimensions. The core
/// packed types are:
///
/// - Integer vector types: `bit`, `logic`, `reg`
/// - Integer atom types: `byte`, `shortint`, `int`, `longint`, `integer`,
///   `time`
/// - Packed structs and unions
/// - Enums
/// - Packed named types
/// - Packed type references
///
/// The packed dimensions can be:
///
/// - Unsized (`[]`)
/// - Ranges (`[x:y]`)
///
/// Note that every packed type is also a valid unpacked type. But unpacked
/// types are *not* valid packed types.
class PackedType : public UnpackedType {
public:
  static bool classof(Type type) {
    return type.isa<VoidType>() || type.isa<IntType>() ||
           type.isa<PackedIndirectType>() || type.isa<PackedDim>() ||
           type.isa<EnumType>() || type.isa<PackedStructType>();
  }

  /// Resolve one level of name or type reference indirection.
  ///
  /// For example, given `typedef int foo; typedef foo bar;`, resolves `bar`
  /// to `foo`.
  PackedType resolved() const;

  /// Resolve all name or type reference indirections.
  ///
  /// For example, given `typedef int foo; typedef foo bar;`, resolves `bar`
  /// to `int`.
  PackedType fullyResolved() const;

  /// Get the value domain of this type.
  Domain getDomain() const;

  /// Get the sign for this type.
  Sign getSign() const;

  /// Get the size of this type in bits.
  ///
  /// Returns `None` if any of the type's dimensions is unsized.
  std::optional<unsigned> getBitSize() const;

  /// Format this type in SystemVerilog syntax into an output stream. Useful to
  /// present the type back to the user in diagnostics.
  void format(llvm::raw_ostream &os) const;

protected:
  using UnpackedType::UnpackedType;
};

//===----------------------------------------------------------------------===//
// Unit Types
//===----------------------------------------------------------------------===//

/// The `void` type.
class VoidType
    : public Type::TypeBase<VoidType, PackedType, DefaultTypeStorage> {
public:
  static VoidType get(MLIRContext *context);

protected:
  using Base::Base;
};

/// The `string` type.
class StringType
    : public Type::TypeBase<StringType, UnpackedType, DefaultTypeStorage> {
public:
  static StringType get(MLIRContext *context);

protected:
  using Base::Base;
};

/// The `chandle` type.
class ChandleType
    : public Type::TypeBase<ChandleType, UnpackedType, DefaultTypeStorage> {
public:
  static ChandleType get(MLIRContext *context);

protected:
  using Base::Base;
};

/// The `event` type.
class EventType
    : public Type::TypeBase<EventType, UnpackedType, DefaultTypeStorage> {
public:
  static EventType get(MLIRContext *context);

protected:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// Packed Integers
//===----------------------------------------------------------------------===//

/// An integer vector or atom type.
class IntType
    : public Type::TypeBase<IntType, PackedType, detail::IntTypeStorage> {
public:
  enum Kind {
    // The integer vector types. These are the builtin single-bit integer types.
    /// A `bit`.
    Bit,
    /// A `logic`.
    Logic,
    /// A `reg`.
    Reg,

    // The integer atom types. These are the builtin multi-bit integer types.
    /// A `byte`.
    Byte,
    /// A `shortint`.
    ShortInt,
    /// An `int`.
    Int,
    /// A `longint`.
    LongInt,
    /// An `integer`.
    Integer,
    /// A `time`.
    Time,
  };

  /// Get the integer type that corresponds to a keyword (like `bit`).
  static std::optional<Kind> getKindFromKeyword(StringRef keyword);
  /// Get the keyword (like `bit`) for one of the integer types.
  static StringRef getKeyword(Kind kind);
  /// Get the default sign for one of the integer types.
  static Sign getDefaultSign(Kind kind);
  /// Get the value domain for one of the integer types.
  static Domain getDomain(Kind kind);
  /// Get the size of one of the integer types.
  static unsigned getBitSize(Kind kind);
  /// Get the integer type that corresponds to a domain and bit size. For
  /// example, returns `int` for `(TwoValued, 32)`.
  static std::optional<Kind> getKindFromDomainAndSize(Domain domain,
                                                      unsigned size);

  static IntType get(MLIRContext *context, Kind kind,
                     std::optional<Sign> sign = {});

  /// Create a `logic` type.
  static IntType getLogic(MLIRContext *context) { return get(context, Logic); }

  /// Create a `int` type.
  static IntType getInt(MLIRContext *context) { return get(context, Int); }

  /// Create a `time` type.
  static IntType getTime(MLIRContext *context) { return get(context, Time); }

  /// Get the concrete integer vector or atom type.
  Kind getKind() const;
  /// Get the sign of this type.
  Sign getSign() const;
  /// Whether the sign of the type was specified explicitly. This allows us to
  /// distinguish `bit unsigned` from `bit`.
  bool isSignExplicit() const;

  /// Get the keyword (like `bit`) for this type.
  StringRef getKeyword() const { return getKeyword(getKind()); }
  /// Get the default sign for this type.
  Sign getDefaultSign() const { return getDefaultSign(getKind()); }
  /// Get the value domain for this type.
  Domain getDomain() const { return getDomain(getKind()); }
  /// Get the size of this type.
  unsigned getBitSize() const { return getBitSize(getKind()); }

  /// Format this type in SystemVerilog syntax. Useful to present the type back
  /// to the user in diagnostics.
  void format(llvm::raw_ostream &os) const;

protected:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// Unpacked Reals
//===----------------------------------------------------------------------===//

/// A real type.
class RealType
    : public Type::TypeBase<RealType, UnpackedType, detail::RealTypeStorage> {
public:
  enum Kind {
    /// A `shortreal`.
    ShortReal,
    /// A `real`.
    Real,
    /// A `realtime`.
    RealTime,
  };

  /// Get the integer type that corresponds to a keyword (like `bit`).
  static std::optional<Kind> getKindFromKeyword(StringRef keyword);
  /// Get the keyword (like `bit`) for one of the integer types.
  static StringRef getKeyword(Kind kind);
  /// Get the size of one of the integer types.
  static unsigned getBitSize(Kind kind);

  static RealType get(MLIRContext *context, Kind kind);

  /// Get the concrete integer vector or atom type.
  Kind getKind() const;

  /// Get the keyword (like `bit`) for this type.
  StringRef getKeyword() const { return getKeyword(getKind()); }
  /// Get the size of this type.
  unsigned getBitSize() const { return getBitSize(getKind()); }

protected:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// Packed and Unpacked Type Indirections
//===----------------------------------------------------------------------===//

class PackedNamedType;
class PackedRefType;
class UnpackedNamedType;
class UnpackedRefType;

namespace detail {
UnpackedType getIndirectTypeInner(const TypeStorage *impl);
Location getIndirectTypeLoc(const TypeStorage *impl);
StringAttr getIndirectTypeName(const TypeStorage *impl);
} // namespace detail

/// Common base class for name and type reference indirections.
///
/// These handle the cases where the source text uses a `typedef` or
/// `type(<decl>)` construct to build a type. We keep track of these
/// indirections alongside the location in the source text where they were
/// created, in order to be able to reproduce the exact source text type in
/// diagnostics.
///
/// We use this templated base class to construct separate packed and unpacked
/// indirect types, where holding a packed indirect type guarantees  that the
/// inner type is a packed type as well. The resulting inheritance trees are:
///
/// - `PackedNamedType -> PackedIndirectType -> PackedType`
/// - `PackedRefType -> PackedIndirectType -> PackedType`
/// - `UnpackedNamedType -> UnpackedIndirectType -> UnpackedType`
/// - `UnpackedRefType -> UnpackedIndirectType -> UnpackedType`
template <class BaseTy>
class IndirectTypeBase : public BaseTy {
protected:
  using InnerType = BaseTy;
  using BaseTy::BaseTy;
  using Base = IndirectTypeBase<BaseTy>;

public:
  /// Get the type this indirection wraps.
  BaseTy getInner() const {
    return detail::getIndirectTypeInner(this->impl).template cast<BaseTy>();
  }

  /// Get the location in the source text where the indirection was generated.
  Location getLoc() const { return detail::getIndirectTypeLoc(this->impl); }

  /// Resolve one level of name or type reference indirection. This simply
  /// returns the inner type, which removes the name indirection introduced by
  /// this type. See `PackedType::resolved` and `UnpackedType::resolved`.
  BaseTy resolved() const { return getInner(); }

  /// Resolve all name or type reference indirections. This always returns the
  /// fully resolved inner type. See `PackedType::fullyResolved` and
  /// `UnpackedType::fullyResolved`.
  BaseTy fullyResolved() const { return getInner().fullyResolved(); }
};

/// A named type.
///
/// Named types are user-defined types that are introduced with a `typedef
/// <inner> <name>` construct in the source file. They are composed of the
/// following information:
///
/// - `inner: The type that this name expands to.
/// - `name`: How the user originally called the type.
/// - `loc`: The location of the typedef in the source file.
template <class ConcreteTy, class BaseTy>
class NamedTypeBase
    : public Type::TypeBase<ConcreteTy, BaseTy, detail::IndirectTypeStorage> {
protected:
  using InnerType = typename BaseTy::InnerType;
  using Type::TypeBase<ConcreteTy, BaseTy,
                       detail::IndirectTypeStorage>::TypeBase;
  using NamedBase = NamedTypeBase<ConcreteTy, BaseTy>;

public:
  static ConcreteTy get(InnerType inner, StringAttr name, Location loc);
  static ConcreteTy get(InnerType inner, StringRef name, Location loc) {
    return get(inner, StringAttr::get(inner.getContext(), name), loc);
  }

  /// Get the name assigned to the wrapped type.
  StringAttr getName() const { return detail::getIndirectTypeName(this->impl); }
};

/// A type reference.
///
/// Type references are introduced with a `type(<decl>)` construct in the source
/// file. They are composed of the following information:
///
/// - `inner`: The type that this reference expands to.
/// - `loc`: The location of the `type(...)` in the source file.
template <class ConcreteTy, class BaseTy>
class RefTypeBase
    : public Type::TypeBase<ConcreteTy, BaseTy, detail::IndirectTypeStorage> {
protected:
  using InnerType = typename BaseTy::InnerType;
  using Type::TypeBase<ConcreteTy, BaseTy,
                       detail::IndirectTypeStorage>::TypeBase;
  using RefBase = RefTypeBase<ConcreteTy, BaseTy>;

public:
  static ConcreteTy get(InnerType inner, Location loc);
};

/// A packed type indirection. See `IndirectTypeBase` for details.
class PackedIndirectType : public IndirectTypeBase<PackedType> {
public:
  static bool classof(Type type) {
    return type.isa<PackedNamedType>() || type.isa<PackedRefType>();
  }

protected:
  using Base::Base;
};

/// An unpacked type indirection. See `IndirectTypeBase` for details.
class UnpackedIndirectType : public IndirectTypeBase<UnpackedType> {
public:
  static bool classof(Type type) {
    return type.isa<UnpackedNamedType>() || type.isa<UnpackedRefType>();
  }

protected:
  using Base::Base;
};

/// A packed named type. See `NamedTypeBase` for details.
class PackedNamedType
    : public NamedTypeBase<PackedNamedType, PackedIndirectType> {
protected:
  using NamedBase::NamedBase;
};

/// An unpacked named type. See `NamedTypeBase` for details.
class UnpackedNamedType
    : public NamedTypeBase<UnpackedNamedType, UnpackedIndirectType> {
protected:
  using NamedBase::NamedBase;
};

/// A packed named type. See `NamedTypeBase` for details.
class PackedRefType : public RefTypeBase<PackedRefType, PackedIndirectType> {
protected:
  using RefBase::RefBase;
};

/// An unpacked named type. See `NamedTypeBase` for details.
class UnpackedRefType
    : public RefTypeBase<UnpackedRefType, UnpackedIndirectType> {
protected:
  using RefBase::RefBase;
};

//===----------------------------------------------------------------------===//
// Packed Dimensions
//===----------------------------------------------------------------------===//

class PackedRangeDim;
class PackedUnsizedDim;

/// A packed dimension.
class PackedDim : public PackedType {
public:
  static bool classof(Type type) {
    return type.isa<PackedRangeDim>() || type.isa<PackedUnsizedDim>();
  }

  /// Get the element type of the dimension. This is the `x` in `x[a:b]`.
  PackedType getInner() const;

  /// Format this type in SystemVerilog syntax. Useful to present the type back
  /// to the user in diagnostics.
  void format(llvm::raw_ostream &os) const;
  /// Format just the dimension part, `[...]`.
  void formatDim(llvm::raw_ostream &os) const;

  /// Resolve one level of name or type reference indirection. See
  /// `PackedType::resolved`.
  PackedType resolved() const;

  /// Resolve all name or type reference indirections. See
  /// `PackedType::fullyResolved`.
  PackedType fullyResolved() const;

  /// Get the dimension's range, or `None` if it is unsized.
  std::optional<Range> getRange() const;
  /// Get the dimension's size, or `None` if it is unsized.
  std::optional<unsigned> getSize() const;

protected:
  using PackedType::PackedType;
  const detail::DimStorage *getImpl() const;
};

/// A packed unsized dimension, like `[]`.
class PackedUnsizedDim : public Type::TypeBase<PackedUnsizedDim, PackedDim,
                                               detail::UnsizedDimStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static PackedUnsizedDim get(PackedType inner);

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

/// A packed range dimension, like `[a:b]`.
class PackedRangeDim
    : public Type::TypeBase<PackedRangeDim, PackedDim, detail::RangeDimStorage,
                            ::mlir::TypeTrait::IsMutable> {
public:
  static PackedRangeDim get(PackedType inner, Range range);

  /// Get a packed range with arguments forwarded to the `Range` constructor.
  /// See `Range::Range` for details.
  template <typename... Args>
  static PackedRangeDim get(PackedType inner, Args... args) {
    return get(inner, Range(args...));
  }

  /// Get the range of this dimension.
  Range getRange() const;

  /// Allow implicit casts from `PackedRangeDim` to the actual range.
  operator Range() const { return getRange(); }

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

//===----------------------------------------------------------------------===//
// Unpacked Dimensions
//===----------------------------------------------------------------------===//

class UnpackedUnsizedDim;
class UnpackedArrayDim;
class UnpackedRangeDim;
class UnpackedAssocDim;
class UnpackedQueueDim;

/// An unpacked dimension.
class UnpackedDim : public UnpackedType {
public:
  static bool classof(Type type) {
    return type.isa<UnpackedUnsizedDim>() || type.isa<UnpackedArrayDim>() ||
           type.isa<UnpackedRangeDim>() || type.isa<UnpackedAssocDim>() ||
           type.isa<UnpackedQueueDim>();
  }

  /// Get the element type of the dimension. This is the `x` in `x[a:b]`.
  UnpackedType getInner() const;

  /// Format this type in SystemVerilog syntax. Useful to present the type back
  /// to the user in diagnostics. The unpacked dimensions are separated from any
  /// packed dimensions by calling the provided `around` callback, or a `$` if
  /// no callback has been provided. This can be useful when printing
  /// declarations like `bit [7:0] foo [16]` to have the type properly surround
  /// the declaration name `foo`, and to easily tell packed from unpacked
  /// dimensions in types like `bit [7:0] $ [15]`.
  void format(llvm::raw_ostream &os,
              llvm::function_ref<void(llvm::raw_ostream &)> around = {}) const;
  /// Format just the dimension part, `[...]`.
  void formatDim(llvm::raw_ostream &os) const;

  /// Resolve one level of name or type reference indirection. See
  /// `UnpackedType::resolved`.
  UnpackedType resolved() const;

  /// Resolve all name or type reference indirections. See
  /// `UnpackedType::fullyResolved`.
  UnpackedType fullyResolved() const;

protected:
  using UnpackedType::UnpackedType;
  const detail::DimStorage *getImpl() const;
};

/// An unpacked unsized dimension, like `[]`.
class UnpackedUnsizedDim
    : public Type::TypeBase<UnpackedUnsizedDim, UnpackedDim,
                            detail::UnsizedDimStorage,
                            ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedUnsizedDim get(UnpackedType inner);

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

/// An unpacked array dimension, like `[a]`.
class UnpackedArrayDim : public Type::TypeBase<UnpackedArrayDim, UnpackedDim,
                                               detail::SizedDimStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedArrayDim get(UnpackedType inner, unsigned size);

  /// Get the size of the array, i.e. the `a` in `[a]`.
  unsigned getSize() const;

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

/// An unpacked range dimension, like `[a:b]`.
class UnpackedRangeDim : public Type::TypeBase<UnpackedRangeDim, UnpackedDim,
                                               detail::RangeDimStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedRangeDim get(UnpackedType inner, Range range);

  /// Get a packed range with arguments forwarded to the `Range` constructor.
  /// See `Range::Range` for details.
  template <typename... Args>
  static UnpackedRangeDim get(UnpackedType inner, Args... args) {
    return get(inner, Range(args...));
  }

  /// Get the range of this dimension.
  Range getRange() const;

  /// Allow implicit casts from `UnpackedRangeDim` to the actual range.
  operator Range() const { return getRange(); }

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

/// An unpacked associative dimension, like `[T]` or `[*]`.
///
/// Associative arrays in SystemVerilog can have a concrete index type (`[T]`),
/// or a wildcard index type (`[*]`, §7.8.1). The latter is exceptionally
/// strange, as it applies only to integer indices, but supports arbitrarily
/// sized indices by always removing leading zeros from any index that is used
/// in the lookup. This is interesting if a `string` is used to index into such
/// an array, because strings are automatically cast to a bit vector of
/// equivalent size, which results in a sort-of string key lookup. However, note
/// that there are also dedicated semantics for using `string` as the actual
/// index type (§7.8.2).
///
/// See IEEE 1800-2017 §7.8 "Associative arrays".
class UnpackedAssocDim : public Type::TypeBase<UnpackedAssocDim, UnpackedDim,
                                               detail::AssocDimStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedAssocDim get(UnpackedType inner, UnpackedType indexType = {});

  /// Get the index type of the associative dimension. This returns either the
  /// type `T` in a dimension `[T]`, or a null type in a dimension `[*]`.
  UnpackedType getIndexType() const;

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

/// An unpacked queue dimension with optional bound, like `[$]` or `[$:a]`.
class UnpackedQueueDim : public Type::TypeBase<UnpackedQueueDim, UnpackedDim,
                                               detail::SizedDimStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedQueueDim get(UnpackedType inner,
                              std::optional<unsigned> bound = {});

  /// Get the bound of the queue, i.e. the `a` in `[$:a]`. Returns `None` if the
  /// queue is unbounded.
  std::optional<unsigned> getBound() const;

protected:
  using Base::Base;
  friend struct detail::DimStorage;
};

//===----------------------------------------------------------------------===//
// Enumerations
//===----------------------------------------------------------------------===//

/// An enum type.
class EnumType
    : public Type::TypeBase<EnumType, PackedType, detail::EnumTypeStorage> {
public:
  static EnumType get(StringAttr name, Location loc, PackedType base = {});

  /// Get the base type of the enumeration.
  PackedType getBase() const;
  /// Returns whether the base type was explicitly specified by the user. This
  /// allows us to distinguish `enum` from `enum int`.
  bool isBaseExplicit() const;
  /// Get the name of the surrounding typedef, if this enum is embedded in a
  /// typedef. Otherwise this returns a null attribute.
  StringAttr getName() const;
  /// Get the location in the source text where the enum was declared. This
  /// shall be the location of the `enum` keyword or, if the enum is embedded in
  /// a typedef, the location of the typedef name.
  Location getLoc() const;

  /// Format this enum in SystemVerilog syntax. Useful to present the enum back
  /// to the user in diagnostics.
  void format(llvm::raw_ostream &os) const;

protected:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// Packed and Unpacked Structs
//===----------------------------------------------------------------------===//

/// Whether a struct is a `struct`, `union`, or `union tagged`.
enum class StructKind {
  /// A `struct`.
  Struct,
  /// A `union`.
  Union,
  /// A `union tagged`.
  TaggedUnion,
};

/// Map a `StructKind` to the corresponding mnemonic.
StringRef getMnemonicFromStructKind(StructKind kind);
/// Map a mnemonic to the corresponding `StructKind`.
std::optional<StructKind> getStructKindFromMnemonic(StringRef mnemonic);

template <typename Os>
Os &operator<<(Os &os, const StructKind &kind) {
  static constexpr StringRef keywords[] = {"struct", "union", "union tagged"};
  os << keywords[static_cast<unsigned>(kind)];
  return os;
}

/// A member of a struct.
struct StructMember {
  /// The name of this member.
  StringAttr name;
  /// The location in the source text where this member was declared.
  Location loc;
  /// The type of this member.
  UnpackedType type;

  bool operator==(const StructMember &other) const {
    return name == other.name && loc == other.loc && type == other.type;
  }
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline llvm::hash_code hash_value(const StructMember &x) {
  return llvm::hash_combine(x.name, x.loc, x.type);
}

/// A struct.
///
/// This represents both packed and unpacked structs. Which one it is depends on
/// whether this struct is embedded in a `PackedStructType` or a
/// `UnpackedStructType`. For the packed version the struct members are
/// guaranteed to be packed types as well.
struct Struct {
  /// Whether this is a `struct`, `union`, or `union tagged`.
  StructKind kind;
  /// The list of members.
  SmallVector<StructMember, 4> members;
  /// The value domain of this struct. If all members are two-valued, the
  /// overall struct is two-valued. Otherwise the struct is four-valued.
  Domain domain;
  /// The size of this struct in bits. This is `None` if any member type has an
  /// unknown size. This is commonly the case for unpacked member types, or
  /// dimensions with unknown size such as `[]` or `[$]`.
  std::optional<unsigned> bitSize;
  /// The name of the surrounding typedef, if this struct is embedded in a
  /// typedef. Otherwise this is a null attribute.
  StringAttr name;
  /// The location in the source text where the struct was declared. This shall
  /// be the location of the `struct` or `union` keyword, or, if the struct is
  /// embedded in a typedef, the location of the typedef name.
  Location loc;

  /// Create a new struct.
  Struct(StructKind kind, ArrayRef<StructMember> members, StringAttr name,
         Location loc);

  /// Format this struct in SystemVerilog syntax. Useful to present the struct
  /// back to the user in diagnostics.
  void format(llvm::raw_ostream &os, bool packed = false,
              std::optional<Sign> signing = {}) const;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Struct &strukt) {
  strukt.format(os);
  return os;
}

/// A packed struct.
class PackedStructType : public Type::TypeBase<PackedStructType, PackedType,
                                               detail::StructTypeStorage,
                                               ::mlir::TypeTrait::IsMutable> {
public:
  static PackedStructType get(StructKind kind, ArrayRef<StructMember> members,
                              StringAttr name, Location loc,
                              std::optional<Sign> sign = {});
  static PackedStructType get(const Struct &strukt,
                              std::optional<Sign> sign = {}) {
    return get(strukt.kind, strukt.members, strukt.name, strukt.loc, sign);
  }

  /// Get the sign of this struct.
  Sign getSign() const;
  /// Returns whether the sign was explicitly mentioned by the user.
  bool isSignExplicit() const;
  /// Get the struct definition.
  const Struct &getStruct() const;

  /// Format this struct in SystemVerilog syntax. Useful to present the struct
  /// back to the user in diagnostics.
  void format(llvm::raw_ostream &os) const {
    getStruct().format(os, true,
                       isSignExplicit() ? std::optional<Sign>(getSign())
                                        : std::optional<Sign>());
  }

  /// Allow implicit casts from `PackedStructType` to the actual struct
  /// definition.
  operator const Struct &() const { return getStruct(); }

protected:
  using Base::Base;
};

/// An unpacked struct.
class UnpackedStructType
    : public Type::TypeBase<UnpackedStructType, UnpackedType,
                            detail::StructTypeStorage,
                            ::mlir::TypeTrait::IsMutable> {
public:
  static UnpackedStructType get(StructKind kind, ArrayRef<StructMember> members,
                                StringAttr name, Location loc);
  static UnpackedStructType get(const Struct &strukt) {
    return get(strukt.kind, strukt.members, strukt.name, strukt.loc);
  }

  /// Get the struct definition.
  const Struct &getStruct() const;

  /// Format this struct in SystemVerilog syntax. Useful to present the struct
  /// back to the user in diagnostics.
  void format(llvm::raw_ostream &os) const { getStruct().format(os); }

  /// Allow implicit casts from `UnpackedStructType` to the actual struct
  /// definition.
  operator const Struct &() const { return getStruct(); }

protected:
  using Base::Base;
};

} // namespace moore
} // namespace circt

//===----------------------------------------------------------------------===//
// Hashing
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct DenseMapInfo<circt::moore::Range> {
  using Range = circt::moore::Range;
  static inline Range getEmptyKey() { return Range(-1); }
  static inline Range getTombstoneKey() { return Range(-2); }
  static unsigned getHashValue(const Range &x) {
    return circt::moore::hash_value(x);
  }
  static bool isEqual(const Range &lhs, const Range &rhs) { return lhs == rhs; }
};

} // namespace llvm

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Moore/MooreTypes.h.inc"

#endif // CIRCT_DIALECT_MOORE_MOORETYPES_H
