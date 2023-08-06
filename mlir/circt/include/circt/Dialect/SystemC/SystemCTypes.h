//===- SystemCTypes.h - Declare SystemC dialect types ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the SystemC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES_H
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES_H

#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

namespace circt {
namespace systemc {
// Forward declarations.
class IntBaseType;
class UIntBaseType;
class IntType;
class UIntType;
class BigIntType;
class BigUIntType;
class SignedType;
class UnsignedType;
class BitVectorBaseType;
class BitVectorType;
class LogicVectorBaseType;
class LogicVectorType;

namespace detail {
// Forward declarations.
struct IntegerWidthStorage;

/// A struct containing minimal information for a systemc module port. Thus, can
/// be used as parameter in attributes or types.
struct PortInfo {
  mlir::StringAttr name;
  mlir::Type type;
};
} // namespace detail

/// Get the type wrapped by a signal or port (in, inout, out) type.
Type getSignalBaseType(Type type);

/// Return the bitwidth of a type. SystemC types with a dynamic bit-width and
/// unsupported types result in a None return value.
std::optional<size_t> getBitWidth(Type type);

//===----------------------------------------------------------------------===//
// Integer types
//===----------------------------------------------------------------------===//

/// This provides a common base class for all SystemC integers.
/// It represents the sc_value_base class described in IEEE 1666-2011 §7.4.
/// Note that this is an abstract type that cannot be instantiated but can only
/// be used as a base class and for type checks.
class ValueBaseType : public Type {
public:
  static bool classof(Type type) {
    return type.isa<SignedType, UnsignedType, IntBaseType, UIntBaseType,
                    BigIntType, BigUIntType, IntType, UIntType>();
  }

  bool isSigned() { return isa<SignedType, IntBaseType>(); }

protected:
  using Type::Type;
};

/// Represents a limited word-length signed integer in SystemC as described in
/// IEEE 1666-2011 §7.5.2. The word-length is not known statically, but is
/// constant over the lifetime of the value. The maximum supported bit-width
/// is 64 bits such that it can be stored in native C integers. It is the base
/// class of 'IntType'.
class IntBaseType
    : public Type::TypeBase<IntBaseType, ValueBaseType, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<IntType>();
  }
  static IntBaseType get(MLIRContext *context);
  static constexpr StringLiteral getMnemonic() { return "int_base"; }

protected:
  using Base::Base;
};

/// Represents a limited word-length signed integer in SystemC as described in
/// IEEE 1666-2011 §7.5.4. It is allowed at all places where 'IntBaseType' is
/// allowed as it inherits and preserved all its functionality as described in
/// IEEE 1666-2011 §7.5.4.1. The difference is that the bit-width has to be
/// passed as a template argument and is thus known at compile-time.
class IntType : public Type::TypeBase<IntType, IntBaseType,
                                      systemc::detail::IntegerWidthStorage> {
public:
  static IntType get(MLIRContext *context, unsigned width);
  unsigned getWidth();
  static constexpr StringLiteral getMnemonic() { return "int"; }

protected:
  using Base::Base;
};

/// Represents a limited word-length unsigned integer in SystemC as described in
/// IEEE 1666-2011 §7.5.3. The word-length is not known statically, but is
/// constant over the lifetime of the value. The maximum supported bit-width
/// is 64 bits such that it can be stored in native C integers. It is the base
/// class of 'UIntType'.
class UIntBaseType
    : public Type::TypeBase<UIntBaseType, ValueBaseType, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<UIntType>();
  }
  static UIntBaseType get(MLIRContext *context);

  static constexpr StringLiteral getMnemonic() { return "uint_base"; }

protected:
  using Base::Base;
};

/// Represents a limited word-length unsigned integer in SystemC as described in
/// IEEE 1666-2011 §7.5.5. It is allowed at all places where 'IntBaseType' is
/// allowed as it inherits and preserved all its functionality as described in
/// IEEE 1666-2011 §7.5.5.1. The difference is that the bit-width has to be
/// passed as a template argument and is thus known at compile-time.
class UIntType : public Type::TypeBase<UIntType, UIntBaseType,
                                       systemc::detail::IntegerWidthStorage> {
public:
  static UIntType get(MLIRContext *context, unsigned width);
  unsigned getWidth();

  static constexpr StringLiteral getMnemonic() { return "uint"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length signed integer in SystemC as described in
/// IEEE 1666-2011 §7.6.3. The word-length is not known statically, but is
/// constant over the lifetime of the value. It supports arbitrary precision
/// integers, but is often limited to 512 bits (implementation dependent) for
/// performance reasons. It is the base class of 'BigIntType'.
class SignedType
    : public Type::TypeBase<SignedType, ValueBaseType, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<BigIntType>();
  }
  static SignedType get(MLIRContext *context);

  static constexpr StringLiteral getMnemonic() { return "signed"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length signed integer in SystemC as described in
/// IEEE 1666-2011 §7.6.5. It is allowed at all places where 'SignedType' is
/// allowed as it inherits and preserved all its functionality as described in
/// IEEE 1666-2011 §7.6.5.1. The difference is that the bit-width has to be
/// passed as a template argument and is thus known at compile-time.
class BigIntType : public Type::TypeBase<BigIntType, SignedType,
                                         systemc::detail::IntegerWidthStorage> {
public:
  static BigIntType get(MLIRContext *context, unsigned width);
  unsigned getWidth();
  static constexpr StringLiteral getMnemonic() { return "bigint"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length unsigned integer in SystemC as described in
/// IEEE 1666-2011 §7.6.4. The word-length is not known statically, but is
/// constant over the lifetime of the value. It supports arbitrary precision
/// integers, but is often limited to 512 bits (implementation dependent) for
/// performance reasons. It is the base class of 'BigUIntType'.
class UnsignedType
    : public Type::TypeBase<UnsignedType, ValueBaseType, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<BigUIntType>();
  }
  static UnsignedType get(MLIRContext *context);

  static constexpr StringLiteral getMnemonic() { return "unsigned"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length unsigned integer in SystemC as described in
/// IEEE 1666-2011 §7.6.6. It is allowed at all places where 'UnsignedType' is
/// allowed as it inherits and preserved all its functionality as described in
/// IEEE 1666-2011 §7.6.6.1. The difference is that the bit-width has to be
/// passed as a template argument and is thus known at compile-time.
class BigUIntType
    : public Type::TypeBase<BigUIntType, UnsignedType,
                            systemc::detail::IntegerWidthStorage> {
public:
  static BigUIntType get(MLIRContext *context, unsigned width);
  unsigned getWidth();

  static constexpr StringLiteral getMnemonic() { return "biguint"; }

protected:
  using Base::Base;
};

//===----------------------------------------------------------------------===//
// Bit-vector types
//===----------------------------------------------------------------------===//

/// Represents a finite word-length bit vector in SystemC as described in
/// IEEE 1666-2011 §7.9.3. The word-length is not known statically, but is
/// constant over the lifetime of the value. It is the base class of
/// 'BitVectorType'.
class BitVectorBaseType
    : public Type::TypeBase<BitVectorBaseType, Type, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<BitVectorType>();
  }
  static BitVectorBaseType get(MLIRContext *context);
  static constexpr StringLiteral getMnemonic() { return "bv_base"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length bit vector in SystemC as described in
/// IEEE 1666-2011 §7.9.5. It is allowed at all places where 'BitVectorBaseType'
/// is allowed as it inherits and preserved all its functionality as described
/// in IEEE 1666-2011 §7.9.5.1. The difference is that the bit-width has to be
/// passed as a template argument and is thus known at compile-time.
class BitVectorType
    : public Type::TypeBase<BitVectorType, BitVectorBaseType,
                            systemc::detail::IntegerWidthStorage> {
public:
  static BitVectorType get(MLIRContext *context, unsigned width);
  unsigned getWidth();

  static constexpr StringLiteral getMnemonic() { return "bv"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length bit vector in SystemC as described in
/// IEEE 1666-2011 §7.9.4. The word-length is not known statically, but is
/// constant over the lifetime of the value. Each bit is of type 'LogicType'
/// and thus four-valued. It is the base class of 'LogicVectorType'.
class LogicVectorBaseType
    : public Type::TypeBase<LogicVectorBaseType, Type, DefaultTypeStorage> {
public:
  static bool classof(Type type) {
    return type.getTypeID() == getTypeID() || type.isa<LogicVectorType>();
  }
  static LogicVectorBaseType get(MLIRContext *context);
  static constexpr StringLiteral getMnemonic() { return "lv_base"; }

protected:
  using Base::Base;
};

/// Represents a finite word-length bit vector (of four-state values) in SystemC
/// as described in IEEE 1666-2011 §7.9.6. It is allowed at all places where
/// 'LogicVectorBaseType' is allowed as it inherits and preserved all its
/// functionality as described in IEEE 1666-2011 §7.9.6.1. The difference is
/// that the bit-width has to be passed as a template argument and is thus known
/// at compile-time.
class LogicVectorType
    : public Type::TypeBase<LogicVectorType, LogicVectorBaseType,
                            systemc::detail::IntegerWidthStorage> {
public:
  static LogicVectorType get(MLIRContext *context, unsigned width);
  unsigned getWidth();

  static constexpr StringLiteral getMnemonic() { return "lv"; }

protected:
  using Base::Base;
};

} // namespace systemc
} // namespace circt

// Include generated types.
#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SystemC/SystemCTypes.h.inc"

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES_H
