//===- SVOps.h - Declare SV dialect operations ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_OPS_H
#define CIRCT_DIALECT_SV_OPS_H

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace hw {
class InstanceOp;
class HWSymbolCache;
class InnerRefAttr;
} // namespace hw

namespace sv {

/// Return true if the specified operation is an expression.
bool isExpression(Operation *op);

/// Returns if the expression is known to be 2-state (binary)
bool is2StateExpression(Value v);

//===----------------------------------------------------------------------===//
// CaseOp Support
//===----------------------------------------------------------------------===//

/// This describes the bit in a pattern, 0/1/x/z.
enum class CasePatternBit { Zero = 0, One = 1, AnyX = 2, AnyZ = 3 };

/// Return the letter for the specified pattern bit, e.g. "0", "1", "x" or "z".
char getLetter(CasePatternBit bit);

// This is provides convenient access to encode and decode a pattern.
class CasePattern {

public:
  enum CasePatternKind { CPK_bit, CPK_enum, CPK_default };
  CasePattern(CasePatternKind kind) : kind(kind) {}
  virtual ~CasePattern() {}
  CasePatternKind getKind() const { return kind; }

  /// Return true if this pattern has an X.
  virtual bool hasX() const { return false; }

  /// Return true if this pattern has an Z.
  virtual bool hasZ() const { return false; }

  virtual Attribute attr() const = 0;

private:
  const CasePatternKind kind;
};

class CaseDefaultPattern : public CasePattern {
public:
  using AttrType = mlir::UnitAttr;
  CaseDefaultPattern(MLIRContext *ctx)
      : CasePattern(CasePatternKind::CPK_default) {
    unitAttr = AttrType::get(ctx);
  }

  Attribute attr() const override { return unitAttr; }

  static bool classof(const CasePattern *S) {
    return S->getKind() == CPK_default;
  }

private:
  // The default pattern is recognized by a UnitAttr. This is needed since we
  // need to be able to determine the pattern type of a case based on an
  // attribute attached to the sv.case op.
  UnitAttr unitAttr;
};

class CaseBitPattern : public CasePattern {
public:
  // Return the number of bits in the pattern.
  size_t getWidth() const { return intAttr.getValue().getBitWidth() / 2; }

  /// Return the specified bit, bit 0 is the least significant bit.
  CasePatternBit getBit(size_t bitNumber) const;

  bool hasX() const override;
  bool hasZ() const override;

  Attribute attr() const override { return intAttr; }

  /// Get a CasePattern from a specified list of CasePatternBit.  Bits are
  /// specified in most least significant order - element zero is the least
  /// significant bit.
  CaseBitPattern(ArrayRef<CasePatternBit> bits, MLIRContext *context);

  /// Get a CasePattern for the specified constant value.
  CaseBitPattern(const APInt &value, MLIRContext *context);

  /// Get a CasePattern with a correctly encoded attribute.
  CaseBitPattern(IntegerAttr attr) : CasePattern(CPK_bit), intAttr(attr) {}

  static bool classof(const CasePattern *S) { return S->getKind() == CPK_bit; }

private:
  IntegerAttr intAttr;
};

class CaseEnumPattern : public CasePattern {
public:
  // Get a CasePattern for the specified enum value attribute.
  CaseEnumPattern(hw::EnumFieldAttr attr)
      : CasePattern(CPK_enum), enumAttr(attr) {}

  // Return the named value of this enumeration.
  StringRef getFieldValue() const;

  Attribute attr() const override { return enumAttr; }

  static bool classof(const CasePattern *S) { return S->getKind() == CPK_enum; }

private:
  hw::EnumFieldAttr enumAttr;
};

// This provides information about one case.
struct CaseInfo {
  std::unique_ptr<CasePattern> pattern;
  Block *block;
};

//===----------------------------------------------------------------------===//
// Other Supporting Logic
//===----------------------------------------------------------------------===//

/// Return true if the specified operation is in a procedural region.
LogicalResult verifyInProceduralRegion(Operation *op);
/// Return true if the specified operation is not in a procedural region.
LogicalResult verifyInNonProceduralRegion(Operation *op);

/// Signals that an operations regions are procedural.
template <typename ConcreteType>
class ProceduralRegion
    : public mlir::OpTrait::TraitBase<ConcreteType, ProceduralRegion> {
  static LogicalResult verifyTrait(Operation *op) {
    return mlir::OpTrait::impl::verifyAtLeastNRegions(op, 1);
  }
};

/// This class verifies that the specified op is located in a procedural region.
template <typename ConcreteType>
class ProceduralOp
    : public mlir::OpTrait::TraitBase<ConcreteType, ProceduralOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyInProceduralRegion(op);
  }
};

/// This class verifies that the specified op is not located in a procedural
/// region.
template <typename ConcreteType>
class NonProceduralOp
    : public mlir::OpTrait::TraitBase<ConcreteType, NonProceduralOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifyInNonProceduralRegion(op);
  }
};

/// This class provides a verifier for ops that are expecting their parent
/// to be one of the given parent ops
template <typename ConcreteType>
class VendorExtension
    : public mlir::OpTrait::TraitBase<ConcreteType, VendorExtension> {
public:
  static LogicalResult verifyTrait(Operation *op) { return success(); }
};

} // namespace sv
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/SV/SV.h.inc"

#endif // CIRCT_DIALECT_SV_OPS_H
