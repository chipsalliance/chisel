//===- HWVisitors.h - HW Dialect Visitors ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with HW IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWVISITORS_H
#define CIRCT_DIALECT_HW_HWVISITORS_H

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace hw {

/// This helps visit TypeOp nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class TypeOpVisitor {
public:
  ResultType dispatchTypeOpVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<ConstantOp, AggregateConstantOp,
                       // Array operations
                       ArraySliceOp, ArrayCreateOp, ArrayConcatOp, ArrayGetOp,
                       // Struct operations
                       StructCreateOp, StructExtractOp, StructInjectOp,
                       // Union operations
                       UnionCreateOp, UnionExtractOp,
                       // Cast operation
                       BitcastOp, ParamValueOp,
                       // Enum operations
                       EnumConstantOp, EnumCmpOp>([&](auto expr) -> ResultType {
          return thisCast->visitTypeOp(expr, args...);
        })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidTypeOp(op, args...);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidTypeOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown HW combinational node");
    abort();
  }

  /// This callback is invoked on any combinational operations that are not
  /// handled by the concrete visitor.
  ResultType visitUnhandledTypeOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitTypeOp(OPTYPE op, ExtraArgs... args) {                       \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##TypeOp(op,        \
                                                                    args...);  \
  }

  HANDLE(ConstantOp, Unhandled);
  HANDLE(AggregateConstantOp, Unhandled);
  HANDLE(BitcastOp, Unhandled);
  HANDLE(ParamValueOp, Unhandled);
  HANDLE(StructCreateOp, Unhandled);
  HANDLE(StructExtractOp, Unhandled);
  HANDLE(StructInjectOp, Unhandled);
  HANDLE(UnionCreateOp, Unhandled);
  HANDLE(UnionExtractOp, Unhandled);
  HANDLE(ArraySliceOp, Unhandled);
  HANDLE(ArrayGetOp, Unhandled);
  HANDLE(ArrayCreateOp, Unhandled);
  HANDLE(ArrayConcatOp, Unhandled);
  HANDLE(EnumCmpOp, Unhandled);
  HANDLE(EnumConstantOp, Unhandled);
#undef HANDLE
};

/// This helps visit TypeOp nodes.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class StmtVisitor {
public:
  ResultType dispatchStmtVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<OutputOp, InstanceOp, TypeScopeOp, TypedeclOp>(
            [&](auto expr) -> ResultType {
              return thisCast->visitStmt(expr, args...);
            })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidStmt(op, args...);
        });
  }

  /// This callback is invoked on any non-expression operations.
  ResultType visitInvalidStmt(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown hw statement");
    abort();
  }

  /// This callback is invoked on any combinational operations that are not
  /// handled by the concrete visitor.
  ResultType visitUnhandledTypeOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

  /// This fallback is invoked on any binary node that isn't explicitly handled.
  /// The default implementation delegates to the 'unhandled' fallback.
  ResultType visitBinaryTypeOp(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledTypeOp(op, args...);
  }

  ResultType visitUnaryTypeOp(Operation *op, ExtraArgs... args) {
    return static_cast<ConcreteType *>(this)->visitUnhandledTypeOp(op, args...);
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitStmt(OPTYPE op, ExtraArgs... args) {                         \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##Stmt(op,          \
                                                                  args...);    \
  }

  // Basic nodes.
  HANDLE(OutputOp, Unhandled);
  HANDLE(InstanceOp, Unhandled);
  HANDLE(TypeScopeOp, Unhandled);
  HANDLE(TypedeclOp, Unhandled);
#undef HANDLE
};

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_HWVISITORS_H
