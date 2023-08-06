//===- CHIRRTLVisitors.h - CHIRRTL Dialect Visitors -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines visitors that make it easier to work with CHIRRTL IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_CHIRRTLVISITORS_H
#define CIRCT_DIALECT_FIRRTL_CHIRRTLVISITORS_H

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace chirrtl {

/// CHIRRTLVisitor is a visitor for CHIRRTL operations.
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class CHIRRTLVisitor {
public:
  ResultType dispatchCHIRRTLVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<CombMemOp, MemoryPortOp, MemoryDebugPortOp,
                       MemoryPortAccessOp, SeqMemOp>(
            [&](auto opNode) -> ResultType {
              return thisCast->visitCHIRRTL(opNode, args...);
            })
        .Default([&](auto expr) -> ResultType {
          return thisCast->visitInvalidCHIRRTL(op, args...);
        });
  }

  /// This callback is invoked on any non-CHIRRTL operations.
  ResultType visitInvalidCHIRRTL(Operation *op, ExtraArgs... args) {
    op->emitOpError("unknown chirrtl op");
    abort();
  }

  /// This callback is invoked on any CHIRRTL operations that are not handled
  /// by the concrete visitor.
  ResultType visitUnhandledCHIRRTL(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitCHIRRTL(OPTYPE op, ExtraArgs... args) {                      \
    return static_cast<ConcreteType *>(this)->visitUnhandledCHIRRTL(op,        \
                                                                    args...);  \
  }

  HANDLE(CombMemOp);
  HANDLE(MemoryPortOp);
  HANDLE(MemoryDebugPortOp);
  HANDLE(MemoryPortAccessOp);
  HANDLE(SeqMemOp);
#undef HANDLE
};

} // namespace chirrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_CHIRRTLVISITORS_H
