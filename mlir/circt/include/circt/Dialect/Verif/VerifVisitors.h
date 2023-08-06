//===- VerifVisitors.h - Verif dialect visitors -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_VERIF_VERIFVISITORS_H
#define CIRCT_DIALECT_VERIF_VERIFVISITORS_H

#include "circt/Dialect/Verif/VerifOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace verif {
template <typename ConcreteType, typename ResultType = void,
          typename... ExtraArgs>
class Visitor {
public:
  ResultType dispatchVerifVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<AssertOp, AssumeOp, CoverOp>([&](auto op) -> ResultType {
          return thisCast->visitVerif(op, args...);
        })
        .Default([&](auto) -> ResultType {
          return thisCast->visitInvalidVerif(op, args...);
        });
  }

  /// This callback is invoked on any non-verif operations.
  ResultType visitInvalidVerif(Operation *op, ExtraArgs... args) {
    op->emitOpError("is not a verif operation");
    abort();
  }

  /// This callback is invoked on any verif operations that were not handled by
  /// their concrete `visitVerif(...)` callback.
  ResultType visitUnhandledVerif(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE, OPKIND)                                                 \
  ResultType visitVerif(OPTYPE op, ExtraArgs... args) {                        \
    return static_cast<ConcreteType *>(this)->visit##OPKIND##Verif(op,         \
                                                                   args...);   \
  }

  HANDLE(AssertOp, Unhandled);
  HANDLE(AssumeOp, Unhandled);
  HANDLE(CoverOp, Unhandled);
#undef HANDLE
};

} // namespace verif
} // namespace circt

#endif // CIRCT_DIALECT_VERIF_VERIFVISITORS_H
