//===- PassDetail.h - Transforms Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef TRANSFORMS_PASSDETAIL_H
#define TRANSFORMS_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
class MemrefDialect;
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // namespace arith

namespace memref {
class MemRefDialect;
} // end namespace memref

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace func {
class FuncDialect;
} // namespace func
} // end namespace mlir

namespace circt {
namespace hw {
class HWModuleLike;
class HWDialect;
} // namespace hw

namespace comb {
class CombDialect;
} // namespace comb

#define GEN_PASS_CLASSES
#include "circt/Transforms/Passes.h.inc"

} // namespace circt

#endif // TRANSFORMS_PASSDETAIL_H
