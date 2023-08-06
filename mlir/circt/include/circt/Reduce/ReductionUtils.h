//===- ReductionUtils.h - Reduction pattern utilities -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_REDUCE_REDUCTIONUTILS_H
#define CIRCT_REDUCE_REDUCTIONUTILS_H

#include "circt/Support/LLVM.h"

namespace circt {
// Forward declarations.
struct Reduction;

namespace reduce {

/// Starting at the given `op`, traverse through it and its operands and erase
/// operations that have no more uses.
void pruneUnusedOps(Operation *initialOp, Reduction &reduction);

} // namespace reduce
} // namespace circt

#endif // CIRCT_REDUCE_REDUCTIONUTILS_H
