//===- GenericReductions.h - Generic reduction patterns ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_REDUCE_GENERICREDUCTIONS_H
#define CIRCT_REDUCE_GENERICREDUCTIONS_H

#include "circt/Reduce/Reduction.h"

namespace circt {

/// Populate reduction patterns that are not specific to certain operations or
/// dialects
void populateGenericReducePatterns(MLIRContext *context,
                                   ReducePatternSet &patterns);

} // namespace circt

#endif // CIRCT_REDUCE_GENERICREDUCTIONS_H
