//===- ArcReductions.h - Arc reduction interface declaration ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCREDUCTIONS_H
#define CIRCT_DIALECT_ARC_ARCREDUCTIONS_H

#include "circt/Reduce/Reduction.h"

namespace circt {
namespace arc {

/// A dialect interface to provide reduction patterns to a reducer tool.
struct ArcReducePatternDialectInterface : public ReducePatternDialectInterface {
  using ReducePatternDialectInterface::ReducePatternDialectInterface;
  void populateReducePatterns(circt::ReducePatternSet &patterns) const override;
};

/// Register the Arc Reduction pattern dialect interface to the given registry.
void registerReducePatternDialectInterface(mlir::DialectRegistry &registry);

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_ARCREDUCTIONS_H
