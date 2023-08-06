//===- HWReductions.h - HW reduction interface declaration ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWREDUCTIONS_H
#define CIRCT_DIALECT_HW_HWREDUCTIONS_H

#include "circt/Reduce/Reduction.h"

namespace circt {
namespace hw {

/// A dialect interface to provide reduction patterns to a reducer tool.
struct HWReducePatternDialectInterface : public ReducePatternDialectInterface {
  using ReducePatternDialectInterface::ReducePatternDialectInterface;
  void populateReducePatterns(circt::ReducePatternSet &patterns) const override;
};

/// Register the HW Reduction pattern dialect interface to the given registry.
void registerReducePatternDialectInterface(mlir::DialectRegistry &registry);

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_HWREDUCTIONS_H
