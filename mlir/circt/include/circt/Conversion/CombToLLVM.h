//===- CombToLLVM.h - Comb to LLVM pass entry point -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the CombToLLVM pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_COMBTOLLVM_COMBTOLLVM_H
#define CIRCT_CONVERSION_COMBTOLLVM_COMBTOLLVM_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
} // namespace mlir

namespace circt {

/// Get the Comb to LLVM conversion patterns.
void populateCombToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);

} // namespace circt

#endif // CIRCT_CONVERSION_COMBTOLLVM_COMBTOLLVM_H
