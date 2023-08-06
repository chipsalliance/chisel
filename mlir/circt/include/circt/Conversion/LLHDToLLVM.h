//===- LLHDToLLVM.h - LLHD to LLVM pass entry point -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the LLHDToLLVM pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_LLHDTOLLVM_LLHDTOLLVM_H
#define CIRCT_CONVERSION_LLHDTOLLVM_LLHDTOLLVM_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
} // namespace mlir

namespace circt {

/// Get the LLHD to LLVM type conversions
void populateLLHDToLLVMTypeConversions(mlir::LLVMTypeConverter &converter);

/// Get the LLHD to LLVM conversion patterns.
void populateLLHDToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          size_t &sigCounter,
                                          size_t &regCounter);

/// Create an LLHD to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createConvertLLHDToLLVMPass();

} // namespace circt

#endif // CIRCT_CONVERSION_LLHDTOLLVM_LLHDTOLLVM_H
