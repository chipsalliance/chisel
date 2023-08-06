//===- HWToLLVM.h - HW to LLVM pass entry point -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the HWToLLVM pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HWTOLLVM_HWTOLLVM_H
#define CIRCT_CONVERSION_HWTOLLVM_HWTOLLVM_H

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
namespace LLVM {
class GlobalOp;
} // namespace LLVM
} // namespace mlir

namespace circt {
class Namespace;

struct HWToLLVMEndianessConverter {
  /// Convert an index into a HW ArrayType or StructType to LLVM Endianess.
  static uint32_t convertToLLVMEndianess(Type type, uint32_t index);

  /// Get the index of a specific StructType field in the LLVM lowering of the
  /// StructType
  static uint32_t llvmIndexOfStructField(hw::StructType type,
                                         StringRef fieldName);
};

/// Get the HW to LLVM type conversions.
void populateHWToLLVMTypeConversions(mlir::LLVMTypeConverter &converter);

/// Get the HW to LLVM conversion patterns.
void populateHWToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, RewritePatternSet &patterns,
    Namespace &globals,
    DenseMap<std::pair<Type, ArrayAttr>, mlir::LLVM::GlobalOp>
        &constAggregateGlobalsMap);

/// Create an HW to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createConvertHWToLLVMPass();

} // namespace circt

#endif // CIRCT_CONVERSION_HWTOLLVM_HWTOLLVM_H
