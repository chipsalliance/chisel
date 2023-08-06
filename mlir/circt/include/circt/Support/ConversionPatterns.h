//===- ConversionPatterns.h - Common Conversion patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_CONVERSIONPATTERNS_H
#define CIRCT_SUPPORT_CONVERSIONPATTERNS_H

#include "circt/Support/LLVM.h"

#include "mlir/Transforms/DialectConversion.h"

namespace circt {

/// Generic pattern which replaces an operation by one of the same operation
/// name, but with converted attributes, operands, and result types to eliminate
/// illegal types. Uses generic builders based on OperationState to make sure
/// that this pattern can apply to _any_ operation.
///
/// Useful when a conversion can be entirely defined by a TypeConverter.
struct TypeConversionPattern : public mlir::ConversionPattern {
public:
  TypeConversionPattern(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}
  using ConversionPattern::ConversionPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace circt

#endif // CIRCT_SUPPORT_CONVERSIONPATTERNS_H
