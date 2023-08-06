//===- HWArithToHW.h - HWArith to HW conversions ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the HWArithToHW pass
// constructor.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HWARITHTOHW_HWARITHTOHW_H
#define CIRCT_CONVERSION_HWARITHTOHW_HWARITHTOHW_H

#include "circt/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

/// A helper type converter class that automatically populates the relevant
/// materializations and type conversions for converting HWArith to HW.
class HWArithToHWTypeConverter : public mlir::TypeConverter {
public:
  HWArithToHWTypeConverter();

  // A function which recursively converts any integer type with signedness
  // semantics to a signless counterpart.
  mlir::Type removeSignedness(mlir::Type type);

  // Returns true if any subtype in 'type' has signedness semantics.
  bool hasSignednessSemantics(mlir::Type type);
  bool hasSignednessSemantics(mlir::TypeRange types);

private:
  // Memoizations for signedness info and conversions.
  struct ConvertedType {
    mlir::Type type;
    bool hadSignednessSemantics;
  };
  llvm::DenseMap<mlir::Type, ConvertedType> conversionCache;
};

/// Get the HWArith to HW conversion patterns.
void populateHWArithToHWConversionPatterns(
    HWArithToHWTypeConverter &typeConverter, RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createHWArithToHWPass();
} // namespace circt

#endif // CIRCT_CONVERSION_HWARITHTOHW_HWARITHTOHW_H
