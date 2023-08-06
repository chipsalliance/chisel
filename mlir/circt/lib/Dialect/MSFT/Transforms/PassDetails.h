//===- PassDetails.h - MSFT pass class details ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different MSFT passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_MSFT_TRANSFORMS_PASSDETAILS_H
#define DIALECT_MSFT_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace msft {

/// TODO: Migrate these to some sort of OpInterface shared with hw.
bool isAnyModule(Operation *module);

/// Utility for creating {0, 1, 2, ..., size}.
SmallVector<unsigned> makeSequentialRange(unsigned size);

/// Try to get a "good" name for the given Value.
StringRef getValueName(Value v, const SymbolCache &syms, std::string &buff);

/// Generic pattern for removing an op during pattern conversion.
template <typename OpTy>
struct RemoveOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

#define GEN_PASS_CLASSES
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"

} // namespace msft
} // namespace circt

#endif // DIALECT_MSFT_TRANSFORMS_PASSDETAILS_H
