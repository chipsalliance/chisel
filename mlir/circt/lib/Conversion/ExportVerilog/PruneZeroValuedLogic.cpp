//===- PruneZeroValuedLogic.cpp - Prune zero-valued logic -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform removes zero-valued logic from a `hw.module`.
//
//===----------------------------------------------------------------------===//

#include "ExportVerilogInternals.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace hw;

static bool noI0Type(TypeRange types) {
  return llvm::none_of(
      types, [](Type type) { return ExportVerilog::isZeroBitType(type); });
}

static bool noI0TypedValue(ValueRange values) {
  return noI0Type(values.getTypes());
}

namespace {

class PruneTypeConverter : public mlir::TypeConverter {
public:
  PruneTypeConverter() {
    addConversion([&](Type type, SmallVectorImpl<Type> &results) {
      if (!ExportVerilog::isZeroBitType(type))
        results.push_back(type);
      return success();
    });
  }
};

template <typename TOp>
struct NoI0OperandsConversionPattern : public OpConversionPattern<TOp> {
public:
  using OpConversionPattern<TOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<TOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(TOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (noI0TypedValue(adaptor.getOperands()))
      return failure();

    // Part of i0-typed logic - prune it!
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename... TOp>
static void addNoI0OperandsLegalizationPattern(ConversionTarget &target) {
  target.addDynamicallyLegalOp<TOp...>(
      [&](auto op) { return noI0TypedValue(op->getOperands()); });
}

template <>
struct NoI0OperandsConversionPattern<comb::ICmpOp>
    : public OpConversionPattern<comb::ICmpOp> {
public:
  using OpConversionPattern<comb::ICmpOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<comb::ICmpOp>::OpAdaptor;

  // Returns the result of applying the predicate when the LHS and RHS are the
  // exact same value.
  static bool
  applyCmpPredicateToEqualOperands(circt::comb::ICmpPredicate predicate) {
    switch (predicate) {
    case circt::comb::ICmpPredicate::eq:
    case circt::comb::ICmpPredicate::sle:
    case circt::comb::ICmpPredicate::sge:
    case circt::comb::ICmpPredicate::ule:
    case circt::comb::ICmpPredicate::uge:
    case circt::comb::ICmpPredicate::ceq:
    case circt::comb::ICmpPredicate::weq:
      return true;
    case circt::comb::ICmpPredicate::ne:
    case circt::comb::ICmpPredicate::slt:
    case circt::comb::ICmpPredicate::sgt:
    case circt::comb::ICmpPredicate::ult:
    case circt::comb::ICmpPredicate::ugt:
    case circt::comb::ICmpPredicate::cne:
    case circt::comb::ICmpPredicate::wne:
      return false;
    }
    llvm_unreachable("unknown comparison predicate");
  }

  LogicalResult
  matchAndRewrite(comb::ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (noI0TypedValue(adaptor.getOperands()))
      return failure();

    // Caluculate the result of i0 value comparison.
    bool result = applyCmpPredicateToEqualOperands(op.getPredicate());

    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, APInt(1, result, /*isSigned=*/false));
    return success();
  }
};

template <>
struct NoI0OperandsConversionPattern<comb::ParityOp>
    : public OpConversionPattern<comb::ParityOp> {
public:
  using OpConversionPattern<comb::ParityOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<comb::ParityOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(comb::ParityOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (noI0TypedValue(adaptor.getOperands()))
      return failure();

    // The value of "comb.parity i0" is 0.
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, APInt(1, 0, /*isSigned=*/false));
    return success();
  }
};

template <>
struct NoI0OperandsConversionPattern<comb::ConcatOp>
    : public OpConversionPattern<comb::ConcatOp> {
public:
  using OpConversionPattern<comb::ConcatOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<comb::ConcatOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(comb::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace an i0 value with i0 constant.
    if (op.getType().isInteger(0)) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(
          op, APInt(1, 0, /*isSigned=*/false));
      return success();
    }

    if (noI0TypedValue(adaptor.getOperands()))
      return failure();

    // Filter i0 operands and create a new concat op.
    SmallVector<Value> newOperands;
    llvm::copy_if(op.getOperands(), std::back_inserter(newOperands),
                  [](auto op) { return !op.getType().isInteger(0); });
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, newOperands);
    return success();
  }
};

// A generic pruning pattern which prunes any operation which has an operand
// with an i0 typed value. Similarly, an operation is legal if all of its
// operands are not i0 typed.
template <typename TOp>
struct NoI0OperandPruningPattern {
  using ConversionPattern = NoI0OperandsConversionPattern<TOp>;
  static void addLegalizer(ConversionTarget &target) {
    addNoI0OperandsLegalizationPattern<TOp>(target);
  }
};

// The NoI0ResultsConversionPattern will aggressively remove any operation
// which has a zero-width result. Furthermore, it will recursively erase any
// downstream users of the operation.
template <typename TOp>
struct NoI0ResultsConversionPattern : public OpConversionPattern<TOp> {
public:
  using OpConversionPattern<TOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<TOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(TOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (noI0TypedValue(op->getResults()))
      return failure();

    // Part of i0-typed logic - prune!
    assert(op->getNumResults() == 1 &&
           "expected single result if using rewriter.replaceOpWith");
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, APInt(0, 0, /*isSigned=*/false));
    return success();
  }
};

template <typename... TOp>
static void addNoI0ResultsLegalizationPattern(ConversionTarget &target) {
  target.addDynamicallyLegalOp<TOp...>(
      [&](auto op) { return noI0TypedValue(op->getResults()); });
}

// A generic pruning pattern which prunes any operation that returns an i0
// value.
template <typename TOp>
struct NoI0ResultPruningPattern {
  using ConversionPattern = NoI0ResultsConversionPattern<TOp>;
  static void addLegalizer(ConversionTarget &target) {
    addNoI0ResultsLegalizationPattern<TOp>(target);
  }
};

// Adds a pruning pattern to the conversion target. TPattern is expected to
// provides ConversionPattern definition and an addLegalizer function.
template <typename... TPattern>
static void addPruningPattern(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              PruneTypeConverter &typeConverter) {
  (patterns.add<typename TPattern::ConversionPattern>(typeConverter,
                                                      patterns.getContext()),
   ...);
  (TPattern::addLegalizer(target), ...);
}

template <typename... TOp>
static void addNoI0ResultPruningPattern(ConversionTarget &target,
                                        RewritePatternSet &patterns,
                                        PruneTypeConverter &typeConverter) {
  (patterns.add<typename NoI0ResultPruningPattern<TOp>::ConversionPattern>(
       typeConverter, patterns.getContext()),
   ...);
  (NoI0ResultPruningPattern<TOp>::addLegalizer(target), ...);
}

} // namespace

void ExportVerilog::pruneZeroValuedLogic(hw::HWModuleOp module) {
  ConversionTarget target(*module.getContext());
  RewritePatternSet patterns(module.getContext());
  PruneTypeConverter typeConverter;

  target.addLegalDialect<sv::SVDialect, comb::CombDialect, hw::HWDialect>();
  addPruningPattern<NoI0OperandPruningPattern<sv::PAssignOp>,
                    NoI0OperandPruningPattern<sv::BPAssignOp>,
                    NoI0OperandPruningPattern<sv::AssignOp>,
                    NoI0OperandPruningPattern<comb::ICmpOp>,
                    NoI0OperandPruningPattern<comb::ParityOp>,
                    NoI0OperandPruningPattern<comb::ConcatOp>>(target, patterns,
                                                               typeConverter);

  addNoI0ResultPruningPattern<
      // SV ops
      sv::WireOp, sv::RegOp, sv::ReadInOutOp,
      // Prune all zero-width combinational logic.
      comb::AddOp, comb::AndOp, comb::DivSOp, comb::DivUOp, comb::ExtractOp,
      comb::ModSOp, comb::ModUOp, comb::MulOp, comb::MuxOp, comb::OrOp,
      comb::ReplicateOp, comb::ShlOp, comb::ShrSOp, comb::ShrUOp, comb::SubOp,
      comb::XorOp>(target, patterns, typeConverter);

  (void)applyPartialConversion(module, target, std::move(patterns));
}
