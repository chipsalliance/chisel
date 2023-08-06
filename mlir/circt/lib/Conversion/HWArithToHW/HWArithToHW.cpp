//===- HWArithToHW.cpp - HWArith to HW Lowering pass ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HWArith to HW Lowering Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWArithToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/ConversionPatterns.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace hwarith;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Function for setting the 'sv.namehint' attribute on 'newOp' based on any
// currently existing 'sv.namehint' attached to the source operation of 'value'.
// The user provides a callback which returns a new namehint based on the old
// namehint.
static void
improveNamehint(Value oldValue, Operation *newOp,
                llvm::function_ref<std::string(StringRef)> namehintCallback) {
  if (auto *sourceOp = oldValue.getDefiningOp()) {
    if (auto namehint =
            sourceOp->getAttrOfType<mlir::StringAttr>("sv.namehint")) {
      auto newNamehint = namehintCallback(namehint.strref());
      newOp->setAttr("sv.namehint",
                     StringAttr::get(oldValue.getContext(), newNamehint));
    }
  }
}

// Extract a bit range, specified via start bit and width, from a given value.
static Value extractBits(OpBuilder &builder, Location loc, Value value,
                         unsigned startBit, unsigned bitWidth) {
  SmallVector<Value, 1> result;
  builder.createOrFold<comb::ExtractOp>(result, loc, value, startBit, bitWidth);
  Value extractedValue = result[0];
  if (extractedValue != value) {
    // only change namehint if a new operation was created.
    auto *newOp = extractedValue.getDefiningOp();
    improveNamehint(value, newOp, [&](StringRef oldNamehint) {
      return (oldNamehint + "_" + std::to_string(startBit) + "_to_" +
              std::to_string(startBit + bitWidth))
          .str();
    });
  }
  return extractedValue;
}

// Perform the specified bit-extension (either sign- or zero-extension) for a
// given value to a desired target width.
static Value extendTypeWidth(OpBuilder &builder, Location loc, Value value,
                             unsigned targetWidth, bool signExtension) {
  unsigned sourceWidth = value.getType().getIntOrFloatBitWidth();
  unsigned extensionLength = targetWidth - sourceWidth;

  if (extensionLength == 0)
    return value;

  Value extensionBits;
  // https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/#no-complement-negate-zext-sext-operators
  if (signExtension) {
    // Sign extension
    Value highBit = extractBits(builder, loc, value,
                                /*startBit=*/sourceWidth - 1, /*bitWidth=*/1);
    SmallVector<Value, 1> result;
    builder.createOrFold<comb::ReplicateOp>(result, loc, highBit,
                                            extensionLength);
    extensionBits = result[0];
  } else {
    // Zero extension
    extensionBits = builder
                        .create<hw::ConstantOp>(
                            loc, builder.getIntegerType(extensionLength), 0)
                        ->getOpResult(0);
  }

  auto extOp = builder.create<comb::ConcatOp>(loc, extensionBits, value);
  improveNamehint(value, extOp, [&](StringRef oldNamehint) {
    return (oldNamehint + "_" + (signExtension ? "sext_" : "zext_") +
            std::to_string(targetWidth))
        .str();
  });

  return extOp->getOpResult(0);
}

static bool isSignednessType(Type type) {
  auto match =
      llvm::TypeSwitch<Type, bool>(type)
          .Case<IntegerType>([](auto type) { return !type.isSignless(); })
          .Case<hw::ArrayType>(
              [](auto type) { return isSignednessType(type.getElementType()); })
          .Case<hw::StructType>([](auto type) {
            return llvm::any_of(type.getElements(), [](auto element) {
              return isSignednessType(element.type);
            });
          })
          .Case<hw::InOutType>(
              [](auto type) { return isSignednessType(type.getElementType()); })
          .Default([](auto type) { return false; });

  return match;
}

static bool isSignednessAttr(Attribute attr) {
  if (auto typeAttr = attr.dyn_cast<TypeAttr>())
    return isSignednessType(typeAttr.getValue());
  return false;
}

/// Returns true if the given `op` is considered as legal for HWArith
/// conversion.
static bool isLegalOp(Operation *op) {
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    return llvm::none_of(funcOp.getArgumentTypes(), isSignednessType) &&
           llvm::none_of(funcOp.getResultTypes(), isSignednessType) &&
           llvm::none_of(funcOp.getFunctionBody().getArgumentTypes(),
                         isSignednessType);
  }

  auto attrs = llvm::map_range(op->getAttrs(), [](const NamedAttribute &attr) {
    return attr.getValue();
  });

  bool operandsOK = llvm::none_of(op->getOperandTypes(), isSignednessType);
  bool resultsOK = llvm::none_of(op->getResultTypes(), isSignednessType);
  bool attrsOK = llvm::none_of(attrs, isSignednessAttr);
  return operandsOK && resultsOK && attrsOK;
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(constOp,
                                                constOp.getConstantValue());
    return success();
  }
};
struct DivOpLowering : public OpConversionPattern<DivOp> {
  using OpConversionPattern<DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto isLhsTypeSigned =
        op.getOperand(0).getType().template cast<IntegerType>().isSigned();
    auto rhsType = op.getOperand(1).getType().template cast<IntegerType>();
    auto targetType = op.getResult().getType().template cast<IntegerType>();

    // comb.div* needs identical bitwidths for its operands and its result.
    // Hence, we need to calculate the minimal bitwidth that can be used to
    // represent the result as well as the operands without precision or sign
    // loss. The target size only depends on LHS and already handles the edge
    // cases where the bitwidth needs to be increased by 1. Thus, the targetType
    // is good enough for both the result as well as LHS.
    // The bitwidth for RHS is bit tricky. If the RHS is unsigned and we are
    // about to perform a signed division, then we need one additional bit to
    // avoid misinterpretation of RHS as a signed value!
    bool signedDivision = targetType.isSigned();
    unsigned extendSize = std::max(
        targetType.getWidth(),
        rhsType.getWidth() + (signedDivision && !rhsType.isSigned() ? 1 : 0));

    // Extend the operands
    Value lhsValue = extendTypeWidth(rewriter, loc, adaptor.getInputs()[0],
                                     extendSize, isLhsTypeSigned);
    Value rhsValue = extendTypeWidth(rewriter, loc, adaptor.getInputs()[1],
                                     extendSize, rhsType.isSigned());

    Value divResult;
    if (signedDivision)
      divResult = rewriter.create<comb::DivSOp>(loc, lhsValue, rhsValue, false)
                      ->getOpResult(0);
    else
      divResult = rewriter.create<comb::DivUOp>(loc, lhsValue, rhsValue, false)
                      ->getOpResult(0);

    // Carry over any attributes from the original div op.
    divResult.getDefiningOp()->setDialectAttrs(op->getDialectAttrs());

    // finally truncate back to the expected result size!
    Value truncateResult = extractBits(rewriter, loc, divResult, /*startBit=*/0,
                                       /*bitWidth=*/targetType.getWidth());
    rewriter.replaceOp(op, truncateResult);

    return success();
  }
};
} // namespace

namespace {
struct CastOpLowering : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = op.getIn().getType().cast<IntegerType>();
    auto sourceWidth = sourceType.getWidth();
    bool isSourceTypeSigned = sourceType.isSigned();
    auto targetWidth = op.getOut().getType().cast<IntegerType>().getWidth();

    Value replaceValue;
    if (sourceWidth == targetWidth) {
      // the width does not change, we are done here and can directly use the
      // lowering input value
      replaceValue = adaptor.getIn();
    } else if (sourceWidth < targetWidth) {
      // bit extensions needed, the type of extension required is determined by
      // the source type only!
      replaceValue = extendTypeWidth(rewriter, op.getLoc(), adaptor.getIn(),
                                     targetWidth, isSourceTypeSigned);
    } else {
      // bit truncation needed
      replaceValue = extractBits(rewriter, op.getLoc(), adaptor.getIn(),
                                 /*startBit=*/0, /*bitWidth=*/targetWidth);
    }
    rewriter.replaceOp(op, replaceValue);

    return success();
  }
};
} // namespace

namespace {

// Utility lowering function that maps a hwarith::ICmpPredicate predicate and
// the information whether the comparison contains signed values to the
// corresponding comb::ICmpPredicate.
static comb::ICmpPredicate lowerPredicate(ICmpPredicate pred, bool isSigned) {
#define _CREATE_HWARITH_ICMP_CASE(x)                                           \
  case ICmpPredicate::x:                                                       \
    return isSigned ? comb::ICmpPredicate::s##x : comb::ICmpPredicate::u##x

  switch (pred) {
  case ICmpPredicate::eq:
    return comb::ICmpPredicate::eq;

  case ICmpPredicate::ne:
    return comb::ICmpPredicate::ne;

    _CREATE_HWARITH_ICMP_CASE(lt);
    _CREATE_HWARITH_ICMP_CASE(ge);
    _CREATE_HWARITH_ICMP_CASE(le);
    _CREATE_HWARITH_ICMP_CASE(gt);
  }

#undef _CREATE_HWARITH_ICMP_CASE

  llvm_unreachable(
      "Missing hwarith::ICmpPredicate to comb::ICmpPredicate lowering");
  return comb::ICmpPredicate::eq;
}

struct ICmpOpLowering : public OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhsType = op.getLhs().getType().cast<IntegerType>();
    auto rhsType = op.getRhs().getType().cast<IntegerType>();
    IntegerType::SignednessSemantics cmpSignedness;
    const unsigned cmpWidth =
        inferAddResultType(cmpSignedness, lhsType, rhsType) - 1;

    ICmpPredicate pred = op.getPredicate();
    comb::ICmpPredicate combPred = lowerPredicate(
        pred, cmpSignedness == IntegerType::SignednessSemantics::Signed);

    const auto loc = op.getLoc();
    Value lhsValue = extendTypeWidth(rewriter, loc, adaptor.getLhs(), cmpWidth,
                                     lhsType.isSigned());
    Value rhsValue = extendTypeWidth(rewriter, loc, adaptor.getRhs(), cmpWidth,
                                     rhsType.isSigned());

    auto newOp = rewriter.replaceOpWithNewOp<comb::ICmpOp>(
        op, combPred, lhsValue, rhsValue, false);
    newOp->setDialectAttrs(op->getDialectAttrs());

    return success();
  }
};

template <class BinOp, class ReplaceOp>
struct BinaryOpLowering : public OpConversionPattern<BinOp> {
  using OpConversionPattern<BinOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<BinOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(BinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto isLhsTypeSigned =
        op.getOperand(0).getType().template cast<IntegerType>().isSigned();
    auto isRhsTypeSigned =
        op.getOperand(1).getType().template cast<IntegerType>().isSigned();
    auto targetWidth =
        op.getResult().getType().template cast<IntegerType>().getWidth();

    Value lhsValue = extendTypeWidth(rewriter, loc, adaptor.getInputs()[0],
                                     targetWidth, isLhsTypeSigned);
    Value rhsValue = extendTypeWidth(rewriter, loc, adaptor.getInputs()[1],
                                     targetWidth, isRhsTypeSigned);
    auto newOp =
        rewriter.replaceOpWithNewOp<ReplaceOp>(op, lhsValue, rhsValue, false);
    newOp->setDialectAttrs(op->getDialectAttrs());
    return success();
  }
};

} // namespace

Type HWArithToHWTypeConverter::removeSignedness(Type type) {
  auto it = conversionCache.find(type);
  if (it != conversionCache.end())
    return it->second.type;

  auto convertedType =
      llvm::TypeSwitch<Type, Type>(type)
          .Case<IntegerType>([](auto type) {
            if (type.isSignless())
              return type;
            return IntegerType::get(type.getContext(), type.getWidth());
          })
          .Case<hw::ArrayType>([this](auto type) {
            return hw::ArrayType::get(removeSignedness(type.getElementType()),
                                      type.getSize());
          })
          .Case<hw::StructType>([this](auto type) {
            // Recursively convert each element.
            llvm::SmallVector<hw::StructType::FieldInfo> convertedElements;
            for (auto element : type.getElements()) {
              convertedElements.push_back(
                  {element.name, removeSignedness(element.type)});
            }
            return hw::StructType::get(type.getContext(), convertedElements);
          })
          .Case<hw::InOutType>([this](auto type) {
            return hw::InOutType::get(removeSignedness(type.getElementType()));
          })
          .Default([](auto type) { return type; });

  return convertedType;
}

HWArithToHWTypeConverter::HWArithToHWTypeConverter() {
  // Pass any type through the signedness remover.
  addConversion([this](Type type) { return removeSignedness(type); });

  addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });

  addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

void circt::populateHWArithToHWConversionPatterns(
    HWArithToHWTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<ConstantOpLowering, CastOpLowering, ICmpOpLowering,
               BinaryOpLowering<AddOp, comb::AddOp>,
               BinaryOpLowering<SubOp, comb::SubOp>,
               BinaryOpLowering<MulOp, comb::MulOp>, DivOpLowering>(
      typeConverter, patterns.getContext());
}

namespace {

class HWArithToHWPass : public HWArithToHWBase<HWArithToHWPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal(isLegalOp);
    RewritePatternSet patterns(&getContext());
    HWArithToHWTypeConverter typeConverter;
    target.addIllegalDialect<HWArithDialect>();

    // Add HWArith-specific conversion patterns.
    populateHWArithToHWConversionPatterns(typeConverter, patterns);

    // ALL other operations are converted via the TypeConversionPattern which
    // will replace an operation to an identical operation with replaced
    // result types and operands.
    patterns.add<TypeConversionPattern>(typeConverter, patterns.getContext());

    // Apply a full conversion - all operations must either be legal, be caught
    // by one of the HWArith patterns or be converted by the
    // TypeConversionPattern.
    if (failed(applyFullConversion(module, target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> circt::createHWArithToHWPass() {
  return std::make_unique<HWArithToHWPass>();
}
