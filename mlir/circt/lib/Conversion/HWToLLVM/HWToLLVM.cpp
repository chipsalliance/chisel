//===- HWToLLVM.cpp - HW to LLVM Conversion Pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to LLVM Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToLLVM.h"
#include "../PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Endianess Converter
//===----------------------------------------------------------------------===//

uint32_t
circt::HWToLLVMEndianessConverter::convertToLLVMEndianess(Type type,
                                                          uint32_t index) {
  // This is hardcoded for little endian machines for now.
  return TypeSwitch<Type, uint32_t>(type)
      .Case<hw::ArrayType>(
          [&](hw::ArrayType ty) { return ty.getSize() - index - 1; })
      .Case<hw::StructType>([&](hw::StructType ty) {
        return ty.getElements().size() - index - 1;
      });
}

uint32_t
circt::HWToLLVMEndianessConverter::llvmIndexOfStructField(hw::StructType type,
                                                          StringRef fieldName) {
  auto fieldIter = type.getElements();
  size_t index = 0;

  for (const auto *iter = fieldIter.begin(); iter != fieldIter.end(); ++iter) {
    if (iter->name == fieldName) {
      return HWToLLVMEndianessConverter::convertToLLVMEndianess(type, index);
    }
    ++index;
  }

  // Verifier of StructExtractOp has to ensure that the field name is indeed
  // present.
  llvm_unreachable("Field name attribute of hw::StructExtractOp invalid");
  return 0;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Create a zext operation by one bit on the given value. This is useful when
/// passing unsigned indexes to a GEP instruction, which treats indexes as
/// signed values, to avoid unexpected "sign overflows".
static Value zextByOne(Location loc, ConversionPatternRewriter &rewriter,
                       Value value) {
  auto valueTy = value.getType();
  auto zextTy = IntegerType::get(valueTy.getContext(),
                                 valueTy.getIntOrFloatBitWidth() + 1);
  return rewriter.create<LLVM::ZExtOp>(loc, zextTy, value);
}

//===----------------------------------------------------------------------===//
// Extraction operation conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert a StructExplodeOp to the LLVM dialect.
/// Pattern: struct_explode(input) =>
///          struct_extract(input, structElements_index(index)) ...
struct StructExplodeOpConversion
    : public ConvertOpToLLVMPattern<hw::StructExplodeOp> {
  using ConvertOpToLLVMPattern<hw::StructExplodeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructExplodeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Value> replacements;

    for (size_t i = 0, e = adaptor.getInput()
                               .getType()
                               .cast<LLVM::LLVMStructType>()
                               .getBody()
                               .size();
         i < e; ++i)

      replacements.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), adaptor.getInput(),
          HWToLLVMEndianessConverter::convertToLLVMEndianess(
              op.getInput().getType(), i)));

    rewriter.replaceOp(op, replacements);
    return success();
  }
};
} // namespace

namespace {
/// Convert a StructExtractOp to LLVM dialect.
/// Pattern: struct_extract(input, fieldname) =>
///   extractvalue(input, fieldname_to_index(fieldname))
struct StructExtractOpConversion
    : public ConvertOpToLLVMPattern<hw::StructExtractOp> {
  using ConvertOpToLLVMPattern<hw::StructExtractOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    uint32_t fieldIndex = HWToLLVMEndianessConverter::llvmIndexOfStructField(
        op.getInput().getType().cast<hw::StructType>(), op.getField());
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                      fieldIndex);
    return success();
  }
};
} // namespace

namespace {
/// Convert an ArrayGetOp to the LLVM dialect.
/// Pattern: array_get(input, index) =>
///   load(gep(store(input, alloca), zext(index)))
struct ArrayGetOpConversion : public ConvertOpToLLVMPattern<hw::ArrayGetOp> {
  using ConvertOpToLLVMPattern<hw::ArrayGetOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value arrPtr;
    if (auto load = adaptor.getInput().getDefiningOp<LLVM::LoadOp>()) {
      // In this case the array was loaded from an existing address, so we can
      // just grab that address instead of reallocating the array on the stack.
      arrPtr = load.getAddr();
    } else {
      auto oneC = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(1));
      arrPtr = rewriter.create<LLVM::AllocaOp>(
          op->getLoc(),
          LLVM::LLVMPointerType::get(adaptor.getInput().getType()), oneC,
          /*alignment=*/4);
      rewriter.create<LLVM::StoreOp>(op->getLoc(), adaptor.getInput(), arrPtr);
    }

    auto elemTy = typeConverter->convertType(op.getResult().getType());

    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), IntegerType::get(rewriter.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    auto zextIndex = zextByOne(op->getLoc(), rewriter, op.getIndex());
    auto gep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(elemTy), arrPtr,
        ArrayRef<Value>({zeroC, zextIndex}));
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elemTy, gep);

    return success();
  }
};
} // namespace

namespace {
/// Convert an ArraySliceOp to the LLVM dialect.
/// Pattern: array_slice(input, lowIndex) =>
///   load(bitcast(gep(store(input, alloca), zext(lowIndex))))
struct ArraySliceOpConversion
    : public ConvertOpToLLVMPattern<hw::ArraySliceOp> {
  using ConvertOpToLLVMPattern<hw::ArraySliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArraySliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto dstTy = typeConverter->convertType(op.getDst().getType());
    auto elemTy = typeConverter->convertType(
        op.getDst().getType().cast<hw::ArrayType>().getElementType());

    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto oneC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

    auto arrPtr = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(adaptor.getInput().getType()),
        oneC,
        /*alignment=*/4);

    rewriter.create<LLVM::StoreOp>(op->getLoc(), adaptor.getInput(), arrPtr);

    auto zextIndex = zextByOne(op->getLoc(), rewriter, op.getLowIndex());

    auto gep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(elemTy), arrPtr,
        ArrayRef<Value>({zeroC, zextIndex}));

    auto cast = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(dstTy), gep);

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dstTy, cast);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Insertion operations conversion
//===----------------------------------------------------------------------===//

namespace {
/// Convert a StructInjectOp to LLVM dialect.
/// Pattern: struct_inject(input, index, value) =>
///   insertvalue(input, value, index)
struct StructInjectOpConversion
    : public ConvertOpToLLVMPattern<hw::StructInjectOp> {
  using ConvertOpToLLVMPattern<hw::StructInjectOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    uint32_t fieldIndex = HWToLLVMEndianessConverter::llvmIndexOfStructField(
        op.getInput().getType().cast<hw::StructType>(),
        op.getFieldAttr().getValue());

    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, adaptor.getInput(), op.getNewValue(), fieldIndex);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Concat operations conversion
//===----------------------------------------------------------------------===//

namespace {
/// Lower an ArrayConcatOp operation to the LLVM dialect.
struct ArrayConcatOpConversion
    : public ConvertOpToLLVMPattern<hw::ArrayConcatOp> {
  using ConvertOpToLLVMPattern<hw::ArrayConcatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    hw::ArrayType arrTy = op.getResult().getType().cast<hw::ArrayType>();
    Type resultTy = typeConverter->convertType(arrTy);

    Value arr = rewriter.create<LLVM::UndefOp>(op->getLoc(), resultTy);

    // Attention: j is hardcoded for little endian machines.
    size_t j = op.getInputs().size() - 1, k = 0;

    for (size_t i = 0, e = arrTy.getSize(); i < e; ++i) {
      Value element = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), adaptor.getInputs()[j], k);
      arr = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), arr, element, i);

      ++k;
      if (k >= op.getInputs()[j].getType().cast<hw::ArrayType>().getSize()) {
        k = 0;
        --j;
      }
    }

    rewriter.replaceOp(op, arr);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Bitwise conversions
//===----------------------------------------------------------------------===//

namespace {
/// Lower an ArrayConcatOp operation to the LLVM dialect.
/// Pattern: hw.bitcast(input) ==> load(bitcast_ptr(store(input, alloca)))
/// This is necessary because we cannot bitcast aggregate types directly in
/// LLVMIR.
struct BitcastOpConversion : public ConvertOpToLLVMPattern<hw::BitcastOp> {
  using ConvertOpToLLVMPattern<hw::BitcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultTy = typeConverter->convertType(op.getResult().getType());

    auto oneC = rewriter.createOrFold<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

    auto ptr = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(adaptor.getInput().getType()),
        oneC,
        /*alignment=*/4);

    rewriter.create<LLVM::StoreOp>(op->getLoc(), adaptor.getInput(), ptr);

    auto cast = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(resultTy), ptr);

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultTy, cast);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Value creation conversions
//===----------------------------------------------------------------------===//

namespace {
struct HWConstantOpConversion : public ConvertToLLVMPattern {
  explicit HWConstantOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(hw::ConstantOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operand,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the ConstOp.
    auto constOp = cast<hw::ConstantOp>(op);
    // Get the converted llvm type.
    auto intType = typeConverter->convertType(constOp.getValueAttr().getType());
    // Replace the operation with an llvm constant op.
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, intType,
                                                  constOp.getValueAttr());

    return success();
  }
};
} // namespace

namespace {
/// Convert an ArrayCreateOp with dynamic elements to the LLVM dialect. An
/// equivalent and initialized llvm dialect array type is generated.
struct HWDynamicArrayCreateOpConversion
    : public ConvertOpToLLVMPattern<hw::ArrayCreateOp> {
  using ConvertOpToLLVMPattern<hw::ArrayCreateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayTy = typeConverter->convertType(op->getResult(0).getType());
    assert(arrayTy);

    Value arr = rewriter.create<LLVM::UndefOp>(op->getLoc(), arrayTy);
    for (size_t i = 0, e = op.getInputs().size(); i < e; ++i) {
      Value input =
          adaptor
              .getInputs()[HWToLLVMEndianessConverter::convertToLLVMEndianess(
                  op.getResult().getType(), i)];
      arr = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), arr, input, i);
    }

    rewriter.replaceOp(op, arr);
    return success();
  }
};
} // namespace

namespace {

/// Convert an ArrayCreateOp with constant elements to the LLVM dialect. An
/// equivalent and initialized llvm dialect array type is generated.
class AggregateConstantOpConversion
    : public ConvertOpToLLVMPattern<hw::AggregateConstantOp> {
  using ConvertOpToLLVMPattern<hw::AggregateConstantOp>::ConvertOpToLLVMPattern;

  bool containsArrayAndStructAggregatesOnly(Type type) const;

  bool isMultiDimArrayOfIntegers(Type type,
                                 SmallVectorImpl<int64_t> &dims) const;

  void flatten(Type type, Attribute attr,
               SmallVectorImpl<Attribute> &output) const;

  Value constructAggregate(OpBuilder &builder, TypeConverter &typeConverter,
                           Location loc, Type type, Attribute data) const;

public:
  explicit AggregateConstantOpConversion(
      LLVMTypeConverter &typeConverter,
      DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp>
          &constAggregateGlobalsMap,
      Namespace &globals)
      : ConvertOpToLLVMPattern(typeConverter),
        constAggregateGlobalsMap(constAggregateGlobalsMap), globals(globals) {}

  LogicalResult
  matchAndRewrite(hw::AggregateConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp>
      &constAggregateGlobalsMap;
  Namespace &globals;
};
} // namespace

namespace {
/// Convert a StructCreateOp operation to the LLVM dialect. An equivalent and
/// initialized llvm dialect struct type is generated.
struct HWStructCreateOpConversion
    : public ConvertOpToLLVMPattern<hw::StructCreateOp> {
  using ConvertOpToLLVMPattern<hw::StructCreateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resTy = typeConverter->convertType(op.getResult().getType());

    Value tup = rewriter.create<LLVM::UndefOp>(op->getLoc(), resTy);
    for (size_t i = 0, e = resTy.cast<LLVM::LLVMStructType>().getBody().size();
         i < e; ++i) {
      Value input =
          adaptor.getInput()[HWToLLVMEndianessConverter::convertToLLVMEndianess(
              op.getResult().getType(), i)];
      tup = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), tup, input, i);
    }

    rewriter.replaceOp(op, tup);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern implementations
//===----------------------------------------------------------------------===//

bool AggregateConstantOpConversion::containsArrayAndStructAggregatesOnly(
    Type type) const {
  if (auto intType = type.dyn_cast<IntegerType>())
    return true;

  if (auto arrTy = type.dyn_cast<hw::ArrayType>())
    return containsArrayAndStructAggregatesOnly(arrTy.getElementType());

  if (auto structTy = type.dyn_cast<hw::StructType>()) {
    SmallVector<Type> innerTypes;
    structTy.getInnerTypes(innerTypes);
    return llvm::all_of(innerTypes, [&](auto ty) {
      return containsArrayAndStructAggregatesOnly(ty);
    });
  }

  return false;
}

bool AggregateConstantOpConversion::isMultiDimArrayOfIntegers(
    Type type, SmallVectorImpl<int64_t> &dims) const {
  if (auto intType = type.dyn_cast<IntegerType>())
    return true;

  if (auto arrTy = type.dyn_cast<hw::ArrayType>()) {
    dims.push_back(arrTy.getSize());
    return isMultiDimArrayOfIntegers(arrTy.getElementType(), dims);
  }

  return false;
}

void AggregateConstantOpConversion::flatten(
    Type type, Attribute attr, SmallVectorImpl<Attribute> &output) const {
  if (type.isa<IntegerType>()) {
    assert(attr.isa<IntegerAttr>());
    output.push_back(attr);
    return;
  }

  auto arrAttr = attr.cast<ArrayAttr>();
  for (size_t i = 0, e = arrAttr.size(); i < e; ++i) {
    auto element =
        arrAttr[HWToLLVMEndianessConverter::convertToLLVMEndianess(type, i)];

    flatten(type.cast<hw::ArrayType>().getElementType(), element, output);
  }
}

Value AggregateConstantOpConversion::constructAggregate(
    OpBuilder &builder, TypeConverter &typeConverter, Location loc, Type type,
    Attribute data) const {
  Type llvmType = typeConverter.convertType(type);

  auto getElementType = [](Type type, size_t index) {
    if (auto arrTy = type.dyn_cast<hw::ArrayType>()) {
      return arrTy.getElementType();
    }

    assert(type.isa<hw::StructType>());
    auto structTy = type.cast<hw::StructType>();
    SmallVector<Type> innerTypes;
    structTy.getInnerTypes(innerTypes);
    return innerTypes[index];
  };

  return TypeSwitch<Type, Value>(type)
      .Case<IntegerType>([&](auto ty) {
        return builder.create<LLVM::ConstantOp>(loc, data.cast<TypedAttr>());
      })
      .Case<hw::ArrayType, hw::StructType>([&](auto ty) {
        Value aggVal = builder.create<LLVM::UndefOp>(loc, llvmType);
        auto arrayAttr = data.cast<ArrayAttr>();
        for (size_t i = 0, e = arrayAttr.size(); i < e; ++i) {
          size_t currIdx =
              HWToLLVMEndianessConverter::convertToLLVMEndianess(type, i);
          Attribute input = arrayAttr[currIdx];
          Type elementType = getElementType(ty, currIdx);

          Value element = constructAggregate(builder, typeConverter, loc,
                                             elementType, input);
          aggVal = builder.create<LLVM::InsertValueOp>(loc, aggVal, element, i);
        }

        return aggVal;
      });
}

LogicalResult AggregateConstantOpConversion::matchAndRewrite(
    hw::AggregateConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Type aggregateType = op.getResult().getType();

  // TODO: Only arrays and structs supported at the moment.
  if (!containsArrayAndStructAggregatesOnly(aggregateType))
    return failure();

  auto llvmTy = typeConverter->convertType(op.getResult().getType());
  auto typeAttrPair = std::make_pair(aggregateType, adaptor.getFields());

  if (!constAggregateGlobalsMap.count(typeAttrPair) ||
      !constAggregateGlobalsMap[typeAttrPair]) {
    auto ipSave = rewriter.saveInsertionPoint();

    Operation *parent = op->getParentOp();
    while (!isa<mlir::ModuleOp>(parent->getParentOp())) {
      parent = parent->getParentOp();
    }

    rewriter.setInsertionPoint(parent);

    // Create a global region for this static array.
    auto name = globals.newName("_aggregate_const_global");

    SmallVector<int64_t> dims;
    if (isMultiDimArrayOfIntegers(aggregateType, dims)) {
      SmallVector<Attribute> ints;
      flatten(aggregateType, adaptor.getFields(), ints);
      assert(!ints.empty());
      auto shapedType = RankedTensorType::get(
          dims, ints.front().cast<IntegerAttr>().getType());
      auto denseAttr = DenseElementsAttr::get(shapedType, ints);

      constAggregateGlobalsMap[typeAttrPair] = rewriter.create<LLVM::GlobalOp>(
          op.getLoc(), llvmTy, true, LLVM::Linkage::Internal, name, denseAttr);
    } else {
      auto global = rewriter.create<LLVM::GlobalOp>(op.getLoc(), llvmTy, false,
                                                    LLVM::Linkage::Internal,
                                                    name, Attribute());
      Block *blk = new Block();
      global.getInitializerRegion().push_back(blk);
      rewriter.setInsertionPointToStart(blk);

      Value aggregate =
          constructAggregate(rewriter, *typeConverter, op.getLoc(),
                             aggregateType, adaptor.getFields());
      rewriter.create<LLVM::ReturnOp>(op.getLoc(), aggregate);
      constAggregateGlobalsMap[typeAttrPair] = global;
    }

    rewriter.restoreInsertionPoint(ipSave);
  }

  // Get the global array address and load it to return an array value.
  auto addr = rewriter.create<LLVM::AddressOfOp>(
      op->getLoc(), constAggregateGlobalsMap[typeAttrPair]);
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, llvmTy, addr);

  return success();
}

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

static Type convertArrayType(hw::ArrayType type, LLVMTypeConverter &converter) {
  auto elementTy = converter.convertType(type.getElementType());
  return LLVM::LLVMArrayType::get(elementTy, type.getSize());
}

static Type convertStructType(hw::StructType type,
                              LLVMTypeConverter &converter) {
  llvm::SmallVector<Type, 8> elements;
  mlir::SmallVector<mlir::Type> types;
  type.getInnerTypes(types);

  for (int i = 0, e = types.size(); i < e; ++i)
    elements.push_back(converter.convertType(
        types[HWToLLVMEndianessConverter::convertToLLVMEndianess(type, i)]));

  return LLVM::LLVMStructType::getLiteral(&converter.getContext(), elements);
}

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

namespace {
struct HWToLLVMLoweringPass : public ConvertHWToLLVMBase<HWToLLVMLoweringPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateHWToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    Namespace &globals,
    DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp>
        &constAggregateGlobalsMap) {
  MLIRContext *ctx = converter.getDialect()->getContext();

  // Value creation conversion patterns.
  patterns.add<HWConstantOpConversion>(ctx, converter);
  patterns.add<HWDynamicArrayCreateOpConversion, HWStructCreateOpConversion>(
      converter);
  patterns.add<AggregateConstantOpConversion>(
      converter, constAggregateGlobalsMap, globals);

  // Bitwise conversion patterns.
  patterns.add<BitcastOpConversion>(converter);

  // Extraction operation conversion patterns.
  patterns.add<ArrayGetOpConversion, ArraySliceOpConversion,
               ArrayConcatOpConversion, StructExplodeOpConversion,
               StructExtractOpConversion, StructInjectOpConversion>(converter);
}

void circt::populateHWToLLVMTypeConversions(LLVMTypeConverter &converter) {
  converter.addConversion(
      [&](hw::ArrayType arr) { return convertArrayType(arr, converter); });
  converter.addConversion(
      [&](hw::StructType tup) { return convertStructType(tup, converter); });
}

void HWToLLVMLoweringPass::runOnOperation() {
  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp> constAggregateGlobalsMap;
  Namespace globals;
  SymbolCache cache;
  cache.addDefinitions(getOperation());
  globals.add(cache);

  RewritePatternSet patterns(&getContext());
  auto converter = mlir::LLVMTypeConverter(&getContext());
  populateHWToLLVMTypeConversions(converter);

  LLVMConversionTarget target(getContext());
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<hw::HWDialect>();

  // Setup the conversion.
  populateHWToLLVMConversionPatterns(converter, patterns, globals,
                                     constAggregateGlobalsMap);

  // Apply the partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create an HW to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertHWToLLVMPass() {
  return std::make_unique<HWToLLVMLoweringPass>();
}
