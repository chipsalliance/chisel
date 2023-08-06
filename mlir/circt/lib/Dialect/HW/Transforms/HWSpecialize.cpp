//===- HWSpecialize.cpp - hw module specialization ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform performs specialization of parametric hw.module's.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

// Generates a module name by composing the name of 'moduleOp' and the set of
// provided 'parameters'.
static std::string generateModuleName(Namespace &ns, hw::HWModuleOp moduleOp,
                                      ArrayAttr parameters) {
  assert(parameters.size() != 0);
  std::string name = moduleOp.getName().str();
  for (auto param : parameters) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    int64_t paramValue = paramAttr.getValue().cast<IntegerAttr>().getInt();
    name += "_" + paramAttr.getName().str() + "_" + std::to_string(paramValue);
  }

  // Query the namespace to generate a unique name.
  return ns.newName(name).str();
}

// Returns true if any operand or result of 'op' is parametric.
static bool isParametricOp(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), isParametricType) ||
         llvm::any_of(op->getResultTypes(), isParametricType);
}

// Narrows 'value' using a comb.extract operation to the width of the
// hw.array-typed 'array'.
static FailureOr<Value> narrowValueToArrayWidth(OpBuilder &builder, Value array,
                                                Value value) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointAfterValue(value);
  auto arrayType = array.getType().cast<hw::ArrayType>();
  unsigned hiBit = llvm::Log2_64_Ceil(arrayType.getSize());

  return hiBit == 0 ? builder
                          .create<hw::ConstantOp>(value.getLoc(),
                                                  APInt(arrayType.getSize(), 0))
                          .getResult()
                    : builder
                          .create<comb::ExtractOp>(value.getLoc(), value,
                                                   /*lowBit=*/0, hiBit)
                          .getResult();
}

static hw::HWModuleOp targetModuleOp(hw::InstanceOp instanceOp,
                                     const SymbolCache &sc) {
  auto *targetOp = sc.getDefinition(instanceOp.getModuleNameAttr());
  auto targetHWModule = dyn_cast<hw::HWModuleOp>(targetOp);
  if (!targetHWModule)
    return {}; // Won't specialize external modules.

  if (targetHWModule.getParameters().size() == 0)
    return {}; // nothing to record or specialize

  return targetHWModule;
}

// Stores unique module parameters and references to them
struct ParameterSpecializationRegistry {
  llvm::MapVector<hw::HWModuleOp, llvm::SetVector<ArrayAttr>>
      uniqueModuleParameters;

  bool isRegistered(hw::HWModuleOp moduleOp, ArrayAttr parameters) const {
    auto it = uniqueModuleParameters.find(moduleOp);
    return it != uniqueModuleParameters.end() &&
           it->second.contains(parameters);
  }

  void registerModuleOp(hw::HWModuleOp moduleOp, ArrayAttr parameters) {
    uniqueModuleParameters[moduleOp].insert(parameters);
  }
};

struct EliminateParamValueOpPattern : public OpRewritePattern<ParamValueOp> {
  EliminateParamValueOpPattern(MLIRContext *context, ArrayAttr parameters)
      : OpRewritePattern<ParamValueOp>(context), parameters(parameters) {}

  LogicalResult matchAndRewrite(ParamValueOp op,
                                PatternRewriter &rewriter) const override {
    // Substitute the param value op with an evaluated constant operation.
    FailureOr<Attribute> evaluated =
        evaluateParametricAttr(op.getLoc(), parameters, op.getValue());
    if (failed(evaluated))
      return failure();
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(
        op, op.getType(),
        evaluated->cast<IntegerAttr>().getValue().getSExtValue());
    return success();
  }

  ArrayAttr parameters;
};

// hw.array_get operations require indexes to be of equal width of the
// array itself. Since indexes may originate from constants or parameters,
// emit comb.extract operations to fulfill this invariant.
struct NarrowArrayGetIndexPattern : public OpConversionPattern<ArrayGetOp> {
public:
  using OpConversionPattern<ArrayGetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = type_cast<ArrayType>(op.getInput().getType());
    Type targetIndexType = IntegerType::get(
        getContext(),
        inputType.getSize() == 1 ? 1 : llvm::Log2_64_Ceil(inputType.getSize()));

    if (op.getIndex().getType().getIntOrFloatBitWidth() ==
        targetIndexType.getIntOrFloatBitWidth())
      return failure(); // nothing to do

    // Narrow the index value.
    FailureOr<Value> narrowedIndex =
        narrowValueToArrayWidth(rewriter, op.getInput(), op.getIndex());
    if (failed(narrowedIndex))
      return failure();
    rewriter.replaceOpWithNewOp<ArrayGetOp>(op, op.getInput(), *narrowedIndex);
    return success();
  }
};

// Generic pattern to convert parametric result types.
struct ParametricTypeConversionPattern : public ConversionPattern {
  ParametricTypeConversionPattern(MLIRContext *ctx,
                                  TypeConverter &typeConverter,
                                  ArrayAttr parameters)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          ctx),
        parameters(parameters) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Value, 4> convertedOperands;
    // Update the result types of the operation
    bool ok = true;
    rewriter.updateRootInPlace(op, [&]() {
      // Mutate result types
      for (auto it : llvm::enumerate(op->getResultTypes())) {
        FailureOr<Type> res =
            evaluateParametricType(op->getLoc(), parameters, it.value());
        ok &= succeeded(res);
        if (!ok)
          return;
        op->getResult(it.index()).setType(*res);
      }

      // Note: 'operands' have already been converted with the supplied type
      // converter to this pattern. Make sure that we materialize this
      // conversion by updating the operands to op.
      op->setOperands(operands);
    });

    return success(ok);
  };
  ArrayAttr parameters;
};

struct HWSpecializePass : public hw::HWSpecializeBase<HWSpecializePass> {
  void runOnOperation() override;
};

static void populateTypeConversion(Location loc, TypeConverter &typeConverter,
                                   ArrayAttr parameters) {
  // Possibly parametric types
  typeConverter.addConversion([=](hw::IntType type) {
    return evaluateParametricType(loc, parameters, type);
  });
  typeConverter.addConversion([=](hw::ArrayType type) {
    return evaluateParametricType(loc, parameters, type);
  });

  // Valid target types.
  typeConverter.addConversion([](mlir::IntegerType type) { return type; });
}

// Registers any nested parametric instance ops of `target` for the next
// specialization loop
static LogicalResult registerNestedParametricInstanceOps(
    HWModuleOp target, ArrayAttr parameters, SymbolCache &sc,
    const ParameterSpecializationRegistry &currentRegistry,
    ParameterSpecializationRegistry &nextRegistry,
    llvm::DenseMap<hw::HWModuleOp,
                   llvm::DenseMap<ArrayAttr, llvm::SmallVector<hw::InstanceOp>>>
        &parametersUsers) {
  // Register any nested parametric instance ops for the next loop
  auto walkResult = target->walk([&](InstanceOp instanceOp) -> WalkResult {
    auto instanceParameters = instanceOp.getParameters();
    // We can ignore non-parametric instances
    if (instanceParameters.empty())
      return WalkResult::advance();

    // Replace instance parameters with evaluated versions
    llvm::SmallVector<Attribute> evaluatedInstanceParameters;
    evaluatedInstanceParameters.reserve(instanceParameters.size());
    for (auto instanceParameter : instanceParameters) {
      auto instanceParameterDecl = instanceParameter.cast<hw::ParamDeclAttr>();
      auto instanceParameterValue = instanceParameterDecl.getValue();
      auto evaluated = evaluateParametricAttr(target.getLoc(), parameters,
                                              instanceParameterValue);
      if (failed(evaluated))
        return WalkResult::interrupt();
      evaluatedInstanceParameters.push_back(
          hw::ParamDeclAttr::get(instanceParameterDecl.getName(), *evaluated));
    }

    auto evaluatedInstanceParametersAttr =
        ArrayAttr::get(target.getContext(), evaluatedInstanceParameters);

    if (auto targetHWModule = targetModuleOp(instanceOp, sc)) {
      if (!currentRegistry.isRegistered(targetHWModule,
                                        evaluatedInstanceParametersAttr))
        nextRegistry.registerModuleOp(targetHWModule,
                                      evaluatedInstanceParametersAttr);
      parametersUsers[targetHWModule][evaluatedInstanceParametersAttr]
          .push_back(instanceOp);
    }

    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

// Specializes the provided 'base' module into the 'target' module. By doing
// so, we create a new module which
// 1. has no parameters
// 2. has a name composing the name of 'base' as well as the 'parameters'
// parameters.
// 3. Has a top-level interface with any parametric types resolved.
// 4. Any references to module parameters have been replaced with the
// parameter value.
static LogicalResult specializeModule(
    OpBuilder builder, ArrayAttr parameters, SymbolCache &sc, Namespace &ns,
    HWModuleOp source, HWModuleOp &target,
    const ParameterSpecializationRegistry &currentRegistry,
    ParameterSpecializationRegistry &nextRegistry,
    llvm::DenseMap<hw::HWModuleOp,
                   llvm::DenseMap<ArrayAttr, llvm::SmallVector<hw::InstanceOp>>>
        &parametersUsers) {
  auto *ctx = builder.getContext();
  // Update the types of the source module ports based on evaluating any
  // parametric in/output ports.
  auto ports = source.getPortList();
  for (auto in : llvm::enumerate(source.getFunctionType().getInputs())) {
    FailureOr<Type> resType =
        evaluateParametricType(source.getLoc(), parameters, in.value());
    if (failed(resType))
      return failure();
    ports.atInput(in.index()).type = *resType;
  }
  for (auto out : llvm::enumerate(source.getFunctionType().getResults())) {
    FailureOr<Type> resolvedType =
        evaluateParametricType(source.getLoc(), parameters, out.value());
    if (failed(resolvedType))
      return failure();
    ports.atOutput(out.index()).type = *resolvedType;
  }

  // Create the specialized module using the evaluated port info.
  target = builder.create<HWModuleOp>(
      source.getLoc(),
      StringAttr::get(ctx, generateModuleName(ns, source, parameters)), ports);

  // Erase the default created hw.output op - we'll copy the correct operation
  // during body elaboration.
  (*target.getOps<hw::OutputOp>().begin()).erase();

  // Clone body of the source into the target. Use ValueMapper to ensure safe
  // cloning in the presence of backedges.
  BackedgeBuilder bb(builder, source.getLoc());
  ValueMapper mapper(&bb);
  for (auto &&[src, dst] :
       llvm::zip(source.getArguments(), target.getArguments()))
    mapper.set(src, dst);
  builder.setInsertionPointToStart(target.getBodyBlock());

  for (auto &op : source.getOps()) {
    IRMapping bvMapper;
    for (auto operand : op.getOperands())
      bvMapper.map(operand, mapper.get(operand));
    auto *newOp = builder.clone(op, bvMapper);
    for (auto &&[oldRes, newRes] :
         llvm::zip(op.getResults(), newOp->getResults()))
      mapper.set(oldRes, newRes);
  }

  // Register any nested parametric instance ops for the next loop
  auto nestedRegistrationResult = registerNestedParametricInstanceOps(
      target, parameters, sc, currentRegistry, nextRegistry, parametersUsers);
  if (failed(nestedRegistrationResult))
    return failure();

  // We've now created a separate copy of the source module with a rewritten
  // top-level interface. Next, we enter the module to convert parametric
  // types within operations.
  RewritePatternSet patterns(ctx);
  TypeConverter t;
  populateTypeConversion(target.getLoc(), t, parameters);
  patterns.add<EliminateParamValueOpPattern>(ctx, parameters);
  patterns.add<NarrowArrayGetIndexPattern>(ctx);
  patterns.add<ParametricTypeConversionPattern>(ctx, t, parameters);
  ConversionTarget convTarget(*ctx);
  convTarget.addLegalOp<hw::HWModuleOp>();
  convTarget.addIllegalOp<hw::ParamValueOp>();

  // Generic legalization of converted operations.
  convTarget.markUnknownOpDynamicallyLegal(
      [](Operation *op) { return !isParametricOp(op); });

  return applyPartialConversion(target, convTarget, std::move(patterns));
}

void HWSpecializePass::runOnOperation() {
  ModuleOp module = getOperation();

  // Record unique module parameters and references to these.
  llvm::DenseMap<hw::HWModuleOp,
                 llvm::DenseMap<ArrayAttr, llvm::SmallVector<hw::InstanceOp>>>
      parametersUsers;
  ParameterSpecializationRegistry registry;

  // Maintain a symbol cache for fast lookup during module specialization.
  SymbolCache sc;
  sc.addDefinitions(module);
  Namespace ns;
  ns.add(sc);

  for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
    // If this module is parametric, defer registering its parametric
    // instantiations until this module is specialized
    if (!hwModule.getParameters().empty())
      continue;
    for (auto instanceOp : hwModule.getOps<hw::InstanceOp>()) {
      if (auto targetHWModule = targetModuleOp(instanceOp, sc)) {
        auto parameters = instanceOp.getParameters();
        registry.registerModuleOp(targetHWModule, parameters);

        parametersUsers[targetHWModule][parameters].push_back(instanceOp);
      }
    }
  }

  // Create specialized modules.
  OpBuilder builder = OpBuilder(&getContext());
  builder.setInsertionPointToStart(module.getBody());
  llvm::DenseMap<hw::HWModuleOp, llvm::DenseMap<ArrayAttr, hw::HWModuleOp>>
      specializations;

  // For every module specialization, any nested parametric modules will be
  // registered for the next loop. We loop until no new nested modules have been
  // registered.
  while (!registry.uniqueModuleParameters.empty()) {
    // The registry for the next specialization loop
    ParameterSpecializationRegistry nextRegistry;
    for (auto it : registry.uniqueModuleParameters) {
      for (auto parameters : it.second) {
        HWModuleOp specializedModule;
        if (failed(specializeModule(builder, parameters, sc, ns, it.first,
                                    specializedModule, registry, nextRegistry,
                                    parametersUsers))) {
          signalPassFailure();
          return;
        }

        // Extend the symbol cache with the newly created module.
        sc.addDefinition(specializedModule.getNameAttr(), specializedModule);

        // Add the specialization
        specializations[it.first][parameters] = specializedModule;
      }
    }

    // Transfer newly registered specializations to iterate over
    registry.uniqueModuleParameters =
        std::move(nextRegistry.uniqueModuleParameters);
  }

  // Rewrite instances of specialized modules to the specialized module.
  for (auto it : specializations) {
    auto unspecialized = it.getFirst();
    auto &users = parametersUsers[unspecialized];
    for (auto specialization : it.getSecond()) {
      auto parameters = specialization.getFirst();
      auto specializedModule = specialization.getSecond();
      for (auto instanceOp : users[parameters]) {
        instanceOp->setAttr("moduleName",
                            FlatSymbolRefAttr::get(specializedModule));
        instanceOp->setAttr("parameters", ArrayAttr::get(&getContext(), {}));
      }
    }
  }
}

} // namespace

std::unique_ptr<Pass> circt::hw::createHWSpecializePass() {
  return std::make_unique<HWSpecializePass>();
}
