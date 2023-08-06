//===- HWToLLHD.cpp - HW to LLHD Conversion Pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to LLHD Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToLLHD.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace llhd;
using namespace hw;
using namespace comb;

//===----------------------------------------------------------------------===//
// Convert structure operations
//===----------------------------------------------------------------------===//

namespace {

/// This works on each HW module, creates corresponding entities, moves the
/// bodies of the modules into the entities, and converts the bodies.
struct ConvertHWModule : public OpConversionPattern<HWModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HWModuleOp module, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Collect the HW module's port types.
    FunctionType moduleType = module.getFunctionType();
    unsigned numInputs = moduleType.getNumInputs();
    TypeRange moduleInputs = moduleType.getInputs();
    TypeRange moduleOutputs = moduleType.getResults();

    // LLHD entities port types are all expressed as block arguments to the op,
    // so collect all of the types in the expected order (inputs then outputs).
    SmallVector<Type, 4> entityTypes(moduleInputs);
    entityTypes.append(moduleOutputs.begin(), moduleOutputs.end());

    // Ensure the input and output types have all been converted already. This
    // is handled separately by the upstream FunctionLikeTypeConversionPattern.
    if (!llvm::all_of(entityTypes,
                      [](Type type) { return type.isa<SigType>(); }))
      return rewriter.notifyMatchFailure(module, "Not all ports had SigType");

    // Create the entity. Note that LLHD does not support parameterized
    // entities, so this conversion does not support parameterized modules.
    auto entityType = rewriter.getFunctionType(entityTypes, {});
    auto entity = rewriter.create<EntityOp>(module.getLoc(), entityType,
                                            numInputs, /*argAttrs=*/ArrayAttr(),
                                            /*resAttrs=*/ArrayAttr());

    // Inline the HW module body into the entity body.
    Region &entityBodyRegion = entity.getBodyRegion();
    rewriter.inlineRegionBefore(module.getBodyRegion(), entityBodyRegion,
                                entityBodyRegion.end());

    // Set the entity name attributes. Add block arguments for each output,
    // since LLHD entity outputs are still block arguments to the op.
    rewriter.updateRootInPlace(entity, [&] {
      entity.setName(module.getName());
      entityBodyRegion.addArguments(
          moduleOutputs, SmallVector<Location, 4>(moduleOutputs.size(),
                                                  rewriter.getUnknownLoc()));
    });

    // Erase the HW module.
    rewriter.eraseOp(module);

    return success();
  }
};

/// This works on each output op, creating ops to drive the appropriate results.
struct ConvertOutput : public OpConversionPattern<hw::OutputOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::OutputOp output, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the number of inputs in the entity to offset into the block args.
    auto entity = output->getParentOfType<EntityOp>();
    if (!entity)
      return rewriter.notifyMatchFailure(output, "parent was not an EntityOp");
    size_t numInputs = entity.getIns();

    // Drive the results from the mapped operands.
    Value delta;
    for (size_t i = 0, e = adaptor.getOperands().size(); i != e; ++i) {
      // Get the source and destination signals.
      auto src = adaptor.getOperands()[i];
      auto dest = entity.getArgument(numInputs + i);
      if (!src || !dest)
        return rewriter.notifyMatchFailure(
            output, "output operand must map to result block arg");

      // Look through probes on the source side and use the signal directly.
      if (auto prb = src.getDefiningOp<PrbOp>())
        src = prb.getSignal();

      // No work needed if they already are the same.
      if (src == dest)
        continue;

      // If the source has a signal type, connect it.
      if (auto sigTy = src.getType().dyn_cast<SigType>()) {
        rewriter.create<llhd::ConnectOp>(output.getLoc(), dest, src);
        continue;
      }

      // Otherwise, drive the destination block argument value.
      if (!delta) {
        delta = rewriter.create<ConstantTimeOp>(output.getLoc(), 0, "ns", 1, 0);
      }
      rewriter.create<DrvOp>(output.getLoc(), dest, src, delta, Value());
    }

    // Remove the output op.
    rewriter.eraseOp(output);

    return success();
  }
};

/// This works on each instance op, converting them to the LLHD dialect. If the
/// HW instance ops were defined in terms of the CallableOpInterface, we could
/// generalize this in terms of the upstream pattern to rewrite call ops' types.
struct ConvertInstance : public OpConversionPattern<InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp instance, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value delta;

    // Materialize signals for instance arguments that are of non-signal type.
    SmallVector<Value, 4> arguments;
    unsigned argIdx = 0;
    for (auto arg : adaptor.getOperands()) {
      // Connect signals directly.
      auto argType = arg.getType();
      if (argType.isa<SigType>()) {
        arguments.push_back(arg);
        continue;
      }

      // Look through probes and use the signal directly.
      if (auto prb = arg.getDefiningOp<PrbOp>()) {
        arguments.push_back(prb.getSignal());
        continue;
      }

      // Otherwise materialize a signal.
      // TODO: This should be a `llhd.buffer` operation. Ultimately we would
      // want this to be done by the TypeConverter in a materialization
      // callback. That requires adding the mentioned operation and fleshing out
      // the semantics of a zero-delay drive in the simulation. Once this is
      // done, the materializer can insert buffers with no delay and have them
      // collected in a canonicalization later where appropriate.
      // See github.com/llvm/circt/pull/988 for a discussion.
      if (!argType.isa<IntegerType>())
        return rewriter.notifyMatchFailure(instance, [&](Diagnostic &diag) {
          diag << "argument type " << argType << " is not supported";
        });

      auto init = rewriter.create<ConstantOp>(arg.getLoc(), argType, 0);
      SmallString<8> sigName(instance.getInstanceName());
      sigName += "_arg_";
      sigName += std::to_string(argIdx++);
      auto sig = rewriter.createOrFold<SigOp>(
          arg.getLoc(), SigType::get(argType), sigName, init);
      if (!delta) {
        delta = rewriter.create<ConstantTimeOp>(arg.getLoc(), 0, "ns", 1, 0);
      }
      rewriter.create<DrvOp>(arg.getLoc(), sig, arg, delta, Value());
      arguments.push_back(sig);
    }

    // HW instances model output ports as SSA results produced by the op. LLHD
    // instances model output ports as arguments to the op, so we need to find
    // or create SSA values. For each output port in the HW instance, try to
    // find a signal that can be used directly, or else create a new signal.
    SmallVector<Value, 4> resultSigs;
    SmallVector<Value, 4> resultValues;
    for (auto result : instance.getResults()) {
      auto resultType = result.getType();
      if (!resultType.isa<IntegerType>())
        return rewriter.notifyMatchFailure(instance, [&](Diagnostic &diag) {
          diag << "result type " << resultType << " is not supported";
        });

      Location loc = result.getLoc();

      // Since we need to have a signal for this result, see if an OutputOp maps
      // it to an output signal of our parent module. In that case we can just
      // use that signal.
      Value sig;
      for (auto &use : result.getUses()) {
        if (isa<hw::OutputOp>(use.getOwner())) {
          auto entity = instance->getParentOfType<EntityOp>();
          if (!entity)
            continue;
          sig = entity.getArgument(entity.getIns() + use.getOperandNumber());
          break;
        }
      }

      // Otherwise materialize a signal.
      // TODO: This should be a `llhd.buffer` operation. Ultimately we would
      // want this to be done by the TypeConverter in a materialization
      // callback. That requires adding the mentioned operation and fleshing out
      // the semantics of a zero-delay drive in the simulation. Once this is
      // done, the materializer can insert buffers with no delay and have them
      // collected in a canonicalization later where appropriate.
      // See github.com/llvm/circt/pull/988 for a discussion.
      if (!sig) {
        auto init = rewriter.create<ConstantOp>(loc, resultType, 0);
        SmallString<8> sigName(instance.getInstanceName());
        sigName += "_result_";
        sigName += std::to_string(result.getResultNumber());
        sig = rewriter.createOrFold<SigOp>(loc, SigType::get(resultType),
                                           sigName, init);
      }

      // Make OutputOps directly refer to this signal, which allows them to use
      // a ConnectOp rather than a PrbOp+DrvOp combo.
      for (auto &use : llvm::make_early_inc_range(result.getUses())) {
        if (isa<hw::OutputOp>(use.getOwner())) {
          rewriter.updateRootInPlace(use.getOwner(), [&]() { use.set(sig); });
        }
      }

      // Probe the value of the signal such that we end up having a replacement
      // for the InstanceOp results later on.
      auto prb = rewriter.create<PrbOp>(loc, resultType, sig);
      resultSigs.push_back(sig);
      resultValues.push_back(prb);
    }

    // Create the LLHD instance from the operands and results. Then mark the
    // original instance for replacement with the new values probed from the
    // signals attached to the LLHD instance.
    rewriter.create<InstOp>(instance.getLoc(), instance.getInstanceName(),
                            instance.getModuleName(), arguments, resultSigs);
    rewriter.replaceOp(instance, resultValues);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// HW to LLHD Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct HWToLLHDPass : public ConvertHWToLLHDBase<HWToLLHDPass> {
  void runOnOperation() override;
};

/// A helper type converter class that automatically populates the relevant
/// materializations and type conversions for converting HW to LLHD.
struct HWToLLHDTypeConverter : public TypeConverter {
  HWToLLHDTypeConverter();
};
} // namespace

/// Create a HW to LLHD conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertHWToLLHDPass() {
  return std::make_unique<HWToLLHDPass>();
}

/// This is the main entrypoint for the HW to LLHD conversion pass.
void HWToLLHDPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  // Mark the HW structure ops as illegal such that they get rewritten.
  ConversionTarget target(context);
  target.addLegalDialect<LLHDDialect>();
  target.addLegalDialect<CombDialect>();
  target.addLegalOp<ConstantOp>();
  target.addIllegalOp<HWModuleOp>();
  target.addIllegalOp<hw::OutputOp>();
  target.addIllegalOp<InstanceOp>();

  // Rewrite `hw.module`, `hw.output`, and `hw.instance`.
  HWToLLHDTypeConverter typeConverter;
  RewritePatternSet patterns(&context);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<HWModuleOp>(
      patterns, typeConverter);
  patterns.add<ConvertHWModule>(&context);
  patterns.add<ConvertInstance>(&context);
  patterns.add<ConvertOutput>(&context);
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// TypeConverter conversions and materializations
//===----------------------------------------------------------------------===//

HWToLLHDTypeConverter::HWToLLHDTypeConverter() {
  // Convert any type by just wrapping it in `SigType`.
  addConversion([](Type type) { return SigType::get(type); });

  // Mark `SigType` legal by converting it to itself.
  addConversion([](SigType type) { return type; });

  // Materialze probes when arguments are converted from any type to `SigType`.
  addSourceMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        auto op = builder.create<PrbOp>(loc, type, values[0]);
        return op.getResult();
      });
}
