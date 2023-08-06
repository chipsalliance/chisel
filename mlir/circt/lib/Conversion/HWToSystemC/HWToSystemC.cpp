//===- HWToSystemC.cpp - HW To SystemC Conversion Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to SystemC Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToSystemC.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace systemc;

//===----------------------------------------------------------------------===//
// Operation Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// This works on each HW module, creates corresponding SystemC modules, moves
/// the body of the module into the new SystemC module by splitting up the body
/// into field declarations, initializations done in a newly added systemc.ctor,
/// and internal methods to be registered in the constructor.
struct ConvertHWModule : public OpConversionPattern<HWModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HWModuleOp module, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Parameterized modules are supported yet.
    if (!module.getParameters().empty())
      return emitError(module->getLoc(), "module parameters not supported yet");

    auto ports = module.getPortList();
    if (llvm::any_of(ports, [](auto &port) { return port.isInOut(); }))
      return emitError(module->getLoc(), "inout arguments not supported yet");

    // Create the SystemC module.
    for (size_t i = 0; i < ports.size(); ++i)
      ports.at(i).type = typeConverter->convertType(ports.at(i).type);

    auto scModule = rewriter.create<SCModuleOp>(module.getLoc(),
                                                module.getNameAttr(), ports);
    auto *outputOp = module.getBodyBlock()->getTerminator();
    scModule.setVisibility(module.getVisibility());

    SmallVector<Attribute> portAttrs;
    if (auto argAttrs = module.getAllArgAttrs())
      portAttrs.append(argAttrs.begin(), argAttrs.end());
    else
      portAttrs.append(module.getNumInputs(), Attribute());
    if (auto resultAttrs = module.getAllResultAttrs())
      portAttrs.append(resultAttrs.begin(), resultAttrs.end());
    else
      portAttrs.append(module.getNumOutputs(), Attribute());

    scModule.setAllArgAttrs(portAttrs);

    // Create a systemc.func operation inside the module after the ctor.
    // TODO: implement logic to extract a better name and properly unique it.
    rewriter.setInsertionPointToStart(scModule.getBodyBlock());
    auto scFunc = rewriter.create<SCFuncOp>(
        module.getLoc(), rewriter.getStringAttr("innerLogic"));

    // Inline the HW module body into the systemc.func body.
    // TODO: do some dominance analysis to detect use-before-def and cycles in
    // the use chain, which are allowed in graph regions but not in SSACFG
    // regions, and when possible fix them.
    scFunc.getBodyBlock()->erase();
    Region &scFuncBody = scFunc.getBody();
    rewriter.inlineRegionBefore(module.getBody(), scFuncBody, scFuncBody.end());

    // Register the systemc.func inside the systemc.ctor
    rewriter.setInsertionPointToStart(
        scModule.getOrCreateCtor().getBodyBlock());
    rewriter.create<MethodOp>(scModule.getLoc(), scFunc.getHandle());

    // Register the sensitivities of above SC_METHOD registration.
    SmallVector<Value> sensitivityValues(
        llvm::make_filter_range(scModule.getArguments(), [](BlockArgument arg) {
          return !arg.getType().isa<OutputType>();
        }));
    if (!sensitivityValues.empty())
      rewriter.create<SensitiveOp>(scModule.getLoc(), sensitivityValues);

    // Move the block arguments of the systemc.func (that we got from the
    // hw.module) to the systemc.module
    rewriter.setInsertionPointToStart(scFunc.getBodyBlock());
    auto portsLocal = module.getPortList();
    for (size_t i = 0, e = scFunc.getRegion().getNumArguments(); i < e; ++i) {
      auto inputRead =
          rewriter
              .create<SignalReadOp>(scFunc.getLoc(), scModule.getArgument(i))
              .getResult();
      auto converted = typeConverter->materializeSourceConversion(
          rewriter, scModule.getLoc(), portsLocal.at(i).type, inputRead);
      scFuncBody.getArgument(0).replaceAllUsesWith(converted);
      scFuncBody.eraseArgument(0);
    }

    // Erase the HW module.
    rewriter.eraseOp(module);

    SmallVector<Value> outPorts;
    for (auto val : scModule.getArguments()) {
      if (val.getType().isa<OutputType>())
        outPorts.push_back(val);
    }

    rewriter.setInsertionPoint(outputOp);
    for (auto args : llvm::zip(outPorts, outputOp->getOperands())) {
      Value portValue = std::get<0>(args);
      auto converted = typeConverter->materializeTargetConversion(
          rewriter, scModule.getLoc(), getSignalBaseType(portValue.getType()),
          std::get<1>(args));
      rewriter.create<SignalWriteOp>(outputOp->getLoc(), portValue, converted);
    }

    // Erase the HW OutputOp.
    outputOp->dropAllReferences();
    rewriter.eraseOp(outputOp);

    return success();
  }
};

/// Convert hw.instance operations to systemc.instance.decl and a
/// systemc.instance.bind_port operation for each port in the constructor. Also
/// insert the necessary intermediate signals and write or read their state in
/// the update function accordingly.
class ConvertInstance : public OpConversionPattern<InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

private:
  template <typename PortTy>
  LogicalResult
  collectPortInfo(ValueRange ports, ArrayAttr portNames,
                  SmallVector<systemc::ModuleType::PortInfo> &portInfo) const {
    for (auto inPort : llvm::zip(ports, portNames)) {
      Type ty = std::get<0>(inPort).getType();
      systemc::ModuleType::PortInfo info;

      if (ty.isa<hw::InOutType>())
        return failure();

      info.type = typeConverter->convertType(PortTy::get(ty));
      info.name = std::get<1>(inPort).cast<StringAttr>();
      portInfo.push_back(info);
    }

    return success();
  }

public:
  LogicalResult
  matchAndRewrite(InstanceOp instanceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Make sure the parent is already converted such that we already have a
    // constructor and update function to insert operations into.
    auto scModule = instanceOp->getParentOfType<SCModuleOp>();
    if (!scModule)
      return rewriter.notifyMatchFailure(instanceOp,
                                         "parent was not an SCModuleOp");

    // Get the builders for the different places to insert operations.
    auto ctor = scModule.getOrCreateCtor();
    OpBuilder stateBuilder(ctor);
    OpBuilder initBuilder = OpBuilder::atBlockEnd(ctor.getBodyBlock());

    // Collect the port types and names of the instantiated module and convert
    // them to appropriate systemc types.
    SmallVector<systemc::ModuleType::PortInfo> portInfo;
    if (failed(collectPortInfo<InputType>(adaptor.getInputs(),
                                          adaptor.getArgNames(), portInfo)) ||
        failed(collectPortInfo<OutputType>(instanceOp->getResults(),
                                           adaptor.getResultNames(), portInfo)))
      return instanceOp->emitOpError("inout ports not supported");

    Location loc = instanceOp->getLoc();
    auto instanceName = instanceOp.getInstanceNameAttr();
    auto instModuleName = instanceOp.getModuleNameAttr();

    // Declare the instance.
    auto instDecl = stateBuilder.create<InstanceDeclOp>(
        loc, instanceName, instModuleName, portInfo);

    // Bind the input ports.
    for (size_t i = 0, numInputs = adaptor.getInputs().size(); i < numInputs;
         ++i) {
      Value input = adaptor.getInputs()[i];
      auto portId = rewriter.getIndexAttr(i);
      StringAttr signalName = rewriter.getStringAttr(
          instanceName.getValue() + "_" + portInfo[i].name.getValue());

      if (auto readOp = input.getDefiningOp<SignalReadOp>()) {
        // Use the read channel directly without adding an
        // intermediate signal.
        initBuilder.create<BindPortOp>(loc, instDecl, portId,
                                       readOp.getInput());
        continue;
      }

      // Otherwise, create an intermediate signal to bind the instance port to.
      Type sigType = SignalType::get(getSignalBaseType(portInfo[i].type));
      Value channel = stateBuilder.create<SignalOp>(loc, sigType, signalName);
      initBuilder.create<BindPortOp>(loc, instDecl, portId, channel);
      rewriter.create<SignalWriteOp>(loc, channel, input);
    }

    // Bind the output ports.
    for (size_t i = 0, numOutputs = instanceOp->getNumResults(); i < numOutputs;
         ++i) {
      size_t numInputs = adaptor.getInputs().size();
      Value output = instanceOp->getResult(i);
      auto portId = rewriter.getIndexAttr(i + numInputs);
      StringAttr signalName =
          rewriter.getStringAttr(instanceName.getValue() + "_" +
                                 portInfo[i + numInputs].name.getValue());

      if (output.hasOneUse()) {
        if (auto writeOp = dyn_cast<SignalWriteOp>(*output.user_begin())) {
          // Use the channel written to directly. When there are multiple
          // channels this value is written to or it is used somewhere else, we
          // cannot shortcut it and have to insert an intermediate value because
          // we cannot insert multiple bind statements for one submodule port.
          // It is also necessary to bind it to an intermediate signal when it
          // has no uses as every port has to be bound to a channel.
          initBuilder.create<BindPortOp>(loc, instDecl, portId,
                                         writeOp.getDest());
          writeOp->erase();
          continue;
        }
      }

      // Otherwise, create an intermediate signal.
      Type sigType =
          SignalType::get(getSignalBaseType(portInfo[i + numInputs].type));
      Value channel = stateBuilder.create<SignalOp>(loc, sigType, signalName);
      initBuilder.create<BindPortOp>(loc, instDecl, portId, channel);
      auto instOut = rewriter.create<SignalReadOp>(loc, channel);
      output.replaceAllUsesWith(instOut);
    }

    rewriter.eraseOp(instanceOp);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static void populateLegality(ConversionTarget &target) {
  target.addIllegalDialect<HWDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<systemc::SystemCDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<emitc::EmitCDialect>();
  target.addLegalOp<hw::ConstantOp>();
}

static void populateOpConversion(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  patterns.add<ConvertHWModule, ConvertInstance>(typeConverter,
                                                 patterns.getContext());
}

static void populateTypeConversion(TypeConverter &converter) {
  converter.addConversion([](Type type) { return type; });
  converter.addConversion([&](SignalType type) {
    return SignalType::get(converter.convertType(type.getBaseType()));
  });
  converter.addConversion([&](InputType type) {
    return InputType::get(converter.convertType(type.getBaseType()));
  });
  converter.addConversion([&](systemc::InOutType type) {
    return systemc::InOutType::get(converter.convertType(type.getBaseType()));
  });
  converter.addConversion([&](OutputType type) {
    return OutputType::get(converter.convertType(type.getBaseType()));
  });
  converter.addConversion([](IntegerType type) -> Type {
    auto bw = type.getIntOrFloatBitWidth();
    if (bw == 1)
      return type;

    if (bw <= 64) {
      if (type.isSigned())
        return systemc::IntType::get(type.getContext(), bw);

      return UIntType::get(type.getContext(), bw);
    }

    if (bw <= 512) {
      if (type.isSigned())
        return BigIntType::get(type.getContext(), bw);

      return BigUIntType::get(type.getContext(), bw);
    }

    return BitVectorType::get(type.getContext(), bw);
  });

  converter.addSourceMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        auto op = builder.create<ConvertOp>(loc, type, values[0]);
        return op.getResult();
      });

  converter.addTargetMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        auto op = builder.create<ConvertOp>(loc, type, values[0]);
        return op.getResult();
      });
}

//===----------------------------------------------------------------------===//
// HW to SystemC Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct HWToSystemCPass : public ConvertHWToSystemCBase<HWToSystemCPass> {
  void runOnOperation() override;
};
} // namespace

/// Create a HW to SystemC dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertHWToSystemCPass() {
  return std::make_unique<HWToSystemCPass>();
}

/// This is the main entrypoint for the HW to SystemC conversion pass.
void HWToSystemCPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  // Create the include operation here to have exactly one 'systemc' include at
  // the top instead of one per module.
  OpBuilder builder(module.getRegion());
  builder.create<emitc::IncludeOp>(module->getLoc(), "systemc.h", true);

  ConversionTarget target(context);
  TypeConverter typeConverter;
  RewritePatternSet patterns(&context);
  populateLegality(target);
  populateTypeConversion(typeConverter);
  populateOpConversion(patterns, typeConverter);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
