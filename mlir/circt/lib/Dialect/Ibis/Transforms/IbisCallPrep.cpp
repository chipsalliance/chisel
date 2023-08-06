//===- IbisCallPrep.cpp - Implementation of call prep lowering ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ConversionPatterns.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace ibis;

/// Build indexes to make lookups faster. Create the new argument types as well.
struct CallPrepPrecomputed {
  CallPrepPrecomputed(ModuleOp mod);

  // Lookup a class from its symbol.
  DenseMap<StringAttr, ClassOp> classSymbols;

  // Mapping of method to argument type.
  DenseMap<SymbolRefAttr, std::pair<hw::StructType, Location>> argTypes;

  // Lookup the class to which a particular instance (in a particular class) is
  // referring.
  DenseMap<std::pair<ClassOp, StringAttr>, ClassOp> instanceMap;

  // Lookup an entry in instanceMap. If not found, return null.
  ClassOp lookupNext(ClassOp scope, StringAttr instSym) const {
    auto entry = instanceMap.find(std::make_pair(scope, instSym));
    if (entry == instanceMap.end())
      return {};
    return entry->second;
  }

  // Given an instance path, get the class::func symbolref for it.
  SymbolRefAttr resolveInstancePath(Operation *scope, SymbolRefAttr path) const;

  // Utility function to create a symbolref to a method.
  static SymbolRefAttr getSymbol(MethodOp method) {
    ClassOp cls = method.getParentOp();
    return SymbolRefAttr::get(cls.getSymNameAttr(),
                              {FlatSymbolRefAttr::get(method)});
  }
};

CallPrepPrecomputed::CallPrepPrecomputed(ModuleOp mod) {
  auto *ctxt = mod.getContext();

  // Populate the class-symbol lookup table.
  for (auto cls : mod.getOps<ClassOp>())
    classSymbols[cls.getSymNameAttr()] = cls;

  for (auto cls : mod.getOps<ClassOp>()) {
    // Compute new argument types for each method.
    for (auto method : cls.getOps<MethodOp>()) {

      // Create the struct type.
      SmallVector<hw::StructType::FieldInfo> argFields;
      for (auto [argName, argType] :
           llvm::zip(method.getArgNamesAttr().getAsRange<StringAttr>(),
                     method.getArgumentTypes()))
        argFields.push_back({argName, argType});
      auto argStruct = hw::StructType::get(ctxt, argFields);

      // Later we're gonna want the block locations, so compute a fused location
      // and store it.
      Location argLoc = UnknownLoc::get(ctxt);
      if (method->getNumRegions() > 0) {
        SmallVector<Location> argLocs;
        Block *body = &method.getBody().front();
        for (auto arg : body->getArguments())
          argLocs.push_back(arg.getLoc());
        argLoc = FusedLoc::get(ctxt, argLocs);
      }

      // Add both to the lookup table.
      argTypes.insert(
          std::make_pair(getSymbol(method), std::make_pair(argStruct, argLoc)));
    }

    // Populate the instances table.
    for (auto inst : cls.getOps<InstanceOp>()) {
      auto clsEntry = classSymbols.find(inst.getClassNameAttr().getAttr());
      assert(clsEntry != classSymbols.end() &&
             "class being instantiated doesn't exist");
      instanceMap[std::make_pair(cls, inst.getSymNameAttr())] =
          clsEntry->second;
    }
  }
}

SymbolRefAttr
CallPrepPrecomputed::resolveInstancePath(Operation *scope,
                                         SymbolRefAttr path) const {
  auto cls = scope->getParentOfType<ClassOp>();
  assert(cls && "scope outside of ibis class");

  // SymbolRefAttr is rather silly. The start of the path is root reference...
  cls = lookupNext(cls, path.getRootReference());
  if (!cls)
    return {};

  // ... then the rest are the nested references. The last one is the function
  // name rather than an instance.
  for (auto instSym : path.getNestedReferences().drop_back()) {
    cls = lookupNext(cls, instSym.getAttr());
    if (!cls)
      return {};
  }

  // The last one is the function symbol.
  return SymbolRefAttr::get(cls.getSymNameAttr(),
                            {FlatSymbolRefAttr::get(path.getLeafReference())});
}

namespace {
/// For each CallOp, the corresponding method signature will have changed. Pack
/// all the operands into a struct.
struct MergeCallArgs : public OpConversionPattern<CallOp> {
  MergeCallArgs(MLIRContext *ctxt, const CallPrepPrecomputed &info)
      : OpConversionPattern(ctxt), info(info) {}

  void rewrite(CallOp, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final;
  LogicalResult match(CallOp) const override { return success(); }

private:
  const CallPrepPrecomputed &info;
};
} // anonymous namespace

void MergeCallArgs::rewrite(CallOp call, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
  auto loc = call.getLoc();
  rewriter.setInsertionPoint(call);
  auto func = call->getParentOfType<mlir::FunctionOpInterface>();

  // Use the 'info' accelerator structures to find the argument type.
  SymbolRefAttr calleeSym =
      info.resolveInstancePath(func, adaptor.getCalleeAttr());
  auto argStructEntry = info.argTypes.find(calleeSym);
  assert(argStructEntry != info.argTypes.end() && "Method symref not found!");
  auto [argStruct, argLoc] = argStructEntry->second;

  // Pack all of the operands into it.
  auto newArg = rewriter.create<hw::StructCreateOp>(loc, argStruct,
                                                    adaptor.getOperands());
  newArg->setAttr("sv.namehint",
                  rewriter.getStringAttr(
                      call.getCalleeAttr().getLeafReference().getValue() +
                      "_args_called_from_" + func.getName()));

  // Update the call to use just the new struct.
  rewriter.updateRootInPlace(call, [&]() {
    call.getOperandsMutable().clear();
    call.getOperandsMutable().append(newArg.getResult());
  });
}

namespace {
/// Change the method signatures to only have one argument: a struct capturing
/// all of the original arguments.
struct MergeMethodArgs : public OpConversionPattern<MethodOp> {
  MergeMethodArgs(MLIRContext *ctxt, const CallPrepPrecomputed &info)
      : OpConversionPattern(ctxt), info(info) {}

  void rewrite(MethodOp, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final;
  LogicalResult match(MethodOp) const override { return success(); }

private:
  const CallPrepPrecomputed &info;
};
} // anonymous namespace

void MergeMethodArgs::rewrite(MethodOp func, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  auto loc = func.getLoc();
  auto *ctxt = getContext();

  // Find the pre-computed arg struct for this method.
  auto argStructEntry =
      info.argTypes.find(CallPrepPrecomputed::getSymbol(func));
  assert(argStructEntry != info.argTypes.end() && "Cannot find symref!");
  auto [argStruct, argLoc] = argStructEntry->second;

  // Create a new method with the new signature.
  FunctionType funcType = func.getFunctionType();
  FunctionType newFuncType =
      FunctionType::get(ctxt, {argStruct}, funcType.getResults());
  auto newArgNames = ArrayAttr::get(ctxt, {StringAttr::get(ctxt, "arg")});
  auto newMethod =
      rewriter.create<MethodOp>(loc, func.getSymNameAttr(), newFuncType,
                                newArgNames, ArrayAttr(), ArrayAttr());

  if (func->getNumRegions() > 0) {
    // Create a body block with a struct explode to the arg struct into the
    // original arguments.
    Block *b = rewriter.createBlock(&newMethod.getBodyRegion(), {}, {argStruct},
                                    {argLoc});
    rewriter.setInsertionPointToStart(b);
    auto replacementArgs =
        rewriter.create<hw::StructExplodeOp>(loc, b->getArgument(0));

    // Merge the original method body, rewiring the args.
    Block *funcBody = &func.getBody().front();
    rewriter.mergeBlocks(funcBody, b, replacementArgs.getResults());
  }

  rewriter.eraseOp(func);
}

namespace {
/// Run all the physical lowerings.
struct CallPrepPass : public IbisCallPrepBase<CallPrepPass> {
  void runOnOperation() override;

private:
  // Merge the arguments into one struct.
  LogicalResult merge(const CallPrepPrecomputed &);
};
} // anonymous namespace

void CallPrepPass::runOnOperation() {
  CallPrepPrecomputed info(getOperation());

  if (failed(merge(info))) {
    signalPassFailure();
    return;
  }
}

LogicalResult CallPrepPass::merge(const CallPrepPrecomputed &info) {
  // Set up a conversion and give it a set of laws.
  ConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addDynamicallyLegalOp<CallOp>([](CallOp call) {
    auto argValues = call.getArgOperands();
    return argValues.size() == 1 &&
           hw::type_isa<hw::StructType>(argValues.front().getType());
  });
  target.addDynamicallyLegalOp<MethodOp>([](MethodOp func) {
    ArrayRef<Type> argTypes = func.getFunctionType().getInputs();
    return argTypes.size() == 1 &&
           hw::type_isa<hw::StructType>(argTypes.front());
  });

  // Add patterns to merge the args on both the call and method sides.
  RewritePatternSet patterns(&getContext());
  patterns.insert<MergeCallArgs>(&getContext(), info);
  patterns.insert<MergeMethodArgs>(&getContext(), info);

  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

std::unique_ptr<Pass> circt::ibis::createCallPrepPass() {
  return std::make_unique<CallPrepPass>();
}
