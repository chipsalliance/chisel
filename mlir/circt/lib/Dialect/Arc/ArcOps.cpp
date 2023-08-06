//===- ArcOps.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace circt;
using namespace arc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static LogicalResult verifyTypeListEquivalence(Operation *op,
                                               TypeRange expectedTypeList,
                                               TypeRange actualTypeList,
                                               StringRef elementName) {
  if (expectedTypeList.size() != actualTypeList.size())
    return op->emitOpError("incorrect number of ")
           << elementName << "s: expected " << expectedTypeList.size()
           << ", but got " << actualTypeList.size();

  for (unsigned i = 0, e = expectedTypeList.size(); i != e; ++i) {
    if (expectedTypeList[i] != actualTypeList[i]) {
      auto diag = op->emitOpError(elementName)
                  << " type mismatch: " << elementName << " #" << i;
      diag.attachNote() << "expected type: " << expectedTypeList[i];
      diag.attachNote() << "  actual type: " << actualTypeList[i];
      return diag;
    }
  }

  return success();
}

static LogicalResult verifyArcSymbolUse(Operation *op, TypeRange inputs,
                                        TypeRange results,
                                        SymbolTableCollection &symbolTable) {
  // Check that the arc attribute was specified.
  auto arcName = op->getAttrOfType<FlatSymbolRefAttr>("arc");
  // The arc attribute is verified by the tablegen generated verifier as it is
  // an ODS defined attribute.
  assert(arcName && "FlatSymbolRefAttr called 'arc' missing");
  DefineOp arc = symbolTable.lookupNearestSymbolFrom<DefineOp>(op, arcName);
  if (!arc)
    return op->emitOpError() << "`" << arcName.getValue()
                             << "` does not reference a valid `arc.define`";

  // Verify that the operand and result types match the arc.
  auto type = arc.getFunctionType();
  if (failed(
          verifyTypeListEquivalence(op, type.getInputs(), inputs, "operand")))
    return failure();

  if (failed(
          verifyTypeListEquivalence(op, type.getResults(), results, "result")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// DefineOp
//===----------------------------------------------------------------------===//

ParseResult DefineOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void DefineOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, "function_type", getArgAttrsAttrName(),
      getResAttrsAttrName());
}

LogicalResult DefineOp::verifyRegions() {
  // Check that the body does not contain any side-effecting operations. We can
  // simply iterate over the ops directly within the body; operations with
  // regions, like scf::IfOp, implement the `HasRecursiveMemoryEffects` trait
  // which causes the `isMemoryEffectFree` check to already recur into their
  // regions.
  for (auto &op : getBodyBlock()) {
    if (isMemoryEffectFree(&op))
      continue;

    // We don't use a op-error here because that leads to the whole arc being
    // printed. This can be switched of when creating the context, but one
    // might not want to switch that off for other error messages. Here it's
    // definitely not desirable as arcs can be very big and would fill up the
    // error log, making it hard to read. Currently, only the signature (first
    // line) of the arc is printed.
    auto diag = mlir::emitError(getLoc(), "body contains non-pure operation");
    diag.attachNote(op.getLoc()).append("first non-pure operation here: ");
    return diag;
  }
  return success();
}

bool DefineOp::isPassthrough() {
  if (getNumArguments() != getNumResults())
    return false;

  return llvm::all_of(
      llvm::zip(getArguments(), getBodyBlock().getTerminator()->getOperands()),
      [](const auto &argAndRes) {
        return std::get<0>(argAndRes) == std::get<1>(argAndRes);
      });
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify() {
  auto *parent = (*this)->getParentOp();
  TypeRange expectedTypes = parent->getResultTypes();
  if (auto defOp = dyn_cast<DefineOp>(parent))
    expectedTypes = defOp.getResultTypes();

  TypeRange actualTypes = getOperands().getTypes();
  return verifyTypeListEquivalence(*this, expectedTypes, actualTypes, "output");
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

LogicalResult StateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyArcSymbolUse(*this, getInputs().getTypes(),
                            getResults().getTypes(), symbolTable);
}

LogicalResult StateOp::verify() {
  if (getLatency() > 0 && !getOperation()->getParentOfType<ClockDomainOp>() &&
      !getClock())
    return emitOpError(
        "with non-zero latency outside a clock domain requires a clock");

  if (getLatency() == 0) {
    if (getClock())
      return emitOpError("with zero latency cannot have a clock");
    if (getEnable())
      return emitOpError("with zero latency cannot have an enable");
    if (getReset())
      return emitOpError("with zero latency cannot have a reset");
  }

  if (getOperation()->getParentOfType<ClockDomainOp>() && getClock())
    return emitOpError("inside a clock domain cannot have a clock");

  return success();
}

bool StateOp::isClocked() { return getLatency() > 0; }

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyArcSymbolUse(*this, getInputs().getTypes(),
                            getResults().getTypes(), symbolTable);
}

//===----------------------------------------------------------------------===//
// MemoryWritePortOp
//===----------------------------------------------------------------------===//

SmallVector<Type> MemoryWritePortOp::getArcResultTypes() {
  auto memType = cast<MemoryType>(getMemory().getType());
  SmallVector<Type> resultTypes{memType.getAddressType(),
                                memType.getWordType()};
  if (getEnable())
    resultTypes.push_back(IntegerType::get(getContext(), 1));
  if (getMask())
    resultTypes.push_back(memType.getWordType());
  return resultTypes;
}

LogicalResult
MemoryWritePortOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyArcSymbolUse(*this, getInputs().getTypes(), getArcResultTypes(),
                            symbolTable);
}

LogicalResult MemoryWritePortOp::verify() {
  if (getLatency() < 1)
    return emitOpError("latency must be at least 1");

  if (!getOperation()->getParentOfType<ClockDomainOp>() && !getClock())
    return emitOpError("outside a clock domain requires a clock");

  if (getOperation()->getParentOfType<ClockDomainOp>() && getClock())
    return emitOpError("inside a clock domain cannot have a clock");

  return success();
}

//===----------------------------------------------------------------------===//
// ClockDomainOp
//===----------------------------------------------------------------------===//

LogicalResult ClockDomainOp::verifyRegions() {
  return verifyTypeListEquivalence(*this, getBodyBlock().getArgumentTypes(),
                                   getInputs().getTypes(), "input");
}

//===----------------------------------------------------------------------===//
// RootInputOp
//===----------------------------------------------------------------------===//

void RootInputOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buf("in_");
  buf += getName();
  setNameFn(getState(), buf);
}

//===----------------------------------------------------------------------===//
// RootOutputOp
//===----------------------------------------------------------------------===//

void RootOutputOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buf("out_");
  buf += getName();
  setNameFn(getState(), buf);
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

LogicalResult ModelOp::verify() {
  if (getBodyBlock().getArguments().size() != 1)
    return emitOpError("must have exactly one argument");
  if (auto type = getBodyBlock().getArgument(0).getType();
      !isa<StorageType>(type))
    return emitOpError("argument must be of storage type");
  return success();
}

//===----------------------------------------------------------------------===//
// LutOp
//===----------------------------------------------------------------------===//

LogicalResult LutOp::verify() {
  Location firstSideEffectOpLoc = UnknownLoc::get(getContext());
  const WalkResult result = getBody().walk([&](Operation *op) {
    if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
      memOp.getEffects(effects);

      if (!effects.empty()) {
        firstSideEffectOpLoc = memOp->getLoc();
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return emitOpError("no operations with side-effects allowed inside a LUT")
               .attachNote(firstSideEffectOpLoc)
           << "first operation with side-effects here";

  return success();
}

#include "circt/Dialect/Arc/ArcInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Arc/Arc.cpp.inc"
