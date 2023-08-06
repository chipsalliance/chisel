//===- InteropOps.cpp - Implement the Interop operations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Interop operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Interop/InteropOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace circt;
using namespace circt::interop;

//===----------------------------------------------------------------------===//
// ProceduralInitOp
//===----------------------------------------------------------------------===//

void ProceduralInitOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             ValueRange states,
                             InteropMechanism interopMechanism) {
  Region *region = odsState.addRegion();
  region->push_back(new Block);
  odsState.addAttribute(
      getInteropMechanismAttrName(odsState.name),
      InteropMechanismAttr::get(odsBuilder.getContext(), interopMechanism));
  odsState.addOperands(states);
}

LogicalResult ProceduralInitOp::verify() {
  if (getBody()->getNumArguments() > 0)
    return emitOpError("region must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// ProceduralUpdateOp
//===----------------------------------------------------------------------===//

void ProceduralUpdateOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               TypeRange outputs, ValueRange inputs,
                               ValueRange states,
                               InteropMechanism interopMechanism) {
  Region *region = odsState.addRegion();
  Block *bodyBlock = new Block;
  bodyBlock->addArguments(
      states.getTypes(),
      SmallVector<Location>(states.size(), odsState.location));
  bodyBlock->addArguments(
      inputs.getTypes(),
      SmallVector<Location>(inputs.size(), odsState.location));
  region->push_back(bodyBlock);
  odsState.addOperands(states);
  odsState.addOperands(inputs);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name),
                        odsBuilder.getDenseI32ArrayAttr(
                            {(int32_t)states.size(), (int32_t)inputs.size()}));
  odsState.addAttribute(
      getInteropMechanismAttrName(odsState.name),
      InteropMechanismAttr::get(odsBuilder.getContext(), interopMechanism));
  odsState.addTypes(outputs);
}

LogicalResult ProceduralUpdateOp::verify() {
  if (getBody()->getNumArguments() != getStates().size() + getInputs().size())
    return emitOpError("region must have the same number of arguments ")
           << "as inputs and states together, but got "
           << getBody()->getNumArguments() << " arguments and "
           << (getStates().size() + getInputs().size())
           << " state plus input types";

  SmallVector<Type> types{getStates().getTypes()};
  types.append(SmallVector<Type>{getInputs().getTypes()});
  if (getBody()->getArgumentTypes() != types)
    return emitOpError("region argument types must match state types");

  return success();
}

//===----------------------------------------------------------------------===//
// ProceduralDeallocOp
//===----------------------------------------------------------------------===//

void ProceduralDeallocOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                ValueRange states,
                                InteropMechanism interopMechanism) {
  Region *region = odsState.addRegion();
  Block *bodyBlock = new Block;
  region->push_back(bodyBlock);
  bodyBlock->addArguments(
      states.getTypes(),
      SmallVector<Location>(states.size(), odsState.location));
  odsState.addAttribute(
      getInteropMechanismAttrName(odsState.name),
      InteropMechanismAttr::get(odsBuilder.getContext(), interopMechanism));
  odsState.addOperands(states);
}

LogicalResult ProceduralDeallocOp::verify() {
  if (getBody()->getNumArguments() != getStates().size())
    return emitOpError("region must have the same number of arguments ")
           << "as states, but got " << getBody()->getNumArguments()
           << " arguments and " << getStates().size() << " states";

  if (getBody()->getArgumentTypes() != getStates().getTypes())
    return emitOpError("region argument types must match state types");

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto *parent = getOperation()->getParentOp();
  ValueRange values;

  if (isa<ProceduralUpdateOp>(parent))
    values = parent->getResults();
  else
    values = parent->getOperands();

  if (getNumOperands() != values.size())
    return emitOpError("has ")
           << getNumOperands()
           << " operands, but enclosing interop operation requires "
           << values.size() << " values";

  for (auto it :
       llvm::enumerate(llvm::zip(getOperandTypes(), values.getTypes()))) {
    auto [returnOperandType, parentType] = it.value();
    if (returnOperandType != parentType)
      return emitError() << "type of return operand " << it.index() << " ("
                         << returnOperandType
                         << ") doesn't match required type (" << parentType
                         << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Interop/Interop.cpp.inc"
#include "circt/Dialect/Interop/InteropEnums.cpp.inc"
