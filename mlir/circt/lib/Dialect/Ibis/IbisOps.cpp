//===- IbisOps.cpp - Implementation of Ibis dialect ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Support/ParsingUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace circt;
using namespace ibis;

ParseResult MethodOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  SmallVector<OpAsmParser::Argument, 4> args;
  SmallVector<Attribute> argNames;
  SmallVector<Type> resultTypes;
  TypeAttr functionType;

  using namespace mlir::function_interface_impl;
  auto *context = parser.getContext();

  // Parse the argument list.
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();

  // Parse the result type.
  if (succeeded(parser.parseOptionalArrow())) {
    Type resultType;
    if (parser.parseType(resultType))
      return failure();
    resultTypes.push_back(resultType);
  }

  // Process the ssa args for the information we're looking for.
  SmallVector<Type> argTypes;
  for (auto &arg : args) {
    argNames.push_back(parsing_util::getNameFromSSA(context, arg.ssaName.name));
    argTypes.push_back(arg.type);
    if (!arg.sourceLoc)
      arg.sourceLoc = parser.getEncodedSourceLoc(arg.ssaName.location);
  }

  functionType =
      TypeAttr::get(FunctionType::get(context, argTypes, resultTypes));

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  result.addAttribute("argNames", ArrayAttr::get(context, argNames));
  result.addAttribute(MethodOp::getFunctionTypeAttrName(result.name),
                      functionType);

  // Parse the function body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();

  ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void MethodOp::print(OpAsmPrinter &p) {
  FunctionType funcTy = getFunctionType();
  p << ' ';
  p.printSymbolName(getSymName());
  function_interface_impl::printFunctionSignature(
      p, *this, funcTy.getInputs(), /*isVariadic=*/false, funcTy.getResults());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     getAttributeNames());
  Region &body = getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

void MethodOp::getAsmBlockArgumentNames(mlir::Region &region,
                                        OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;

  auto func = cast<MethodOp>(region.getParentOp());
  auto argNames = func.getArgNames().getAsRange<StringAttr>();
  auto *block = &region.front();

  for (auto [idx, argName] : llvm::enumerate(argNames))
    if (!argName.getValue().empty())
      setNameFn(block->getArgument(idx), argName);
}

LogicalResult MethodOp::verify() {
  // Check that we have only one return value.
  if (getFunctionType().getNumResults() > 1)
    return failure();
  return success();
}

void ReturnOp::build(OpBuilder &odsBuilder, OperationState &odsState) {}

LogicalResult ReturnOp::verify() {
  // Check that the return operand type matches the function return type.
  auto func = cast<MethodOp>((*this)->getParentOp());
  ArrayRef<Type> resTypes = func.getResultTypes();
  assert(resTypes.size() <= 1);
  assert(getNumOperands() <= 1);

  if (resTypes.empty()) {
    if (getNumOperands() != 0)
      return emitOpError(
          "cannot return a value from a function with no result type");
    return success();
  }

  Value retValue = getRetValue();
  if (!retValue)
    return emitOpError("must return a value");

  Type retType = retValue.getType();
  if (retType != resTypes.front())
    return emitOpError("return type (")
           << retType << ") must match function return type ("
           << resTypes.front() << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto targetClass = getClass(&symbolTable);
  if (!targetClass)
    return emitOpError() << "'" << getClassName() << "' does not exist";

  return success();
}

//===----------------------------------------------------------------------===//
// PortReadOp
//===----------------------------------------------------------------------===//

// Verifies that a given source operation (TSrcOp is a port access) access a
// symbol defined by the expected target operation (TTargetOp).
template <typename TSrcOp, typename TTargetOp>
LogicalResult verifyPortSymbolUses(
    TSrcOp op, StringAttr symName, llvm::function_ref<Type(TSrcOp)> getPortType,
    SymbolTableCollection &symbolTable,
    llvm::function_ref<FailureOr<TTargetOp>(Operation *)> getTargetOp) {

  ClassOp parentClass = op->template getParentOfType<ClassOp>();
  assert(parentClass && " must be contained in a ClassOp");
  // TODO @teqdruid: use innerSym when available.
  Operation *rootAccessOp = symbolTable.lookupSymbolIn(parentClass, symName);

  TTargetOp targetOp;
  if (auto getTargetRes = getTargetOp(rootAccessOp); succeeded(getTargetRes))
    targetOp = *getTargetRes;
  else
    return failure();

  Type expectedType = getPortType(op);
  Type actualType = targetOp.getType();
  if (actualType != expectedType)
    return op->emitOpError() << "Expected type '" << expectedType
                             << "' does not match actual port type, which was '"
                             << actualType << "'";
  return success();
}

// verifies that a local port access (non-nested symbol reference) is valid (the
// target symbol exists) and that the types of the port and the access match.
template <typename TSrcOp, typename TTargetOp>
LogicalResult
verifyLocalPortSymbolUse(TSrcOp op,
                         llvm::function_ref<Type(TSrcOp)> getPortType,
                         SymbolTableCollection &symbolTable) {
  FlatSymbolRefAttr symName = op.getSymNameAttr();
  auto getTargetOp = [&](Operation *rootAccessOp) -> FailureOr<TTargetOp> {
    auto targetOp = dyn_cast_or_null<TTargetOp>(rootAccessOp);
    if (!targetOp)
      return op->emitOpError()
             << "expected '" << symName << "' to refer to a '"
             << TTargetOp::getOperationName() << "' operation";
    return {targetOp};
  };

  return verifyPortSymbolUses<TSrcOp, TTargetOp>(
      op, symName.getAttr(), getPortType, symbolTable, getTargetOp);
}

LogicalResult PortReadOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyLocalPortSymbolUse<PortReadOp, OutputPortOp>(
      *this, [](PortReadOp op) { return op.getOutput().getType(); },
      symbolTable);
}

//===----------------------------------------------------------------------===//
// PortWriteOp
//===----------------------------------------------------------------------===//

LogicalResult
PortWriteOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyLocalPortSymbolUse<PortWriteOp, InputPortOp>(
      *this, [](PortWriteOp op) { return op.getInput().getType(); },
      symbolTable);
}

//===----------------------------------------------------------------------===//
// InstanceReadOp
//===----------------------------------------------------------------------===//

// verifies that an instance port access (nested symbol reference) is valid
// (the target instance exists, and the target port inside the referenced class
// exists) and that the types of the port and the access match.
template <typename TSrcOp, typename TTargetOp>
LogicalResult
verifyInstancePortSymbolUses(TSrcOp op,
                             llvm::function_ref<Type(TSrcOp)> getPortType,
                             SymbolTableCollection &symbolTable) {
  auto symName = op.getSymNameAttr();
  auto getTargetOp = [&](Operation *rootAccessOp) -> FailureOr<TTargetOp> {
    auto targetInstance = dyn_cast_or_null<InstanceOp>(rootAccessOp);
    if (!targetInstance)
      return op->emitOpError()
             << "expected " << symName.getRootReference() << " to refer to a '"
             << InstanceOp::getOperationName() << "' operation";

    // Lookup the port in the instance. For now, only allow top level accesses -
    // can easily extend this to nested instances as well.
    ClassOp referencedClass = targetInstance.getClass();
    // @teqdruid TODO: make this more efficient using
    // innersymtablecollection when that's available to non-firrtl dialects.
    auto targetOp = symbolTable.lookupSymbolIn<TTargetOp>(
        referencedClass, symName.getLeafReference());

    if (!targetOp)
      return op->emitOpError() << "'" << symName << "' does not exist";

    return {targetOp};
  };

  return verifyPortSymbolUses<TSrcOp, TTargetOp>(
      op, symName.getRootReference(), getPortType, symbolTable, getTargetOp);
}

LogicalResult
InstanceReadOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyInstancePortSymbolUses<InstanceReadOp, OutputPortOp>(
      *this, [](InstanceReadOp op) { return op.getOutput().getType(); },
      symbolTable);
}

//===----------------------------------------------------------------------===//
// InstanceWriteOp
//===----------------------------------------------------------------------===//

LogicalResult
InstanceWriteOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyInstancePortSymbolUses<InstanceWriteOp, InputPortOp>(
      *this, [](InstanceWriteOp op) { return op.getInput().getType(); },
      symbolTable);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Ibis/Ibis.cpp.inc"
