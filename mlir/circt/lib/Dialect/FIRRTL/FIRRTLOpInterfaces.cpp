//===- FIRRTLOpInterfaces.cpp - Implement the FIRRTL op interfaces --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the FIRRTL operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace llvm;
using namespace circt::firrtl;

LogicalResult circt::firrtl::verifyModuleLikeOpInterface(FModuleLike module) {
  // Verify port types first.  This is used as the basis for the number of
  // ports required everywhere else.
  auto portTypes = module.getPortTypesAttr();
  if (!portTypes || llvm::any_of(portTypes.getValue(), [](Attribute attr) {
        return !isa<TypeAttr>(attr);
      }))
    return module.emitOpError("requires valid port types");

  auto numPorts = portTypes.size();

  // Verify the port dirctions.
  auto portDirections = module.getPortDirectionsAttr();
  if (!portDirections)
    return module.emitOpError("requires valid port direction");
  // TODO: bitwidth is 1 when there are no ports, since APInt previously did not
  // support 0 bit widths.
  auto bitWidth = portDirections.getValue().getBitWidth();
  if (bitWidth != numPorts)
    return module.emitOpError("requires ") << numPorts << " port directions";

  // Verify the port names.
  auto portNames = module.getPortNamesAttr();
  if (!portNames)
    return module.emitOpError("requires valid port names");
  if (portNames.size() != numPorts)
    return module.emitOpError("requires ") << numPorts << " port names";
  if (llvm::any_of(portNames.getValue(),
                   [](Attribute attr) { return !isa<StringAttr>(attr); }))
    return module.emitOpError("port names should all be string attributes");

  // Verify the port annotations.
  auto portAnnotations = module.getPortAnnotationsAttr();
  if (!portAnnotations)
    return module.emitOpError("requires valid port annotations");
  // TODO: it seems weird to allow empty port annotations.
  if (!portAnnotations.empty() && portAnnotations.size() != numPorts)
    return module.emitOpError("requires ") << numPorts << " port annotations";
  // TODO: Move this into an annotation verifier.
  for (auto annos : portAnnotations.getValue()) {
    auto arrayAttr = dyn_cast<ArrayAttr>(annos);
    if (!arrayAttr)
      return module.emitOpError(
          "requires port annotations be array attributes");
    if (llvm::any_of(arrayAttr.getValue(),
                     [](Attribute attr) { return !isa<DictionaryAttr>(attr); }))
      return module.emitOpError(
          "annotations must be dictionaries or subannotations");
  }

  // Verify the port symbols.
  auto portSymbols = module.getPortSymbolsAttr();
  if (!portSymbols)
    return module.emitOpError("requires valid port symbols");
  if (!portSymbols.empty() && portSymbols.size() != numPorts)
    return module.emitOpError("requires ") << numPorts << " port symbols";
  if (llvm::any_of(portSymbols.getValue(), [](Attribute attr) {
        return !attr || !isa<hw::InnerSymAttr>(attr);
      }))
    return module.emitOpError("port symbols should all be InnerSym attributes");

  // Verify the port locations.
  auto portLocs = module.getPortLocationsAttr();
  if (!portLocs)
    return module.emitOpError("requires valid port locations");
  if (portLocs.size() != numPorts)
    return module.emitOpError("requires ") << numPorts << " port locations";
  if (llvm::any_of(portLocs.getValue(), [](Attribute attr) {
        return !attr || !isa<LocationAttr>(attr);
      }))
    return module.emitOpError("port symbols should all be location attributes");

  // Verify the body.
  if (module->getNumRegions() != 1)
    return module.emitOpError("requires one region");

  return success();
}

//===----------------------------------------------------------------------===//
// Forceable
//===----------------------------------------------------------------------===//

RefType circt::firrtl::detail::getForceableResultType(bool forceable,
                                                      Type type) {
  auto base = dyn_cast_or_null<FIRRTLBaseType>(type);
  // TODO: Find a way to not check same things RefType::get/verify does.
  if (!forceable || !base || base.containsConst())
    return {};
  return circt::firrtl::RefType::get(base.getPassiveType(), forceable);
}

LogicalResult circt::firrtl::detail::verifyForceableOp(Forceable op) {
  bool forceable = op.isForceable();
  auto ref = op.getDataRef();
  if ((bool)ref != forceable)
    return op.emitOpError("must have ref result iff marked forceable");
  if (!forceable)
    return success();
  auto data = op.getDataRaw();
  auto baseType = type_dyn_cast<FIRRTLBaseType>(data.getType());
  if (!baseType)
    return op.emitOpError("has data that is not a base type");
  if (baseType.containsConst())
    return op.emitOpError("cannot force a declaration of constant type");
  auto expectedRefType = getForceableResultType(forceable, baseType);
  if (ref.getType() != expectedRefType)
    return op.emitOpError("reference result of incorrect type, found ")
           << ref.getType() << ", expected " << expectedRefType;
  return success();
}

namespace {
/// Simple wrapper to allow construction from a context for local use.
class TrivialPatternRewriter : public PatternRewriter {
public:
  explicit TrivialPatternRewriter(MLIRContext *context)
      : PatternRewriter(context) {}
};
} // end namespace

Forceable
circt::firrtl::detail::replaceWithNewForceability(Forceable op, bool forceable,
                                                  PatternRewriter *rewriter) {
  if (forceable == op.isForceable())
    return op;

  assert(op->getNumRegions() == 0);

  // Create copy of this operation with/without the forceable marker + result
  // type.

  TrivialPatternRewriter localRewriter(op.getContext());
  PatternRewriter &rw = rewriter ? *rewriter : localRewriter;

  // Grab the current operation's results and attributes.
  SmallVector<Type, 8> resultTypes(op->getResultTypes());
  SmallVector<NamedAttribute, 16> attributes(op->getAttrs());

  // Add/remove the optional ref result.
  auto refType = firrtl::detail::getForceableResultType(true, op.getDataType());
  if (forceable)
    resultTypes.push_back(refType);
  else {
    assert(resultTypes.back() == refType &&
           "expected forceable type as last result");
    resultTypes.pop_back();
  }

  // Add/remove the forceable marker.
  auto forceableMarker =
      rw.getNamedAttr(op.getForceableAttrName(), rw.getUnitAttr());
  if (forceable)
    attributes.push_back(forceableMarker);
  else {
    llvm::erase_value(attributes, forceableMarker);
    assert(attributes.size() != op->getAttrs().size());
  }

  // Create the replacement operation.
  OperationState state(op.getLoc(), op->getName(), op->getOperands(),
                       resultTypes, attributes, op->getSuccessors());
  rw.setInsertionPoint(op);
  auto *replace = rw.create(state);

  // Dropping forceability (!forceable) -> no uses of forceable ref handle.
  assert(forceable || op.getDataRef().use_empty());

  // Replace results.
  for (auto result : llvm::drop_end(op->getResults(), forceable ? 0 : 1))
    rw.replaceAllUsesWith(result, replace->getResult(result.getResultNumber()));
  rw.eraseOp(op);
  return cast<Forceable>(replace);
}

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.cpp.inc"
