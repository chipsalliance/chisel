//===- SVAttributes.cpp - Implement SV attributes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::sv;

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SV/SVAttributes.cpp.inc"

#include "circt/Dialect/SV/SVEnums.cpp.inc"

void SVDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/SV/SVAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// SV Attribute Modification Helpers
//===----------------------------------------------------------------------===//

bool sv::hasSVAttributes(Operation *op) {
  if (auto attrs = getSVAttributes(op))
    return !attrs.empty();
  return false;
}

ArrayAttr sv::getSVAttributes(Operation *op) {
  auto attrs = op->getAttr(SVAttributeAttr::getSVAttributesAttrName());
  if (!attrs)
    return {};
  auto arrayAttr = attrs.dyn_cast<ArrayAttr>();
  if (!arrayAttr) {
    op->emitOpError("'sv.attributes' must be an array attribute");
    return {};
  }
  for (auto attr : arrayAttr) {
    if (!attr.isa<SVAttributeAttr>()) {
      op->emitOpError("'sv.attributes' elements must be `SVAttributeAttr`s");
      return {};
    }
  }
  if (arrayAttr.empty())
    return {};
  return arrayAttr;
}

void sv::setSVAttributes(Operation *op, ArrayAttr attrs) {
  if (attrs && !attrs.getValue().empty())
    op->setAttr(SVAttributeAttr::getSVAttributesAttrName(), attrs);
  else
    op->removeAttr(SVAttributeAttr::getSVAttributesAttrName());
}

void sv::setSVAttributes(Operation *op, ArrayRef<SVAttributeAttr> attrs) {
  if (attrs.empty())
    return sv::setSVAttributes(op, ArrayAttr());
  SmallVector<Attribute> filteredAttrs;
  SmallPtrSet<Attribute, 4> seenAttrs;
  filteredAttrs.reserve(attrs.size());
  for (auto attr : attrs)
    if (seenAttrs.insert(attr).second)
      filteredAttrs.push_back(attr);
  sv::setSVAttributes(op, ArrayAttr::get(op->getContext(), filteredAttrs));
}

bool sv::modifySVAttributes(
    Operation *op, llvm::function_ref<void(SmallVectorImpl<SVAttributeAttr> &)>
                       modifyCallback) {
  ArrayRef<Attribute> oldAttrs;
  if (auto attrs = sv::getSVAttributes(op))
    oldAttrs = attrs.getValue();

  SmallVector<SVAttributeAttr> newAttrs;
  newAttrs.reserve(oldAttrs.size());
  for (auto oldAttr : oldAttrs)
    newAttrs.push_back(cast<SVAttributeAttr>(oldAttr));
  modifyCallback(newAttrs);

  if (newAttrs.size() == oldAttrs.size() &&
      llvm::none_of(llvm::zip(oldAttrs, newAttrs), [](auto pair) {
        return std::get<0>(pair) != std::get<1>(pair);
      }))
    return false;

  sv::setSVAttributes(op, newAttrs);
  return true;
}

unsigned sv::addSVAttributes(Operation *op,
                             ArrayRef<SVAttributeAttr> newAttrs) {
  if (newAttrs.empty())
    return 0;
  unsigned numAdded = 0;
  modifySVAttributes(op, [&](auto &attrs) {
    SmallPtrSet<Attribute, 4> seenAttrs(attrs.begin(), attrs.end());
    for (auto newAttr : newAttrs) {
      if (seenAttrs.insert(newAttr).second) {
        attrs.push_back(newAttr);
        ++numAdded;
      }
    }
  });
  return numAdded;
}

unsigned sv::removeSVAttributes(
    Operation *op, llvm::function_ref<bool(SVAttributeAttr)> removeCallback) {
  unsigned numRemoved = 0;
  sv::modifySVAttributes(op, [&](auto &attrs) {
    // Only keep attributes for which the callback returns false.
    unsigned inIdx = 0, outIdx = 0, endIdx = attrs.size();
    for (; inIdx != endIdx; ++inIdx) {
      if (removeCallback(attrs[inIdx]))
        ++numRemoved;
      else
        attrs[outIdx++] = attrs[inIdx];
    }
    attrs.truncate(outIdx);
  });
  return numRemoved;
}

unsigned sv::removeSVAttributes(Operation *op,
                                ArrayRef<SVAttributeAttr> attrs) {
  SmallPtrSet<Attribute, 4> attrSet;
  for (auto attr : attrs)
    attrSet.insert(attr);
  return removeSVAttributes(op,
                            [&](auto attr) { return attrSet.contains(attr); });
}

//===----------------------------------------------------------------------===//
// SVAttributeAttr
//===----------------------------------------------------------------------===//

mlir::Attribute SVAttributeAttr::parse(mlir::AsmParser &p, mlir::Type type) {
  StringAttr nameAttr;
  if (p.parseLess() || p.parseAttribute(nameAttr))
    return {};

  StringAttr expressionAttr;
  if (!p.parseOptionalEqual())
    if (p.parseAttribute(expressionAttr))
      return {};

  bool emitAsComment = false;
  if (!p.parseOptionalComma()) {
    if (p.parseKeyword("emitAsComment"))
      return {};
    emitAsComment = true;
  }

  if (p.parseGreater())
    return {};

  return SVAttributeAttr::get(p.getContext(), nameAttr, expressionAttr,
                              BoolAttr::get(p.getContext(), emitAsComment));
}

void SVAttributeAttr::print(::mlir::AsmPrinter &p) const {
  p << "<" << getName();
  if (auto expr = getExpression())
    p << " = " << expr;
  if (getEmitAsComment().getValue())
    p << ", emitAsComment";
  p << ">";
}
