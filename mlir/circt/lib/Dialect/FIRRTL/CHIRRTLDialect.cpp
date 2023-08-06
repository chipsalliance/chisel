//===- CHIRRTLDialect.cpp - Implement the CHIRRTL dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CHIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace chirrtl;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Parsing and Printing helpers.
//===----------------------------------------------------------------------===//

static ParseResult parseCHIRRTLOp(OpAsmParser &parser,
                                  NamedAttrList &resultAttrs) {
  // Add an empty annotation array if none were parsed.
  auto result = parser.parseOptionalAttrDict(resultAttrs);
  if (!resultAttrs.get("annotations"))
    resultAttrs.append("annotations", parser.getBuilder().getArrayAttr({}));

  // If the attribute dictionary contains no 'name' attribute, infer it from
  // the SSA name (if specified).
  if (resultAttrs.get("name"))
    return success();

  auto resultName = parser.getResultName(0).first;
  if (!resultName.empty() && isdigit(resultName[0]))
    resultName = "";
  auto nameAttr = parser.getBuilder().getStringAttr(resultName);
  auto *context = parser.getBuilder().getContext();
  resultAttrs.push_back({StringAttr::get(context, "name"), nameAttr});
  return result;
}

static void printCHIRRTLOp(OpAsmPrinter &p, Operation *op, DictionaryAttr attr,
                           ArrayRef<StringRef> extraElides = {}) {
  SmallVector<StringRef> elides(extraElides.begin(), extraElides.end());

  // Elide the symbol.
  elides.push_back(hw::InnerSymbolTable::getInnerSymbolAttrName());

  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  p.printOperand(op->getResult(0), tmpStream);
  auto actualName = tmpStream.str().drop_front();
  auto expectedName = op->getAttrOfType<StringAttr>("name").getValue();
  // Anonymous names are printed as digits, which is fine.
  if (actualName == expectedName ||
      (expectedName.empty() && isdigit(actualName[0])))
    elides.push_back("name");
  elides.push_back("nameKind");

  // Elide "annotations" if it is empty.
  if (op->getAttrOfType<ArrayAttr>("annotations").empty())
    elides.push_back("annotations");

  p.printOptionalAttrDict(op->getAttrs(), elides);
}

//===----------------------------------------------------------------------===//
// NameKind Custom Directive
//===----------------------------------------------------------------------===//

static ParseResult parseNameKind(OpAsmParser &parser,
                                 firrtl::NameKindEnumAttr &result) {
  StringRef keyword;

  if (!parser.parseOptionalKeyword(&keyword,
                                   {"interesting_name", "droppable_name"})) {
    auto kind = symbolizeNameKindEnum(keyword);
    result = NameKindEnumAttr::get(parser.getContext(), kind.value());
    return success();
  }

  // Default is droppable name.
  result =
      NameKindEnumAttr::get(parser.getContext(), NameKindEnum::DroppableName);
  return success();
}

static void printNameKind(OpAsmPrinter &p, Operation *op,
                          firrtl::NameKindEnumAttr attr,
                          ArrayRef<StringRef> extraElides = {}) {
  if (attr.getValue() != NameKindEnum::DroppableName)
    p << " " << stringifyNameKindEnum(attr.getValue());
}

//===----------------------------------------------------------------------===//
// MemoryPortOp
//===----------------------------------------------------------------------===//

void MemoryPortOp::build(OpBuilder &builder, OperationState &result,
                         Type dataType, Value memory, MemDirAttr direction,
                         StringRef name, ArrayRef<Attribute> annotations) {
  build(builder, result, CMemoryPortType::get(builder.getContext()), dataType,
        memory, direction, name, builder.getArrayAttr(annotations));
}

LogicalResult MemoryPortOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  auto inType = operands[0].getType();
  auto memType = type_dyn_cast<CMemoryType>(inType);
  if (!memType) {
    if (loc)
      mlir::emitError(*loc, "memory port requires memory operand");
    return failure();
  }
  results.push_back(memType.getElementType());
  results.push_back(CMemoryPortType::get(context));
  return success();
}

LogicalResult MemoryPortOp::verify() {
  // MemoryPorts require exactly 1 access. Right now there are no other
  // operations that could be using that value due to the types.
  if (!getPort().hasOneUse())
    return emitOpError("port should be used by a chirrtl.memoryport.access");
  return success();
}

MemoryPortAccessOp MemoryPortOp::getAccess() {
  auto uses = getPort().use_begin();
  if (uses == getPort().use_end())
    return {};
  return cast<MemoryPortAccessOp>(uses->getOwner());
}

void MemoryPortOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  StringRef base = getName();
  if (base.empty())
    base = "memport";
  setNameFn(getData(), (base + "_data").str());
  setNameFn(getPort(), (base + "_port").str());
}

static ParseResult parseMemoryPortOp(OpAsmParser &parser,
                                     NamedAttrList &resultAttrs) {
  // Add an empty annotation array if none were parsed.
  auto result = parser.parseOptionalAttrDict(resultAttrs);
  if (!resultAttrs.get("annotations"))
    resultAttrs.append("annotations", parser.getBuilder().getArrayAttr({}));
  return result;
}

/// Always elide "direction" and elide "annotations" if it exists or
/// if it is empty.
static void printMemoryPortOp(OpAsmPrinter &p, Operation *op,
                              DictionaryAttr attr) {
  // "direction" is always elided.
  SmallVector<StringRef> elides = {"direction"};
  // Annotations elided if empty.
  if (op->getAttrOfType<ArrayAttr>("annotations").empty())
    elides.push_back("annotations");
  p.printOptionalAttrDict(op->getAttrs(), elides);
}

//===----------------------------------------------------------------------===//
// MemoryDebugPortOp
//===----------------------------------------------------------------------===//

void MemoryDebugPortOp::build(OpBuilder &builder, OperationState &result,
                              Type dataType, Value memory, StringRef name,
                              ArrayRef<Attribute> annotations) {
  build(builder, result, dataType, memory, name,
        builder.getArrayAttr(annotations));
}

LogicalResult MemoryDebugPortOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  auto inType = operands[0].getType();
  auto memType = type_dyn_cast<CMemoryType>(inType);
  if (!memType) {
    if (loc)
      mlir::emitError(*loc, "memory port requires memory operand");
    return failure();
  }
  results.push_back(RefType::get(
      FVectorType::get(memType.getElementType(), memType.getNumElements())));
  return success();
}

void MemoryDebugPortOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  StringRef base = getName();
  if (base.empty())
    base = "memport";
  setNameFn(getData(), (base + "_data").str());
}

static ParseResult parseMemoryDebugPortOp(OpAsmParser &parser,
                                          NamedAttrList &resultAttrs) {
  // Add an empty annotation array if none were parsed.
  auto result = parser.parseOptionalAttrDict(resultAttrs);
  if (!resultAttrs.get("annotations"))
    resultAttrs.append("annotations", parser.getBuilder().getArrayAttr({}));
  return result;
}

/// Always elide "direction" and elide "annotations" if it exists or
/// if it is empty.
static void printMemoryDebugPortOp(OpAsmPrinter &p, Operation *op,
                                   DictionaryAttr attr) {
  SmallVector<StringRef, 1> elides;
  // Annotations elided if empty.
  if (op->getAttrOfType<ArrayAttr>("annotations").empty())
    elides.push_back("annotations");
  p.printOptionalAttrDict(op->getAttrs(), elides);
}

//===----------------------------------------------------------------------===//
// CombMemOp
//===----------------------------------------------------------------------===//

static ParseResult parseCombMemOp(OpAsmParser &parser,
                                  NamedAttrList &resultAttrs) {
  return parseCHIRRTLOp(parser, resultAttrs);
}

static void printCombMemOp(OpAsmPrinter &p, Operation *op,
                           DictionaryAttr attr) {
  printCHIRRTLOp(p, op, attr);
}

void CombMemOp::build(OpBuilder &builder, OperationState &result,
                      FIRRTLBaseType elementType, uint64_t numElements,
                      StringRef name, NameKindEnum nameKind,
                      ArrayAttr annotations, StringAttr innerSym,
                      MemoryInitAttr init) {
  build(builder, result,
        CMemoryType::get(builder.getContext(), elementType, numElements), name,
        nameKind, annotations,
        innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr(), init);
}

void CombMemOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

std::optional<size_t> CombMemOp::getTargetResultIndex() {
  // Inner symbols on comb memory operations target the op not any result.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// SeqMemOp
//===----------------------------------------------------------------------===//

static ParseResult parseSeqMemOp(OpAsmParser &parser,
                                 NamedAttrList &resultAttrs) {
  return parseCHIRRTLOp(parser, resultAttrs);
}

/// Always elide "ruw" and elide "annotations" if it exists or if it is empty.
static void printSeqMemOp(OpAsmPrinter &p, Operation *op, DictionaryAttr attr) {
  printCHIRRTLOp(p, op, attr, {"ruw"});
}

void SeqMemOp::build(OpBuilder &builder, OperationState &result,
                     FIRRTLBaseType elementType, uint64_t numElements,
                     RUWAttr ruw, StringRef name, NameKindEnum nameKind,
                     ArrayAttr annotations, StringAttr innerSym,
                     MemoryInitAttr init) {
  build(builder, result,
        CMemoryType::get(builder.getContext(), elementType, numElements), ruw,
        name, nameKind, annotations,
        innerSym ? hw::InnerSymAttr::get(innerSym) : hw::InnerSymAttr(), init);
}

void SeqMemOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

std::optional<size_t> SeqMemOp::getTargetResultIndex() {
  // Inner symbols on seq memory operations target the op not any result.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// CHIRRTL Dialect
//===----------------------------------------------------------------------===//

// This is used to give custom SSA names which match the "name" attribute of the
// memory operation, which allows us to elide the name attribute.
namespace {
struct CHIRRTLOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  void getAsmResultNames(Operation *op, OpAsmSetValueNameFn setNameFn) const {
    // Many CHIRRTL dialect operations have an optional 'name' attribute.  If
    // present, use it.
    if (op->getNumResults() == 1)
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        setNameFn(op->getResult(0), nameAttr.getValue());
  }
};
} // namespace

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/FIRRTL/CHIRRTLTypes.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/CHIRRTL.cpp.inc"

void CHIRRTLDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/FIRRTL/CHIRRTL.cpp.inc"
      >();

  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/FIRRTL/CHIRRTLTypes.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<CHIRRTLOpAsmDialectInterface>();
}

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CMemory Type
//===----------------------------------------------------------------------===//

void CMemoryType::print(AsmPrinter &printer) const {
  printer << "<";
  // Don't print element types with "!firrtl.".
  firrtl::printNestedType(getElementType(), printer);
  printer << ", " << getNumElements() << ">";
}

Type CMemoryType::parse(AsmParser &parser) {
  FIRRTLBaseType elementType;
  uint64_t numElements;
  if (parser.parseLess() || firrtl::parseNestedBaseType(elementType, parser) ||
      parser.parseComma() || parser.parseInteger(numElements) ||
      parser.parseGreater())
    return {};
  return parser.getChecked<CMemoryType>(elementType, numElements);
}

LogicalResult CMemoryType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  FIRRTLBaseType elementType,
                                  uint64_t numElements) {
  if (!elementType.isPassive()) {
    return emitError() << "behavioral memory element type must be passive";
  }
  return success();
}
