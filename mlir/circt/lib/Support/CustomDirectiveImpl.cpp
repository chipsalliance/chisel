//===- CustomDirectiveImpl.cpp - Custom TableGen directives ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/CustomDirectiveImpl.h"
#include "llvm/ADT/SmallString.h"

using namespace circt;

ParseResult circt::parseImplicitSSAName(OpAsmParser &parser, StringAttr &attr) {
  // Use the explicit name if one is provided as `name "xyz"`.
  if (!parser.parseOptionalKeyword("name")) {
    std::string str;
    if (parser.parseString(&str))
      return failure();
    attr = parser.getBuilder().getStringAttr(str);
    return success();
  }

  // Infer the name from the SSA name of the operation's first result.
  auto resultName = parser.getResultName(0).first;
  if (!resultName.empty() && isdigit(resultName[0]))
    resultName = "";
  attr = parser.getBuilder().getStringAttr(resultName);
  return success();
}

ParseResult circt::parseImplicitSSAName(OpAsmParser &parser,
                                        NamedAttrList &attrs) {
  if (parser.parseOptionalAttrDict(attrs))
    return failure();
  inferImplicitSSAName(parser, attrs);
  return success();
}

bool circt::inferImplicitSSAName(OpAsmParser &parser, NamedAttrList &attrs) {
  // Don't do anything if a `name` attribute is explicitly provided.
  if (attrs.get("name"))
    return false;

  // Infer the name from the SSA name of the operation's first result.
  auto resultName = parser.getResultName(0).first;
  if (!resultName.empty() && isdigit(resultName[0]))
    resultName = "";
  auto nameAttr = parser.getBuilder().getStringAttr(resultName);
  auto *context = parser.getBuilder().getContext();
  attrs.push_back({StringAttr::get(context, "name"), nameAttr});
  return true;
}

void circt::printImplicitSSAName(OpAsmPrinter &printer, Operation *op,
                                 StringAttr attr) {
  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  printer.printOperand(op->getResult(0), tmpStream);
  auto actualName = tmpStream.str().drop_front();
  auto expectedName = attr.getValue();
  // Anonymous names are printed as digits, which is fine.
  if (actualName == expectedName ||
      (expectedName.empty() && isdigit(actualName[0])))
    return;

  printer << " name " << attr;
}

void circt::printImplicitSSAName(OpAsmPrinter &printer, Operation *op,
                                 DictionaryAttr attrs,
                                 ArrayRef<StringRef> extraElides) {
  SmallVector<StringRef, 2> elides(extraElides.begin(), extraElides.end());
  elideImplicitSSAName(printer, op, attrs, elides);
  printer.printOptionalAttrDict(attrs.getValue(), elides);
}

void circt::elideImplicitSSAName(OpAsmPrinter &printer, Operation *op,
                                 DictionaryAttr attrs,
                                 SmallVectorImpl<StringRef> &elides) {
  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  printer.printOperand(op->getResult(0), tmpStream);
  auto actualName = tmpStream.str().drop_front();
  auto expectedName = attrs.getAs<StringAttr>("name").getValue();
  // Anonymous names are printed as digits, which is fine.
  if (actualName == expectedName ||
      (expectedName.empty() && isdigit(actualName[0])))
    elides.push_back("name");
}

ParseResult circt::parseOptionalBinaryOpTypes(OpAsmParser &parser, Type &lhs,
                                              Type &rhs) {
  if (parser.parseType(lhs))
    return failure();

  // Parse an optional rhs type.
  if (parser.parseOptionalComma()) {
    rhs = lhs;
  } else {
    if (parser.parseType(rhs))
      return failure();
  }
  return success();
}

void circt::printOptionalBinaryOpTypes(OpAsmPrinter &p, Operation *op, Type lhs,
                                       Type rhs) {
  p << lhs;
  // If operand types are not same, print a rhs type.
  if (lhs != rhs)
    p << ", " << rhs;
}

ParseResult circt::parseKeywordBool(OpAsmParser &parser, BoolAttr &attr,
                                    StringRef trueKeyword,
                                    StringRef falseKeyword) {
  if (succeeded(parser.parseOptionalKeyword(trueKeyword))) {
    attr = BoolAttr::get(parser.getContext(), true);
  } else if (succeeded(parser.parseOptionalKeyword(falseKeyword))) {
    attr = BoolAttr::get(parser.getContext(), false);
  } else {
    return parser.emitError(parser.getCurrentLocation())
           << "expected keyword \"" << trueKeyword << "\" or \"" << falseKeyword
           << "\"";
  }
  return success();
}

void circt::printKeywordBool(OpAsmPrinter &printer, Operation *op,
                             BoolAttr attr, StringRef trueKeyword,
                             StringRef falseKeyword) {
  if (attr.getValue())
    printer << trueKeyword;
  else
    printer << falseKeyword;
}
