//===- CustomDirectiveImpl.cpp - Custom directive definitions -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/CustomDirectiveImpl.h"
#include "circt/Dialect/HW/HWAttributes.h"

using namespace circt;
using namespace circt::hw;

ParseResult circt::parseInputPortList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inputs,
    SmallVectorImpl<Type> &inputTypes, ArrayAttr &inputNames) {

  SmallVector<Attribute> argNames;
  auto parseInputPort = [&]() -> ParseResult {
    std::string portName;
    if (parser.parseKeywordOrString(&portName))
      return failure();
    argNames.push_back(StringAttr::get(parser.getContext(), portName));
    inputs.push_back({});
    inputTypes.push_back({});
    return failure(parser.parseColon() || parser.parseOperand(inputs.back()) ||
                   parser.parseColon() || parser.parseType(inputTypes.back()));
  };
  llvm::SMLoc inputsOperandsLoc;
  if (parser.getCurrentLocation(&inputsOperandsLoc) ||
      parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseInputPort))
    return failure();

  inputNames = ArrayAttr::get(parser.getContext(), argNames);

  return success();
}

void circt::printInputPortList(OpAsmPrinter &p, Operation *op,
                               OperandRange inputs, TypeRange inputTypes,
                               ArrayAttr inputNames) {
  p << "(";
  llvm::interleaveComma(llvm::zip(inputs, inputNames), p,
                        [&](std::tuple<Value, Attribute> input) {
                          Value val = std::get<0>(input);
                          p.printKeywordOrString(
                              std::get<1>(input).cast<StringAttr>().getValue());
                          p << ": " << val << ": " << val.getType();
                        });
  p << ")";
}

ParseResult circt::parseOutputPortList(OpAsmParser &parser,
                                       SmallVectorImpl<Type> &resultTypes,
                                       ArrayAttr &resultNames) {

  SmallVector<Attribute> names;
  auto parseResultPort = [&]() -> ParseResult {
    std::string portName;
    if (parser.parseKeywordOrString(&portName))
      return failure();
    names.push_back(StringAttr::get(parser.getContext(), portName));
    resultTypes.push_back({});
    return parser.parseColonType(resultTypes.back());
  };
  llvm::SMLoc inputsOperandsLoc;
  if (parser.getCurrentLocation(&inputsOperandsLoc) ||
      parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseResultPort))
    return failure();

  resultNames = ArrayAttr::get(parser.getContext(), names);

  return success();
}

void circt::printOutputPortList(OpAsmPrinter &p, Operation *op,
                                TypeRange resultTypes, ArrayAttr resultNames) {
  p << "(";
  llvm::interleaveComma(
      llvm::zip(resultTypes, resultNames), p,
      [&](std::tuple<Type, Attribute> result) {
        p.printKeywordOrString(
            std::get<1>(result).cast<StringAttr>().getValue());
        p << ": " << std::get<0>(result);
      });
  p << ")";
}

ParseResult circt::parseOptionalParameterList(OpAsmParser &parser,
                                              ArrayAttr &parameters) {
  SmallVector<Attribute> params;

  auto parseParameter = [&]() {
    std::string name;
    Type type;
    Attribute value;

    if (parser.parseKeywordOrString(&name) || parser.parseColonType(type))
      return failure();

    // Parse the default value if present.
    if (succeeded(parser.parseOptionalEqual())) {
      if (parser.parseAttribute(value, type))
        return failure();
    }

    auto &builder = parser.getBuilder();
    params.push_back(ParamDeclAttr::get(
        builder.getContext(), builder.getStringAttr(name), type, value));
    return success();
  };

  if (failed(parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::OptionalLessGreater, parseParameter)))
    return failure();

  parameters = ArrayAttr::get(parser.getContext(), params);

  return success();
}

void circt::printOptionalParameterList(OpAsmPrinter &p, Operation *op,
                                       ArrayAttr parameters) {
  if (parameters.empty())
    return;

  p << '<';
  llvm::interleaveComma(parameters, p, [&](Attribute param) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    p << paramAttr.getName().getValue() << ": " << paramAttr.getType();
    if (auto value = paramAttr.getValue()) {
      p << " = ";
      p.printAttributeWithoutType(value);
    }
  });
  p << '>';
}
