//===- ModuleImplementation.cpp - Utilities for module-like ops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/ParsingUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace circt;
using namespace circt::hw;

/// Parse a function result list.
///
///   function-result-list ::= function-result-list-parens
///   function-result-list-parens ::= `(` `)`
///                                 | `(` function-result-list-no-parens `)`
///   function-result-list-no-parens ::= function-result (`,` function-result)*
///   function-result ::= (percent-identifier `:`) type attribute-dict?
///
static ParseResult
parseFunctionResultList(OpAsmParser &parser,
                        SmallVectorImpl<Attribute> &resultNames,
                        SmallVectorImpl<Type> &resultTypes,
                        SmallVectorImpl<DictionaryAttr> &resultAttrs,
                        SmallVectorImpl<Attribute> &resultLocs) {

  auto parseElt = [&]() -> ParseResult {
    // Stash the current location parser location.
    auto irLoc = parser.getCurrentLocation();

    // Parse the result name.
    std::string portName;
    if (parser.parseKeywordOrString(&portName))
      return failure();
    resultNames.push_back(StringAttr::get(parser.getContext(), portName));

    // Parse the results type.
    resultTypes.emplace_back();
    if (parser.parseColonType(resultTypes.back()))
      return failure();

    // Parse the result attributes.
    NamedAttrList attrs;
    if (failed(parser.parseOptionalAttrDict(attrs)))
      return failure();
    resultAttrs.push_back(attrs.getDictionary(parser.getContext()));

    // Parse the result location.
    std::optional<Location> maybeLoc;
    if (failed(parser.parseOptionalLocationSpecifier(maybeLoc)))
      return failure();
    Location loc = maybeLoc ? *maybeLoc : parser.getEncodedSourceLoc(irLoc);
    resultLocs.push_back(loc);

    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseElt);
}

void module_like_impl::printModuleSignature(OpAsmPrinter &p, Operation *op,
                                            ArrayRef<Type> argTypes,
                                            bool isVariadic,
                                            ArrayRef<Type> resultTypes,
                                            bool &needArgNamesAttr) {
  using namespace mlir::function_interface_impl;

  Region &body = op->getRegion(0);
  bool isExternal = body.empty();
  SmallString<32> resultNameStr;
  mlir::OpPrintingFlags flags;

  auto funcOp = cast<mlir::FunctionOpInterface>(op);

  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    auto argName = getModuleArgumentName(op, i);

    if (!isExternal) {
      // Get the printed format for the argument name.
      resultNameStr.clear();
      llvm::raw_svector_ostream tmpStream(resultNameStr);
      p.printOperand(body.front().getArgument(i), tmpStream);

      // If the name wasn't printable in a way that agreed with argName, make
      // sure to print out an explicit argNames attribute.
      if (tmpStream.str().drop_front() != argName)
        needArgNamesAttr = true;

      p << tmpStream.str() << ": ";
    } else if (!argName.empty()) {
      p << '%' << argName << ": ";
    }

    p.printType(argTypes[i]);
    p.printOptionalAttrDict(getArgAttrs(funcOp, i));

    // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
    // even if they are not printed.  This will have to be fixed upstream.  For
    // now, use what was specified on the command line.
    if (flags.shouldPrintDebugInfo())
      if (auto loc = getModuleArgumentLocAttr(op, i))
        p.printOptionalLocationSpecifier(loc);
  }

  if (isVariadic) {
    if (!argTypes.empty())
      p << ", ";
    p << "...";
  }

  p << ')';

  // We print result types specially since we support named arguments.
  if (!resultTypes.empty()) {
    p << " -> (";
    for (size_t i = 0, e = resultTypes.size(); i < e; ++i) {
      if (i != 0)
        p << ", ";
      p.printKeywordOrString(getModuleResultNameAttr(op, i).getValue());
      p << ": ";
      p.printType(resultTypes[i]);
      p.printOptionalAttrDict(getResultAttrs(funcOp, i));

      // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
      // even if they are not printed.  This will have to be fixed upstream. For
      // now, use what was specified on the command line.
      if (flags.shouldPrintDebugInfo())
        if (auto loc = getModuleResultLocAttr(op, i))
          p.printOptionalLocationSpecifier(loc);
    }
    p << ')';
  }
}

ParseResult module_like_impl::parseModuleFunctionSignature(
    OpAsmParser &parser, bool &isVariadic,
    SmallVectorImpl<OpAsmParser::Argument> &args,
    SmallVectorImpl<Attribute> &argNames, SmallVectorImpl<Attribute> &argLocs,
    SmallVectorImpl<Attribute> &resultNames,
    SmallVectorImpl<DictionaryAttr> &resultAttrs,
    SmallVectorImpl<Attribute> &resultLocs, TypeAttr &type) {

  using namespace mlir::function_interface_impl;
  auto *context = parser.getContext();

  // Parse the argument list.
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true))
    return failure();

  // Parse the result list.
  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow()))
    if (failed(parseFunctionResultList(parser, resultNames, resultTypes,
                                       resultAttrs, resultLocs)))
      return failure();

  // Process the ssa args for the information we're looking for.
  SmallVector<Type> argTypes;
  for (auto &arg : args) {
    argNames.push_back(parsing_util::getNameFromSSA(context, arg.ssaName.name));
    argTypes.push_back(arg.type);
    if (!arg.sourceLoc)
      arg.sourceLoc = parser.getEncodedSourceLoc(arg.ssaName.location);
    argLocs.push_back(*arg.sourceLoc);
  }

  type = TypeAttr::get(FunctionType::get(context, argTypes, resultTypes));

  return success();
}

////////////////////////////////////////////////////////////////////////////////
// New Style
////////////////////////////////////////////////////////////////////////////////

static ParseResult parseDirection(OpAsmParser &p, ModulePort::Direction &dir) {
  StringRef key;
  if (failed(p.parseKeyword(&key)))
    return p.emitError(p.getCurrentLocation(), "expected port direction");
  if (key == "input")
    dir = ModulePort::Direction::Input;
  else if (key == "output")
    dir = ModulePort::Direction::Output;
  else if (key == "inout")
    dir = ModulePort::Direction::InOut;
  else
    return p.emitError(p.getCurrentLocation(), "unknown port direction '")
           << key << "'";
  return success();
}

/// Parse a single argument with the following syntax:
///
///   direction `%ssaname : !type { optionalAttrDict} loc(optionalSourceLoc)`
///
/// If `allowType` is false or `allowAttrs` are false then the respective
/// parts of the grammar are not parsed.
static ParseResult parsePort(OpAsmParser &p,
                             module_like_impl::PortParse &result) {
  NamedAttrList attrs;
  if (parseDirection(p, result.direction) ||
      p.parseOperand(result.ssaName, /*allowResultNumber=*/false) ||
      p.parseColonType(result.type) || p.parseOptionalAttrDict(attrs) ||
      p.parseOptionalLocationSpecifier(result.sourceLoc))
    return failure();
  result.attrs = attrs.getDictionary(p.getContext());
  return success();
}

static ParseResult
parsePortList(OpAsmParser &p,
              SmallVectorImpl<module_like_impl::PortParse> &result) {
  auto parseOnePort = [&]() -> ParseResult {
    return parsePort(p, result.emplace_back());
  };
  return p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseOnePort,
                                   " in port list");
}

ParseResult module_like_impl::parseModuleSignature(
    OpAsmParser &parser, SmallVectorImpl<PortParse> &args, TypeAttr &modType) {

  auto *context = parser.getContext();

  // Parse the port list.
  if (parsePortList(parser, args))
    return failure();

  // Process the ssa args for the information we're looking for.
  SmallVector<ModulePort> ports;
  for (auto &arg : args) {
    ports.push_back({parsing_util::getNameFromSSA(context, arg.ssaName.name),
                     arg.type, arg.direction});
    if (!arg.sourceLoc)
      arg.sourceLoc = parser.getEncodedSourceLoc(arg.ssaName.location);
  }
  modType = TypeAttr::get(ModuleType::get(context, ports));

  return success();
}

static const char *directionAsString(ModulePort::Direction dir) {
  if (dir == ModulePort::Direction::Input)
    return "input";
  if (dir == ModulePort::Direction::Output)
    return "output";
  if (dir == ModulePort::Direction::InOut)
    return "inout";
  assert(0 && "Unknown port direction");
  abort();
  return "unknown";
}

void module_like_impl::printModuleSignatureNew(OpAsmPrinter &p, Operation *op) {

  mlir::OpPrintingFlags flags;

  auto typeAttr = op->getAttrOfType<TypeAttr>("module_type");
  auto modType = cast<ModuleType>(typeAttr.getValue());
  auto portAttrs = op->getAttrOfType<ArrayAttr>("port_attrs");
  auto locAttrs = op->getAttrOfType<ArrayAttr>("port_locs");

  p << '(';
  for (auto [i, port] : llvm::enumerate(modType.getPorts())) {
    if (i > 0)
      p << ", ";
    p.printKeywordOrString(directionAsString(port.dir));
    p << " %";
    p.printKeywordOrString(port.name);
    p << " : ";
    p.printType(port.type);
    if (auto attr = dyn_cast<DictionaryAttr>(portAttrs[i]))
      p.printOptionalAttrDict(attr.getValue());

    // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
    // even if they are not printed.  This will have to be fixed upstream.  For
    // now, use what was specified on the command line.
    if (flags.shouldPrintDebugInfo())
      if (auto loc = locAttrs[i])
        p.printOptionalLocationSpecifier(cast<Location>(loc));
  }

  p << ')';
}
