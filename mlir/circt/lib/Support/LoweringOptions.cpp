//===- LoweringOptions.cpp - CIRCT Lowering Options -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Options for controlling the lowering process. Contains command line
// option definitions and support.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;
using namespace mlir;

//===----------------------------------------------------------------------===//
// LoweringOptions
//===----------------------------------------------------------------------===//

LoweringOptions::LoweringOptions(StringRef options, ErrorHandlerT errorHandler)
    : LoweringOptions() {
  parse(options, errorHandler);
}

LoweringOptions::LoweringOptions(mlir::ModuleOp module) : LoweringOptions() {
  parseFromAttribute(module);
}

static std::optional<LoweringOptions::LocationInfoStyle>
parseLocationInfoStyle(StringRef option) {
  return llvm::StringSwitch<std::optional<LoweringOptions::LocationInfoStyle>>(
             option)
      .Case("plain", LoweringOptions::Plain)
      .Case("wrapInAtSquareBracket", LoweringOptions::WrapInAtSquareBracket)
      .Case("none", LoweringOptions::None)
      .Default(std::nullopt);
}

static std::optional<LoweringOptions::WireSpillingHeuristic>
parseWireSpillingHeuristic(StringRef option) {
  return llvm::StringSwitch<
             std::optional<LoweringOptions::WireSpillingHeuristic>>(option)
      .Case("spillLargeTermsWithNamehints",
            LoweringOptions::SpillLargeTermsWithNamehints)
      .Default(std::nullopt);
}

void LoweringOptions::parse(StringRef text, ErrorHandlerT errorHandler) {
  while (!text.empty()) {
    // Remove the first option from the text.
    auto split = text.split(",");
    auto option = split.first.trim();
    text = split.second;
    if (option == "") {
      // Empty options are fine.
    } else if (option == "noAlwaysComb") {
      noAlwaysComb = true;
    } else if (option == "exprInEventControl") {
      allowExprInEventControl = true;
    } else if (option == "disallowPackedArrays") {
      disallowPackedArrays = true;
    } else if (option == "disallowPackedStructAssignments") {
      disallowPackedStructAssignments = true;
    } else if (option == "disallowLocalVariables") {
      disallowLocalVariables = true;
    } else if (option == "verifLabels") {
      enforceVerifLabels = true;
    } else if (option.consume_front("emittedLineLength=")) {
      if (option.getAsInteger(10, emittedLineLength)) {
        errorHandler("expected integer source width");
        emittedLineLength = DEFAULT_LINE_LENGTH;
      }
    } else if (option == "explicitBitcast") {
      explicitBitcast = true;
    } else if (option == "emitReplicatedOpsToHeader") {
      emitReplicatedOpsToHeader = true;
    } else if (option.consume_front("maximumNumberOfTermsPerExpression=")) {
      if (option.getAsInteger(10, maximumNumberOfTermsPerExpression)) {
        errorHandler("expected integer source width");
        maximumNumberOfTermsPerExpression = DEFAULT_TERM_LIMIT;
      }
    } else if (option.consume_front("locationInfoStyle=")) {
      if (auto style = parseLocationInfoStyle(option)) {
        locationInfoStyle = *style;
      } else {
        errorHandler("expected 'plain', 'wrapInAtSquareBracket', or 'none'");
      }
    } else if (option == "disallowPortDeclSharing") {
      disallowPortDeclSharing = true;
    } else if (option == "printDebugInfo") {
      printDebugInfo = true;
    } else if (option == "disallowExpressionInliningInPorts") {
      disallowExpressionInliningInPorts = true;
    } else if (option == "disallowMuxInlining") {
      disallowMuxInlining = true;
    } else if (option == "mitigateVivadoArrayIndexConstPropBug") {
      mitigateVivadoArrayIndexConstPropBug = true;
    } else if (option.consume_front("wireSpillingHeuristic=")) {
      if (auto heuristic = parseWireSpillingHeuristic(option)) {
        wireSpillingHeuristicSet |= *heuristic;
      } else {
        errorHandler("expected ''spillLargeTermsWithNamehints'");
      }
    } else if (option.consume_front("wireSpillingNamehintTermLimit=")) {
      if (option.getAsInteger(10, wireSpillingNamehintTermLimit)) {
        errorHandler(
            "expected integer for number of namehint heurstic term limit");
        wireSpillingNamehintTermLimit = DEFAULT_NAMEHINT_TERM_LIMIT;
      }
    } else if (option == "emitWireInPorts") {
      emitWireInPorts = true;
    } else if (option == "emitBindComments") {
      emitBindComments = true;
    } else if (option == "omitVersionComment") {
      omitVersionComment = true;
    } else {
      errorHandler(llvm::Twine("unknown style option \'") + option + "\'");
      // We continue parsing options after a failure.
    }
  }
}

std::string LoweringOptions::toString() const {
  std::string options = "";
  // All options should add a trailing comma to simplify the code.
  if (noAlwaysComb)
    options += "noAlwaysComb,";
  if (allowExprInEventControl)
    options += "exprInEventControl,";
  if (disallowPackedArrays)
    options += "disallowPackedArrays,";
  if (disallowPackedStructAssignments)
    options += "disallowPackedStructAssignments,";
  if (disallowLocalVariables)
    options += "disallowLocalVariables,";
  if (enforceVerifLabels)
    options += "verifLabels,";
  if (explicitBitcast)
    options += "explicitBitcast,";
  if (emitReplicatedOpsToHeader)
    options += "emitReplicatedOpsToHeader,";
  if (locationInfoStyle == LocationInfoStyle::WrapInAtSquareBracket)
    options += "locationInfoStyle=wrapInAtSquareBracket,";
  if (locationInfoStyle == LocationInfoStyle::None)
    options += "locationInfoStyle=none,";
  if (disallowPortDeclSharing)
    options += "disallowPortDeclSharing,";
  if (printDebugInfo)
    options += "printDebugInfo,";
  if (isWireSpillingHeuristicEnabled(
          WireSpillingHeuristic::SpillLargeTermsWithNamehints))
    options += "wireSpillingHeuristic=spillLargeTermsWithNamehints,";
  if (disallowExpressionInliningInPorts)
    options += "disallowExpressionInliningInPorts,";
  if (disallowMuxInlining)
    options += "disallowMuxInlining,";
  if (mitigateVivadoArrayIndexConstPropBug)
    options += "mitigateVivadoArrayIndexConstPropBug,";

  if (emittedLineLength != DEFAULT_LINE_LENGTH)
    options += "emittedLineLength=" + std::to_string(emittedLineLength) + ',';
  if (maximumNumberOfTermsPerExpression != DEFAULT_TERM_LIMIT)
    options += "maximumNumberOfTermsPerExpression=" +
               std::to_string(maximumNumberOfTermsPerExpression) + ',';
  if (emitWireInPorts)
    options += "emitWireInPorts,";
  if (emitBindComments)
    options += "emitBindComments,";
  if (omitVersionComment)
    options += "omitVersionComment,";

  // Remove a trailing comma if present.
  if (!options.empty()) {
    assert(options.back() == ',' && "all options should add a trailing comma");
    options.pop_back();
  }
  return options;
}

StringAttr LoweringOptions::getAttributeFrom(ModuleOp module) {
  return module->getAttrOfType<StringAttr>("circt.loweringOptions");
}

void LoweringOptions::setAsAttribute(ModuleOp module) {
  module->setAttr("circt.loweringOptions",
                  StringAttr::get(module.getContext(), toString()));
}

void LoweringOptions::parseFromAttribute(ModuleOp module) {
  if (auto styleAttr = getAttributeFrom(module))
    parse(styleAttr.getValue(), [&](Twine error) { module.emitError(error); });
}
