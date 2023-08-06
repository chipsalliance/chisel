//===- HandshakeOps.cpp - Handshake MLIR Operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Handshake operations struct.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <set>

using namespace circt;
using namespace circt::handshake;

namespace circt {
namespace handshake {
#include "circt/Dialect/Handshake/HandshakeCanonicalization.h.inc"

bool isControlOpImpl(Operation *op) {
  if (auto sostInterface = dyn_cast<SOSTInterface>(op); sostInterface)
    return sostInterface.sostIsControl();

  return false;
}

} // namespace handshake
} // namespace circt

static std::string defaultOperandName(unsigned int idx) {
  return "in" + std::to_string(idx);
}

static ParseResult parseIntInSquareBrackets(OpAsmParser &parser, int &v) {
  if (parser.parseLSquare() || parser.parseInteger(v) || parser.parseRSquare())
    return failure();
  return success();
}

static ParseResult
parseSostOperation(OpAsmParser &parser,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                   OperationState &result, int &size, Type &type,
                   bool explicitSize) {
  if (explicitSize)
    if (parseIntInSquareBrackets(parser, size))
      return failure();

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  if (!explicitSize)
    size = operands.size();
  return success();
}

/// Verifies whether an indexing value is wide enough to index into a provided
/// number of operands.
static LogicalResult verifyIndexWideEnough(Operation *op, Value indexVal,
                                           uint64_t numOperands) {
  auto indexType = indexVal.getType();
  unsigned indexWidth;

  // Determine the bitwidth of the indexing value
  if (auto integerType = indexType.dyn_cast<IntegerType>())
    indexWidth = integerType.getWidth();
  else if (indexType.isIndex())
    indexWidth = IndexType::kInternalStorageBitWidth;
  else
    return op->emitError("unsupported type for indexing value: ") << indexType;

  // Check whether the bitwidth can support the provided number of operands
  if (indexWidth < 64) {
    uint64_t maxNumOperands = (uint64_t)1 << indexWidth;
    if (numOperands > maxNumOperands)
      return op->emitError("bitwidth of indexing value is ")
             << indexWidth << ", which can index into " << maxNumOperands
             << " operands, but found " << numOperands << " operands";
  }
  return success();
}

static bool isControlCheckTypeAndOperand(Type dataType, Value operand) {
  // The operation is a control operation if its operand data type is a
  // NoneType.
  if (dataType.isa<NoneType>())
    return true;

  // Otherwise, the operation is a control operation if the operation's
  // operand originates from the control network
  auto *defOp = operand.getDefiningOp();
  return isa_and_nonnull<ControlMergeOp>(defOp) &&
         operand == defOp->getResult(0);
}

template <typename TMemOp>
llvm::SmallVector<handshake::MemLoadInterface> getLoadPorts(TMemOp op) {
  llvm::SmallVector<MemLoadInterface> ports;
  // Memory interface refresher:
  // Operands:
  //   all stores (stdata1, staddr1, stdata2, staddr2, ...)
  //   then all loads (ldaddr1, ldaddr2,...)
  // Outputs: load addresses (lddata1, lddata2, ...), followed by all none
  // outputs, ordered as operands(stnone1, stnone2, ... ldnone1, ldnone2, ...)
  unsigned stCount = op.getStCount();
  unsigned ldCount = op.getLdCount();
  for (unsigned i = 0, e = ldCount; i != e; ++i) {
    MemLoadInterface ldif;
    ldif.index = i;
    ldif.addressIn = op.getInputs()[stCount * 2 + i];
    ldif.dataOut = op.getResult(i);
    ldif.doneOut = op.getResult(ldCount + stCount + i);
    ports.push_back(ldif);
  }
  return ports;
}

template <typename TMemOp>
llvm::SmallVector<handshake::MemStoreInterface> getStorePorts(TMemOp op) {
  llvm::SmallVector<MemStoreInterface> ports;
  // Memory interface refresher:
  // Operands:
  //   all stores (stdata1, staddr1, stdata2, staddr2, ...)
  //   then all loads (ldaddr1, ldaddr2,...)
  // Outputs: load data (lddata1, lddata2, ...), followed by all none
  // outputs, ordered as operands(stnone1, stnone2, ... ldnone1, ldnone2, ...)
  unsigned ldCount = op.getLdCount();
  for (unsigned i = 0, e = op.getStCount(); i != e; ++i) {
    MemStoreInterface stif;
    stif.index = i;
    stif.dataIn = op.getInputs()[i * 2];
    stif.addressIn = op.getInputs()[i * 2 + 1];
    stif.doneOut = op.getResult(ldCount + i);
    ports.push_back(stif);
  }
  return ports;
}

unsigned ForkOp::getSize() { return getResults().size(); }

static ParseResult parseForkOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  SmallVector<Type, 1> resultTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (parseSostOperation(parser, allOperands, result, size, type, true))
    return failure();

  resultTypes.assign(size, type);
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

ParseResult ForkOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForkOp(parser, result);
}

void ForkOp::print(OpAsmPrinter &p) { sostPrint(p, true); }

namespace {

struct EliminateUnusedForkResultsPattern : mlir::OpRewritePattern<ForkOp> {
  using mlir::OpRewritePattern<ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForkOp op,
                                PatternRewriter &rewriter) const override {
    std::set<unsigned> unusedIndexes;

    for (auto res : llvm::enumerate(op.getResults()))
      if (res.value().getUses().empty())
        unusedIndexes.insert(res.index());

    if (unusedIndexes.empty())
      return failure();

    // Create a new fork op, dropping the unused results.
    rewriter.setInsertionPoint(op);
    auto operand = op.getOperand();
    auto newFork = rewriter.create<ForkOp>(
        op.getLoc(), operand, op.getNumResults() - unusedIndexes.size());
    rewriter.updateRootInPlace(op, [&] {
      unsigned i = 0;
      for (auto oldRes : llvm::enumerate(op.getResults()))
        if (unusedIndexes.count(oldRes.index()) == 0)
          oldRes.value().replaceAllUsesWith(newFork.getResults()[i++]);
    });
    rewriter.eraseOp(op);
    return success();
  }
};

struct EliminateForkToForkPattern : mlir::OpRewritePattern<ForkOp> {
  using mlir::OpRewritePattern<ForkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForkOp op,
                                PatternRewriter &rewriter) const override {
    auto parentForkOp = op.getOperand().getDefiningOp<ForkOp>();
    if (!parentForkOp)
      return failure();

    /// Create the fork with as many outputs as the two source forks.
    /// Keeping the op.operand() output may or may not be redundant (dependning
    /// on if op is the single user of the value), but we'll let
    /// EliminateUnusedForkResultsPattern apply in that case.
    unsigned totalNumOuts = op.getSize() + parentForkOp.getSize();
    rewriter.updateRootInPlace(parentForkOp, [&] {
      /// Create a new parent fork op which produces all of the fork outputs and
      /// replace all of the uses of the old results.
      auto newParentForkOp = rewriter.create<ForkOp>(
          parentForkOp.getLoc(), parentForkOp.getOperand(), totalNumOuts);

      for (auto it :
           llvm::zip(parentForkOp->getResults(), newParentForkOp.getResults()))
        std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

      /// Replace the results of the matches fork op with the corresponding
      /// results of the new parent fork op.
      rewriter.replaceOp(op,
                         newParentForkOp.getResults().take_back(op.getSize()));
    });
    rewriter.eraseOp(parentForkOp);
    return success();
  }
};

} // namespace

void handshake::ForkOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleForksPattern,
                 EliminateForkToForkPattern, EliminateUnusedForkResultsPattern>(
      context);
}

unsigned LazyForkOp::getSize() { return getResults().size(); }

bool LazyForkOp::sostIsControl() {
  return isControlCheckTypeAndOperand(getDataType(), getOperand());
}

ParseResult LazyForkOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseForkOp(parser, result);
}

void LazyForkOp::print(OpAsmPrinter &p) { sostPrint(p, true); }

ParseResult MergeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  SmallVector<Type, 1> resultTypes, dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (parseSostOperation(parser, allOperands, result, size, type, false))
    return failure();

  dataOperandsTypes.assign(size, type);
  resultTypes.push_back(type);
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void MergeOp::print(OpAsmPrinter &p) { sostPrint(p, false); }

void MergeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleMergesPattern>(context);
}

/// Returns a dematerialized version of the value 'v', defined as the source of
/// the value before passing through a buffer or fork operation.
static Value getDematerialized(Value v) {
  Operation *parentOp = v.getDefiningOp();
  if (!parentOp)
    return v;

  return llvm::TypeSwitch<Operation *, Value>(parentOp)
      .Case<ForkOp>(
          [&](ForkOp op) { return getDematerialized(op.getOperand()); })
      .Case<BufferOp>(
          [&](BufferOp op) { return getDematerialized(op.getOperand()); })
      .Default([&](auto) { return v; });
}

namespace {

/// Eliminates muxes with identical data inputs. Data inputs are inspected as
/// their dematerialized versions. This has the side effect of any subsequently
/// unused buffers are DCE'd and forks are optimized to be narrower.
struct EliminateSimpleMuxesPattern : mlir::OpRewritePattern<MuxOp> {
  using mlir::OpRewritePattern<MuxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MuxOp op,
                                PatternRewriter &rewriter) const override {
    Value firstDataOperand = getDematerialized(op.getDataOperands()[0]);
    if (!llvm::all_of(op.getDataOperands(), [&](Value operand) {
          return getDematerialized(operand) == firstDataOperand;
        }))
      return failure();
    rewriter.replaceOp(op, firstDataOperand);
    return success();
  }
};

struct EliminateUnaryMuxesPattern : OpRewritePattern<MuxOp> {
  using mlir::OpRewritePattern<MuxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MuxOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getDataOperands().size() != 1)
      return failure();

    rewriter.replaceOp(op, op.getDataOperands()[0]);
    return success();
  }
};

struct EliminateCBranchIntoMuxPattern : OpRewritePattern<MuxOp> {
  using mlir::OpRewritePattern<MuxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MuxOp op,
                                PatternRewriter &rewriter) const override {

    auto dataOperands = op.getDataOperands();
    if (dataOperands.size() != 2)
      return failure();

    // Both data operands must originate from the same cbranch
    ConditionalBranchOp firstParentCBranch =
        dataOperands[0].getDefiningOp<ConditionalBranchOp>();
    if (!firstParentCBranch)
      return failure();
    auto secondParentCBranch =
        dataOperands[1].getDefiningOp<ConditionalBranchOp>();
    if (!secondParentCBranch || firstParentCBranch != secondParentCBranch)
      return failure();

    rewriter.updateRootInPlace(firstParentCBranch, [&] {
      // Replace uses of the mux's output with cbranch's data input
      rewriter.replaceOp(op, firstParentCBranch.getDataOperand());
    });

    return success();
  }
};

} // namespace

void MuxOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<EliminateSimpleMuxesPattern, EliminateUnaryMuxesPattern,
                 EliminateCBranchIntoMuxPattern>(context);
}

LogicalResult
MuxOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        mlir::OpaqueProperties properties,
                        mlir::RegionRange regions,
                        SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  // MuxOp must have at least one data operand (in addition to the select
  // operand)
  if (operands.size() < 2)
    return failure();
  // Result type is type of any data operand
  inferredReturnTypes.push_back(operands[1].getType());
  return success();
}

bool MuxOp::isControl() { return getResult().getType().isa<NoneType>(); }

std::string handshake::MuxOp::getOperandName(unsigned int idx) {
  return idx == 0 ? "select" : defaultOperandName(idx - 1);
}

ParseResult MuxOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand selectOperand;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type selectType, dataType;
  SmallVector<Type, 1> dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(selectOperand) || parser.parseLSquare() ||
      parser.parseOperandList(allOperands) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(selectType) || parser.parseComma() ||
      parser.parseType(dataType))
    return failure();

  int size = allOperands.size();
  dataOperandsTypes.assign(size, dataType);
  result.addTypes(dataType);
  allOperands.insert(allOperands.begin(), selectOperand);
  if (parser.resolveOperands(
          allOperands,
          llvm::concat<const Type>(ArrayRef<Type>(selectType),
                                   ArrayRef<Type>(dataOperandsTypes)),
          allOperandLoc, result.operands))
    return failure();
  return success();
}

void MuxOp::print(OpAsmPrinter &p) {
  Type selectType = getSelectOperand().getType();
  auto ops = getOperands();
  p << ' ' << ops.front();
  p << " [";
  p.printOperands(ops.drop_front());
  p << "]";
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << selectType << ", " << getResult().getType();
}

LogicalResult MuxOp::verify() {
  return verifyIndexWideEnough(*this, getSelectOperand(),
                               getDataOperands().size());
}

std::string handshake::ControlMergeOp::getResultName(unsigned int idx) {
  assert(idx == 0 || idx == 1);
  return idx == 0 ? "dataOut" : "index";
}

ParseResult ControlMergeOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type resultType, indexType;
  SmallVector<Type> resultTypes, dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (parseSostOperation(parser, allOperands, result, size, resultType, false))
    return failure();
  // Parse type of index result
  if (parser.parseComma() || parser.parseType(indexType))
    return failure();

  dataOperandsTypes.assign(size, resultType);
  resultTypes.push_back(resultType);
  resultTypes.push_back(indexType);
  result.addTypes(resultTypes);
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void ControlMergeOp::print(OpAsmPrinter &p) {
  sostPrint(p, false);
  // Print type of index result
  p << ", " << getIndex().getType();
}

LogicalResult ControlMergeOp::verify() {
  auto operands = getOperands();
  if (operands.empty())
    return emitOpError("operation must have at least one operand");
  if (operands[0].getType() != getResult().getType())
    return emitOpError("type of first result should match type of operands");
  return verifyIndexWideEnough(*this, getIndex(), getNumOperands());
}

LogicalResult FuncOp::verify() {
  // If this function is external there is nothing to do.
  if (isExternal())
    return success();

  // Verify that the argument list of the function and the arg list of the
  // entry block line up.  The trait already verified that the number of
  // arguments is the same between the signature and the block.
  auto fnInputTypes = getArgumentTypes();
  Block &entryBlock = front();

  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  // Verify that we have a name for each argument and result of this function.
  auto verifyPortNameAttr = [&](StringRef attrName,
                                unsigned numIOs) -> LogicalResult {
    auto portNamesAttr = (*this)->getAttrOfType<ArrayAttr>(attrName);

    if (!portNamesAttr)
      return emitOpError() << "expected attribute '" << attrName << "'.";

    auto portNames = portNamesAttr.getValue();
    if (portNames.size() != numIOs)
      return emitOpError() << "attribute '" << attrName << "' has "
                           << portNames.size()
                           << " entries but is expected to have " << numIOs
                           << ".";

    if (llvm::any_of(portNames,
                     [&](Attribute attr) { return !attr.isa<StringAttr>(); }))
      return emitOpError() << "expected all entries in attribute '" << attrName
                           << "' to be strings.";

    return success();
  };
  if (failed(verifyPortNameAttr("argNames", getNumArguments())))
    return failure();
  if (failed(verifyPortNameAttr("resNames", getNumResults())))
    return failure();

  // Verify that all memrefs have a corresponding extmemory operation
  for (auto arg : entryBlock.getArguments()) {
    if (!arg.getType().isa<MemRefType>())
      continue;
    if (arg.getUsers().empty() ||
        !isa<ExternalMemoryOp>(*arg.getUsers().begin()))
      return emitOpError("expected that block argument #")
             << arg.getArgNumber() << " is used by an 'extmemory' operation";
  }

  return success();
}

/// Parses a FuncOp signature using
/// mlir::function_interface_impl::parseFunctionSignature while getting access
/// to the parsed SSA names to store as attributes.
static ParseResult
parseFuncOpArgs(OpAsmParser &parser,
                SmallVectorImpl<OpAsmParser::Argument> &entryArgs,
                SmallVectorImpl<Type> &resTypes,
                SmallVectorImpl<DictionaryAttr> &resAttrs) {
  bool isVariadic;
  if (mlir::function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/true, entryArgs, isVariadic, resTypes,
          resAttrs)
          .failed())
    return failure();

  return success();
}

/// Generates names for a handshake.func input and output arguments, based on
/// the number of args as well as a prefix.
static SmallVector<Attribute> getFuncOpNames(Builder &builder, unsigned cnt,
                                             StringRef prefix) {
  SmallVector<Attribute> resNames;
  for (unsigned i = 0; i < cnt; ++i)
    resNames.push_back(builder.getStringAttr(prefix + std::to_string(i)));
  return resNames;
}

void handshake::FuncOp::build(OpBuilder &builder, OperationState &state,
                              StringRef name, FunctionType type,
                              ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(FuncOp::getFunctionTypeAttrName(state.name),
                     TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());

  if (const auto *argNamesAttrIt = llvm::find_if(
          attrs, [&](auto attr) { return attr.getName() == "argNames"; });
      argNamesAttrIt == attrs.end())
    state.addAttribute("argNames", builder.getArrayAttr({}));

  if (llvm::find_if(attrs, [&](auto attr) {
        return attr.getName() == "resNames";
      }) == attrs.end())
    state.addAttribute("resNames", builder.getArrayAttr({}));

  state.addRegion();
}

/// Helper function for appending a string to an array attribute, and
/// rewriting the attribute back to the operation.
static void addStringToStringArrayAttr(Builder &builder, Operation *op,
                                       StringRef attrName, StringAttr str) {
  llvm::SmallVector<Attribute> attrs;
  llvm::copy(op->getAttrOfType<ArrayAttr>(attrName).getValue(),
             std::back_inserter(attrs));
  attrs.push_back(str);
  op->setAttr(attrName, builder.getArrayAttr(attrs));
}

void handshake::FuncOp::resolveArgAndResNames() {
  Builder builder(getContext());

  /// Generate a set of fallback names. These are used in case names are
  /// missing from the currently set arg- and res name attributes.
  auto fallbackArgNames = getFuncOpNames(builder, getNumArguments(), "in");
  auto fallbackResNames = getFuncOpNames(builder, getNumResults(), "out");
  auto argNames = getArgNames().getValue();
  auto resNames = getResNames().getValue();

  /// Use fallback names where actual names are missing.
  auto resolveNames = [&](auto &fallbackNames, auto &actualNames,
                          StringRef attrName) {
    for (auto fallbackName : llvm::enumerate(fallbackNames)) {
      if (actualNames.size() <= fallbackName.index())
        addStringToStringArrayAttr(
            builder, this->getOperation(), attrName,
            fallbackName.value().template cast<StringAttr>());
    }
  };
  resolveNames(fallbackArgNames, argNames, "argNames");
  resolveNames(fallbackResNames, resNames, "resNames");
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  StringAttr nameAttr;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> resTypes;
  SmallVector<DictionaryAttr> resAttributes;
  SmallVector<Attribute> argNames;

  // Parse visibility.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse signature
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parseFuncOpArgs(parser, args, resTypes, resAttributes))
    return failure();
  mlir::function_interface_impl::addArgAndResultAttrs(
      builder, result, args, resAttributes,
      handshake::FuncOp::getArgAttrsAttrName(result.name),
      handshake::FuncOp::getResAttrsAttrName(result.name));

  // Set function type
  SmallVector<Type> argTypes;
  for (auto arg : args)
    argTypes.push_back(arg.type);

  result.addAttribute(
      handshake::FuncOp::getFunctionTypeAttrName(result.name),
      TypeAttr::get(builder.getFunctionType(argTypes, resTypes)));

  // Determine the names of the arguments. If no SSA values are present, use
  // fallback names.
  bool noSSANames =
      llvm::any_of(args, [](auto arg) { return arg.ssaName.name.empty(); });
  if (noSSANames) {
    argNames = getFuncOpNames(builder, args.size(), "in");
  } else {
    llvm::transform(args, std::back_inserter(argNames), [&](auto arg) {
      return builder.getStringAttr(arg.ssaName.name.drop_front());
    });
  }

  // Parse attributes
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  // If argNames and resNames wasn't provided manually, infer argNames attribute
  // from the parsed SSA names and resNames from our naming convention.
  if (!result.attributes.get("argNames"))
    result.addAttribute("argNames", builder.getArrayAttr(argNames));
  if (!result.attributes.get("resNames")) {
    auto resNames = getFuncOpNames(builder, resTypes.size(), "out");
    result.addAttribute("resNames", builder.getArrayAttr(resNames));
  }

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto *body = result.addRegion();
  llvm::SMLoc loc = parser.getCurrentLocation();
  auto parseResult = parser.parseOptionalRegion(*body, args,
                                                /*enableNameShadowing=*/false);
  if (!parseResult.has_value())
    return success();

  if (failed(*parseResult))
    return failure();
  // Function body was parsed, make sure its not empty.
  if (body->empty())
    return parser.emitError(loc, "expected non-empty function body");

  // If a body was parsed, the arg and res names need to be resolved
  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/true, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

namespace {
struct EliminateSimpleControlMergesPattern
    : mlir::OpRewritePattern<ControlMergeOp> {
  using mlir::OpRewritePattern<ControlMergeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ControlMergeOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult EliminateSimpleControlMergesPattern::matchAndRewrite(
    ControlMergeOp op, PatternRewriter &rewriter) const {
  auto dataResult = op.getResult();
  auto choiceResult = op.getIndex();
  auto choiceUnused = choiceResult.use_empty();
  if (!choiceUnused && !choiceResult.hasOneUse())
    return failure();

  Operation *choiceUser = nullptr;
  if (choiceResult.hasOneUse()) {
    choiceUser = choiceResult.getUses().begin().getUser();
    if (!isa<SinkOp>(choiceUser))
      return failure();
  }

  auto merge = rewriter.create<MergeOp>(op.getLoc(), op.getDataOperands());

  for (auto &use : llvm::make_early_inc_range(dataResult.getUses())) {
    auto *user = use.getOwner();
    rewriter.updateRootInPlace(
        user, [&]() { user->setOperand(use.getOperandNumber(), merge); });
  }

  if (choiceUnused) {
    rewriter.eraseOp(op);
    return success();
  }

  rewriter.eraseOp(choiceUser);
  rewriter.eraseOp(op);
  return success();
}

void ControlMergeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<EliminateSimpleControlMergesPattern>(context);
}

bool BranchOp::sostIsControl() {
  return isControlCheckTypeAndOperand(getDataType(), getOperand());
}

void BranchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<circt::handshake::EliminateSimpleBranchesPattern>(context);
}

ParseResult BranchOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  SmallVector<Type, 1> dataOperandsTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (parseSostOperation(parser, allOperands, result, size, type, false))
    return failure();

  dataOperandsTypes.assign(size, type);
  result.addTypes({type});
  if (parser.resolveOperands(allOperands, dataOperandsTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void BranchOp::print(OpAsmPrinter &p) { sostPrint(p, false); }

ParseResult ConditionalBranchOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type dataType;
  SmallVector<Type> operandTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(dataType))
    return failure();

  if (allOperands.size() != 2)
    return parser.emitError(parser.getCurrentLocation(),
                            "Expected exactly 2 operands");

  result.addTypes({dataType, dataType});
  operandTypes.push_back(IntegerType::get(parser.getContext(), 1));
  operandTypes.push_back(dataType);
  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();

  return success();
}

void ConditionalBranchOp::print(OpAsmPrinter &p) {
  Type type = getDataOperand().getType();
  p << " " << getOperands();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << type;
}

std::string handshake::ConditionalBranchOp::getOperandName(unsigned int idx) {
  assert(idx == 0 || idx == 1);
  return idx == 0 ? "cond" : "data";
}

std::string handshake::ConditionalBranchOp::getResultName(unsigned int idx) {
  assert(idx == 0 || idx == 1);
  return idx == ConditionalBranchOp::falseIndex ? "outFalse" : "outTrue";
}

bool ConditionalBranchOp::isControl() {
  return isControlCheckTypeAndOperand(getDataOperand().getType(),
                                      getDataOperand());
}

ParseResult SinkOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int size;
  if (parseSostOperation(parser, allOperands, result, size, type, false))
    return failure();

  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void SinkOp::print(OpAsmPrinter &p) { sostPrint(p, false); }

std::string handshake::ConstantOp::getOperandName(unsigned int idx) {
  assert(idx == 0);
  return "ctrl";
}

Type SourceOp::getDataType() { return getResult().getType(); }
unsigned SourceOp::getSize() { return 1; }

ParseResult SourceOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addTypes(NoneType::get(result.getContext()));
  return success();
}

void SourceOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult ConstantOp::verify() {
  // Verify that the type of the provided value is equal to the result type.
  auto typedValue = getValue().dyn_cast<mlir::TypedAttr>();
  if (!typedValue)
    return emitOpError("constant value must be a typed attribute; value is ")
           << getValue();
  if (typedValue.getType() != getResult().getType())
    return emitOpError() << "constant value type " << typedValue.getType()
                         << " differs from operation result type "
                         << getResult().getType();

  return success();
}

void handshake::ConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<circt::handshake::EliminateSunkConstantsPattern>(context);
}

LogicalResult BufferOp::verify() {
  // Verify that exactly 'size' number of initial values have been provided, if
  // an initializer list have been provided.
  if (auto initVals = getInitValues()) {
    if (!isSequential())
      return emitOpError()
             << "only bufferType buffers are allowed to have initial values.";

    auto nInits = initVals->size();
    if (nInits != getSize())
      return emitOpError() << "expected " << getSize()
                           << " init values but got " << nInits << ".";
  }

  return success();
}

void handshake::BufferOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<circt::handshake::EliminateSunkBuffersPattern>(context);
}

unsigned BufferOp::getSize() {
  return (*this)->getAttrOfType<IntegerAttr>("slots").getValue().getZExtValue();
}

ParseResult BufferOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> allOperands;
  Type type;
  ArrayRef<Type> operandTypes(type);
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  int slots;
  if (parseIntInSquareBrackets(parser, slots))
    return failure();

  auto bufferTypeAttr = BufferTypeEnumAttr::parse(parser, {});
  if (!bufferTypeAttr)
    return failure();

  result.addAttribute(
      "slots",
      IntegerAttr::get(IntegerType::get(result.getContext(), 32), slots));
  result.addAttribute("bufferType", bufferTypeAttr);

  if (parser.parseOperandList(allOperands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  result.addTypes({type});
  if (parser.resolveOperands(allOperands, operandTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

void BufferOp::print(OpAsmPrinter &p) {
  int size =
      (*this)->getAttrOfType<IntegerAttr>("slots").getValue().getZExtValue();
  p << " [" << size << "]";
  p << " " << stringifyEnum(getBufferType());
  p << " " << (*this)->getOperands();
  p.printOptionalAttrDict((*this)->getAttrs(), {"slots", "bufferType"});
  p << " : " << (*this).getDataType();
}

static std::string getMemoryOperandName(unsigned nStores, unsigned idx) {
  std::string name;
  if (idx < nStores * 2) {
    bool isData = idx % 2 == 0;
    name = isData ? "stData" + std::to_string(idx / 2)
                  : "stAddr" + std::to_string(idx / 2);
  } else {
    idx -= 2 * nStores;
    name = "ldAddr" + std::to_string(idx);
  }
  return name;
}

std::string handshake::MemoryOp::getOperandName(unsigned int idx) {
  return getMemoryOperandName(getStCount(), idx);
}

static std::string getMemoryResultName(unsigned nLoads, unsigned nStores,
                                       unsigned idx) {
  std::string name;
  if (idx < nLoads)
    name = "ldData" + std::to_string(idx);
  else if (idx < nLoads + nStores)
    name = "stDone" + std::to_string(idx - nLoads);
  else
    name = "ldDone" + std::to_string(idx - nLoads - nStores);
  return name;
}

std::string handshake::MemoryOp::getResultName(unsigned int idx) {
  return getMemoryResultName(getLdCount(), getStCount(), idx);
}

LogicalResult MemoryOp::verify() {
  auto memrefType = getMemRefType();

  if (memrefType.getNumDynamicDims() != 0)
    return emitOpError()
           << "memref dimensions for handshake.memory must be static.";
  if (memrefType.getShape().size() != 1)
    return emitOpError() << "memref must have only a single dimension.";

  unsigned opStCount = getStCount();
  unsigned opLdCount = getLdCount();
  int addressCount = memrefType.getShape().size();

  auto inputType = getInputs().getType();
  auto outputType = getOutputs().getType();
  Type dataType = memrefType.getElementType();

  unsigned numOperands = static_cast<int>(getInputs().size());
  unsigned numResults = static_cast<int>(getOutputs().size());
  if (numOperands != (1 + addressCount) * opStCount + addressCount * opLdCount)
    return emitOpError("number of operands ")
           << numOperands << " does not match number expected of "
           << 2 * opStCount + opLdCount << " with " << addressCount
           << " address inputs per port";

  if (numResults != opStCount + 2 * opLdCount)
    return emitOpError("number of results ")
           << numResults << " does not match number expected of "
           << opStCount + 2 * opLdCount << " with " << addressCount
           << " address inputs per port";

  Type addressType = opStCount > 0 ? inputType[1] : inputType[0];

  for (unsigned i = 0; i < opStCount; i++) {
    if (inputType[2 * i] != dataType)
      return emitOpError("data type for store port ")
             << i << ":" << inputType[2 * i] << " doesn't match memory type "
             << dataType;
    if (inputType[2 * i + 1] != addressType)
      return emitOpError("address type for store port ")
             << i << ":" << inputType[2 * i + 1]
             << " doesn't match address type " << addressType;
  }
  for (unsigned i = 0; i < opLdCount; i++) {
    Type ldAddressType = inputType[2 * opStCount + i];
    if (ldAddressType != addressType)
      return emitOpError("address type for load port ")
             << i << ":" << ldAddressType << " doesn't match address type "
             << addressType;
  }
  for (unsigned i = 0; i < opLdCount; i++) {
    if (outputType[i] != dataType)
      return emitOpError("data type for load port ")
             << i << ":" << outputType[i] << " doesn't match memory type "
             << dataType;
  }
  for (unsigned i = 0; i < opStCount; i++) {
    Type syncType = outputType[opLdCount + i];
    if (!syncType.isa<NoneType>())
      return emitOpError("data type for sync port for store port ")
             << i << ":" << syncType << " is not 'none'";
  }
  for (unsigned i = 0; i < opLdCount; i++) {
    Type syncType = outputType[opLdCount + opStCount + i];
    if (!syncType.isa<NoneType>())
      return emitOpError("data type for sync port for load port ")
             << i << ":" << syncType << " is not 'none'";
  }

  return success();
}

std::string handshake::ExternalMemoryOp::getOperandName(unsigned int idx) {
  if (idx == 0)
    return "extmem";

  return getMemoryOperandName(getStCount(), idx - 1);
}

std::string handshake::ExternalMemoryOp::getResultName(unsigned int idx) {
  return getMemoryResultName(getLdCount(), getStCount(), idx);
}

void ExternalMemoryOp::build(OpBuilder &builder, OperationState &result,
                             Value memref, ValueRange inputs, int ldCount,
                             int stCount, int id) {
  SmallVector<Value> ops;
  ops.push_back(memref);
  llvm::append_range(ops, inputs);
  result.addOperands(ops);

  auto memrefType = memref.getType().cast<MemRefType>();

  // Data outputs (get their type from memref)
  result.types.append(ldCount, memrefType.getElementType());

  // Control outputs
  result.types.append(stCount + ldCount, builder.getNoneType());

  // Memory ID (individual ID for each MemoryOp)
  Type i32Type = builder.getIntegerType(32);
  result.addAttribute("id", builder.getIntegerAttr(i32Type, id));
  result.addAttribute("ldCount", builder.getIntegerAttr(i32Type, ldCount));
  result.addAttribute("stCount", builder.getIntegerAttr(i32Type, stCount));
}

llvm::SmallVector<handshake::MemLoadInterface>
ExternalMemoryOp::getLoadPorts() {
  return ::getLoadPorts(*this);
}

llvm::SmallVector<handshake::MemStoreInterface>
ExternalMemoryOp::getStorePorts() {
  return ::getStorePorts(*this);
}

void MemoryOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange operands, int outputs, int controlOutputs,
                     bool lsq, int id, Value memref) {
  result.addOperands(operands);

  auto memrefType = memref.getType().cast<MemRefType>();

  // Data outputs (get their type from memref)
  result.types.append(outputs, memrefType.getElementType());

  // Control outputs
  result.types.append(controlOutputs, builder.getNoneType());
  result.addAttribute("lsq", builder.getBoolAttr(lsq));
  result.addAttribute("memRefType", TypeAttr::get(memrefType));

  // Memory ID (individual ID for each MemoryOp)
  Type i32Type = builder.getIntegerType(32);
  result.addAttribute("id", builder.getIntegerAttr(i32Type, id));

  if (!lsq) {
    result.addAttribute("ldCount", builder.getIntegerAttr(i32Type, outputs));
    result.addAttribute(
        "stCount", builder.getIntegerAttr(i32Type, controlOutputs - outputs));
  }
}

llvm::SmallVector<handshake::MemLoadInterface> MemoryOp::getLoadPorts() {
  return ::getLoadPorts(*this);
}

llvm::SmallVector<handshake::MemStoreInterface> MemoryOp::getStorePorts() {
  return ::getStorePorts(*this);
}

bool handshake::MemoryOp::allocateMemory(
    llvm::DenseMap<unsigned, unsigned> &memoryMap,
    std::vector<std::vector<llvm::Any>> &store,
    std::vector<double> &storeTimes) {
  if (memoryMap.count(getId()))
    return false;

  auto type = getMemRefType();
  std::vector<llvm::Any> in;

  ArrayRef<int64_t> shape = type.getShape();
  int allocationSize = 1;
  unsigned count = 0;
  for (int64_t dim : shape) {
    if (dim > 0)
      allocationSize *= dim;
    else {
      assert(count < in.size());
      allocationSize *= llvm::any_cast<APInt>(in[count++]).getSExtValue();
    }
  }
  unsigned ptr = store.size();
  store.resize(ptr + 1);
  storeTimes.resize(ptr + 1);
  store[ptr].resize(allocationSize);
  storeTimes[ptr] = 0.0;
  mlir::Type elementType = type.getElementType();
  int width = elementType.getIntOrFloatBitWidth();
  for (int i = 0; i < allocationSize; i++) {
    if (elementType.isa<mlir::IntegerType>()) {
      store[ptr][i] = APInt(width, 0);
    } else if (elementType.isa<mlir::FloatType>()) {
      store[ptr][i] = APFloat(0.0);
    } else {
      llvm_unreachable("Unknown result type!\n");
    }
  }

  memoryMap[getId()] = ptr;
  return true;
}

std::string handshake::LoadOp::getOperandName(unsigned int idx) {
  unsigned nAddresses = getAddresses().size();
  std::string opName;
  if (idx < nAddresses)
    opName = "addrIn" + std::to_string(idx);
  else if (idx == nAddresses)
    opName = "dataFromMem";
  else
    opName = "ctrl";
  return opName;
}

std::string handshake::LoadOp::getResultName(unsigned int idx) {
  std::string resName;
  if (idx == 0)
    resName = "dataOut";
  else
    resName = "addrOut" + std::to_string(idx - 1);
  return resName;
}

void handshake::LoadOp::build(OpBuilder &builder, OperationState &result,
                              Value memref, ValueRange indices) {
  // Address indices
  // result.addOperands(memref);
  result.addOperands(indices);

  // Data type
  auto memrefType = memref.getType().cast<MemRefType>();

  // Data output (from load to successor ops)
  result.types.push_back(memrefType.getElementType());

  // Address outputs (to lsq)
  result.types.append(indices.size(), builder.getIndexType());
}

static ParseResult parseMemoryAccessOp(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> addressOperands,
      remainingOperands, allOperands;
  SmallVector<Type, 1> parsedTypes, allTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();

  if (parser.parseLSquare() || parser.parseOperandList(addressOperands) ||
      parser.parseRSquare() || parser.parseOperandList(remainingOperands) ||
      parser.parseColon() || parser.parseTypeList(parsedTypes))
    return failure();

  // The last type will be the data type of the operation; the prior will be the
  // address types.
  Type dataType = parsedTypes.back();
  auto parsedTypesRef = ArrayRef(parsedTypes);
  result.addTypes(dataType);
  result.addTypes(parsedTypesRef.drop_back());
  allOperands.append(addressOperands);
  allOperands.append(remainingOperands);
  allTypes.append(parsedTypes);
  allTypes.push_back(NoneType::get(result.getContext()));
  if (parser.resolveOperands(allOperands, allTypes, allOperandLoc,
                             result.operands))
    return failure();
  return success();
}

template <typename MemOp>
static void printMemoryAccessOp(OpAsmPrinter &p, MemOp op) {
  p << " [";
  p << op.getAddresses();
  p << "] " << op.getData() << ", " << op.getCtrl() << " : ";
  llvm::interleaveComma(op.getAddresses(), p,
                        [&](Value v) { p << v.getType(); });
  p << ", " << op.getData().getType();
}

ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMemoryAccessOp(parser, result);
}

void LoadOp::print(OpAsmPrinter &p) { printMemoryAccessOp(p, *this); }

std::string handshake::StoreOp::getOperandName(unsigned int idx) {
  unsigned nAddresses = getAddresses().size();
  std::string opName;
  if (idx < nAddresses)
    opName = "addrIn" + std::to_string(idx);
  else if (idx == nAddresses)
    opName = "dataIn";
  else
    opName = "ctrl";
  return opName;
}

template <typename TMemoryOp>
static LogicalResult verifyMemoryAccessOp(TMemoryOp op) {
  if (op.getAddresses().size() == 0)
    return op.emitOpError() << "No addresses were specified";

  return success();
}

LogicalResult LoadOp::verify() { return verifyMemoryAccessOp(*this); }

std::string handshake::StoreOp::getResultName(unsigned int idx) {
  std::string resName;
  if (idx == 0)
    resName = "dataToMem";
  else
    resName = "addrOut" + std::to_string(idx - 1);
  return resName;
}

void handshake::StoreOp::build(OpBuilder &builder, OperationState &result,
                               Value valueToStore, ValueRange indices) {

  // Address indices
  result.addOperands(indices);

  // Data
  result.addOperands(valueToStore);

  // Data output (from store to LSQ)
  result.types.push_back(valueToStore.getType());

  // Address outputs (from store to lsq)
  result.types.append(indices.size(), builder.getIndexType());
}

LogicalResult StoreOp::verify() { return verifyMemoryAccessOp(*this); }

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMemoryAccessOp(parser, result);
}

void StoreOp::print(OpAsmPrinter &p) { return printMemoryAccessOp(p, *this); }

bool JoinOp::isControl() { return true; }

ParseResult JoinOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type> types;

  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseTypeList(types))
    return failure();

  if (parser.resolveOperands(operands, types, allOperandLoc, result.operands))
    return failure();

  result.addTypes(NoneType::get(result.getContext()));
  return success();
}

void JoinOp::print(OpAsmPrinter &p) {
  p << " " << getData();
  p.printOptionalAttrDict((*this)->getAttrs(), {"control"});
  p << " : " << getData().getTypes();
}

/// Based on mlir::func::CallOp::verifySymbolUses
LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the module attribute was specified.
  auto fnAttr = this->getModuleAttr();
  assert(fnAttr && "requires a 'module' symbol reference attribute");

  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid handshake function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError(
        "incorrect number of operands for the referenced handshake function");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError(
        "incorrect number of results for the referenced handshake function");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i))
      return emitOpError("result type mismatch: expected result type ")
             << fnType.getResult(i) << ", but provided "
             << getResult(i).getType() << " for result number " << i;

  return success();
}

FunctionType InstanceOp::getModuleType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

ParseResult UnpackOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand tuple;
  TupleType type;

  if (parser.parseOperand(tuple) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  if (parser.resolveOperand(tuple, type, result.operands))
    return failure();

  result.addTypes(type.getTypes());

  return success();
}

void UnpackOp::print(OpAsmPrinter &p) {
  p << " " << getInput();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getInput().getType();
}

ParseResult PackOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  TupleType type;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure();

  if (parser.resolveOperands(operands, type.getTypes(), allOperandLoc,
                             result.operands))
    return failure();

  result.addTypes(type);

  return success();
}

void PackOp::print(OpAsmPrinter &p) {
  p << " " << getInputs();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getResult().getType();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto *parent = (*this)->getParentOp();
  auto function = dyn_cast<handshake::FuncOp>(parent);
  if (!function)
    return emitOpError("must have a handshake.func parent");

  // The operand number and types must match the function signature.
  const auto &results = function.getResultTypes();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function returns "
           << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")";

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/Handshake/Handshake.cpp.inc"
