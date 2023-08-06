//===- PipelineOps.h - Pipeline MLIR Operations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the Pipeline ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Pipeline/PipelineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace circt;
using namespace circt::pipeline;

#include "circt/Dialect/Pipeline/PipelineDialect.cpp.inc"

#define DEBUG_TYPE "pipeline-ops"

Block *circt::pipeline::getParentStageInPipeline(ScheduledPipelineOp pipeline,
                                                 Block *block) {
  // Optional debug check - ensure that 'block' eventually leads to the
  // pipeline.
  LLVM_DEBUG({
    Operation *directParent = block->getParentOp();
    if (directParent != pipeline) {
      auto indirectParent =
          directParent->getParentOfType<ScheduledPipelineOp>();
      assert(indirectParent == pipeline && "block is not in the pipeline");
    }
  });

  while (block && block->getParent() != &pipeline.getRegion()) {
    // Go one level up.
    block = block->getParent()->getParentOp()->getBlock();
  }

  // This is a block within the pipeline region, so it must be a stage.
  return block;
}

Block *circt::pipeline::getParentStageInPipeline(ScheduledPipelineOp pipeline,
                                                 Operation *op) {
  return getParentStageInPipeline(pipeline, op->getBlock());
}

Block *circt::pipeline::getParentStageInPipeline(ScheduledPipelineOp pipeline,
                                                 Value v) {
  if (v.isa<BlockArgument>())
    return getParentStageInPipeline(pipeline,
                                    v.cast<BlockArgument>().getOwner());
  return getParentStageInPipeline(pipeline, v.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// Fancy pipeline-like op printer/parser functions.
//===----------------------------------------------------------------------===//

// An initializer list is a list of operands, types and names on the format:
//  (%arg = %input : type, ...)
static ParseResult parseInitializerList(
    OpAsmParser &parser,
    llvm::SmallVector<OpAsmParser::Argument> &inputArguments,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand> &inputOperands,
    llvm::SmallVector<Type> &inputTypes, ArrayAttr &inputNames) {

  llvm::SmallVector<Attribute> names;
  if (failed(parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
            OpAsmParser::UnresolvedOperand inputOperand;
            Type type;
            auto &arg = inputArguments.emplace_back();
            if (parser.parseArgument(arg) || parser.parseColonType(type) ||
                parser.parseEqual() || parser.parseOperand(inputOperand))
              return failure();

            inputOperands.push_back(inputOperand);
            inputTypes.push_back(type);
            arg.type = type;
            names.push_back(StringAttr::get(
                parser.getContext(),
                /*drop leading %*/ arg.ssaName.name.drop_front()));
            return success();
          })))
    return failure();

  inputNames = ArrayAttr::get(parser.getContext(), names);
  return success();
}

static void printInitializerList(OpAsmPrinter &p, ValueRange ins,
                                 ArrayRef<BlockArgument> args) {
  p << "(";
  llvm::interleaveComma(llvm::zip(ins, args), p, [&](auto it) {
    auto [in, arg] = it;
    p << arg << " : " << in.getType() << " = " << in;
  });
  p << ")";
}

// Like parseInitializerList, but is an optional group based on an 'ext'
// keyword.
static ParseResult parseExtInitializerList(
    OpAsmParser &parser, llvm::SmallVector<OpAsmParser::Argument> &extArguments,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand> &extInputOperands,
    llvm::SmallVector<Type> &extInputTypes, mlir::ArrayAttr &extInputNames) {
  if (parser.parseOptionalKeyword("ext"))
    return success();

  return parseInitializerList(parser, extArguments, extInputOperands,
                              extInputTypes, extInputNames);
}

static void printExtInitializerList(OpAsmPrinter &p, ValueRange extInputs,
                                    ArrayRef<BlockArgument> extArgs) {
  if (extInputs.empty())
    return;

  p << "ext";
  printInitializerList(p, extInputs, extArgs);
}

// Parses a list of operands on the format:
//   (name : type, ...)
static ParseResult parseOutputList(OpAsmParser &parser,
                                   llvm::SmallVector<Type> &inputTypes,
                                   mlir::ArrayAttr &outputNames) {

  llvm::SmallVector<Attribute> names;
  if (parser.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
            StringRef name;
            Type type;
            if (parser.parseKeyword(&name) || parser.parseColonType(type))
              return failure();

            inputTypes.push_back(type);
            names.push_back(StringAttr::get(parser.getContext(), name));
            return success();
          }))
    return failure();

  outputNames = ArrayAttr::get(parser.getContext(), names);
  return success();
}

static void printOutputList(OpAsmPrinter &p, TypeRange types, ArrayAttr names) {
  p << "(";
  llvm::interleaveComma(llvm::zip(types, names), p, [&](auto it) {
    auto [type, name] = it;
    p.printKeywordOrString(name.template cast<StringAttr>().str());
    p << " : " << type;
  });
  p << ")";
}

// Parses `(` %arg `=` %input `)`
static ParseResult parseArgAssignment(OpAsmParser &p,
                                      OpAsmParser::Argument &arg,
                                      OpAsmParser::UnresolvedOperand &operand,
                                      Type type) {
  if (p.parseLParen() || p.parseOperand(arg.ssaName) || p.parseEqual() ||
      p.parseOperand(operand) || p.parseRParen())
    return failure();
  arg.type = type;
  return success();
}

static ParseResult
parseKeywordArgAssignment(OpAsmParser &p, StringRef keyword,
                          OpAsmParser::Argument &arg,
                          OpAsmParser::UnresolvedOperand &operand, Type type) {
  if (p.parseKeyword(keyword))
    return failure();
  return parseArgAssignment(p, arg, operand, type);
}

// Assembly format is roughly:
// ( $name )? initializer-list (ext-initializer list)? (%stall = $stall)?
//   ($clock = %clock) ($reset = %reset) ($valid = %valid) {
//   --- elided inner block ---
static ParseResult parsePipelineOp(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  std::string name;
  if (succeeded(parser.parseOptionalString(&name)))
    result.addAttribute("name", parser.getBuilder().getStringAttr(name));

  llvm::SmallVector<OpAsmParser::UnresolvedOperand> inputOperands;
  llvm::SmallVector<OpAsmParser::Argument> inputArguments;
  llvm::SmallVector<Type> inputTypes;
  ArrayAttr inputNames;
  if (parseInitializerList(parser, inputArguments, inputOperands, inputTypes,
                           inputNames))
    return failure();
  result.addAttribute("inputNames", inputNames);

  llvm::SmallVector<OpAsmParser::UnresolvedOperand> extInputOperands;
  llvm::SmallVector<Type> extInputTypes;
  llvm::SmallVector<OpAsmParser::Argument> extInputArguments;
  ArrayAttr extInputNames;
  if (parseExtInitializerList(parser, extInputArguments, extInputOperands,
                              extInputTypes, extInputNames))
    return failure();
  if (!extInputOperands.empty())
    result.addAttribute("extInputNames", extInputNames);

  Type i1 = parser.getBuilder().getI1Type();
  // Parse optional 'stall %innerStall = %stallArg'
  OpAsmParser::Argument stallArg;
  OpAsmParser::UnresolvedOperand stallOperand;
  bool withStall = false;
  if (succeeded(parser.parseOptionalKeyword("stall"))) {
    if (parseArgAssignment(parser, stallArg, stallOperand, i1))
      return failure();
    withStall = true;
  }

  // Parse clock, reset, and go.
  OpAsmParser::Argument clockArg, resetArg, goArg;
  OpAsmParser::UnresolvedOperand clockOperand, resetOperand, goOperand;
  if (parseKeywordArgAssignment(parser, "clock", clockArg, clockOperand, i1) ||
      parseKeywordArgAssignment(parser, "reset", resetArg, resetOperand, i1) ||
      parseKeywordArgAssignment(parser, "go", goArg, goOperand, i1))
    return failure();

  // Optional attribute dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the output assignment list
  if (parser.parseArrow())
    return failure();

  llvm::SmallVector<Type> outputTypes;
  ArrayAttr outputNames;
  if (parseOutputList(parser, outputTypes, outputNames))
    return failure();
  result.addTypes(outputTypes);
  result.addAttribute("outputNames", outputNames);

  // And the implicit 'done' output.
  result.addTypes({parser.getBuilder().getI1Type()});

  // All operands have been parsed - resolve.
  if (parser.resolveOperands(inputOperands, inputTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperands(extInputOperands, extInputTypes,
                             parser.getNameLoc(), result.operands))
    return failure();

  Type i1Type = parser.getBuilder().getI1Type();
  if (withStall) {
    if (parser.resolveOperand(stallOperand, i1Type, result.operands))
      return failure();
  }

  if (parser.resolveOperand(clockOperand, i1Type, result.operands) ||
      parser.resolveOperand(resetOperand, i1Type, result.operands) ||
      parser.resolveOperand(goOperand, i1Type, result.operands))
    return failure();

  // Assemble the body region block arguments - this is where the magic happens
  // and why we're doing a custom printer/parser - if the user had to magically
  // know the order of these block arguments, we're asking for issues.
  SmallVector<OpAsmParser::Argument> regionArgs;
  // First we add the input arguments.
  llvm::append_range(regionArgs, inputArguments);
  // Then the external input arguments.
  llvm::append_range(regionArgs, extInputArguments);
  // then the optional stall argument.
  if (withStall)
    regionArgs.push_back(stallArg);
  // Then the clock, reset, and go arguments.
  llvm::append_range(regionArgs, SmallVector<OpAsmParser::Argument>{
                                     clockArg, resetArg, goArg});

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  result.addAttribute(
      "operand_segment_sizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(inputTypes.size()),
           static_cast<int32_t>(extInputTypes.size()),
           static_cast<int32_t>(withStall ? 1 : 0),
           /*clock*/ static_cast<int32_t>(1),
           /*reset*/ static_cast<int32_t>(1), /*go*/ static_cast<int32_t>(1)}));

  return success();
}

static void printKeywordAssignment(OpAsmPrinter &p, StringRef keyword,
                                   BlockArgument arg, Value value) {
  p << keyword << "(";
  p.printOperand(arg);
  p << " = ";
  p.printOperand(value);
  p << ")";
}

template <typename TPipelineOp>
static void printPipelineOp(OpAsmPrinter &p, TPipelineOp op) {
  if (auto name = op.getNameAttr()) {
    p << " \"" << name.getValue() << "\"";
  }

  // Print the input list.
  printInitializerList(p, op.getInputs(), op.getInnerInputs());
  p << " ";

  // Print the external input list.
  if (!op.getExtInputs().empty()) {
    printExtInitializerList(p, op.getExtInputs(), op.getInnerExtInputs());
    p << " ";
  }

  // Print the optional stall.
  if (op.hasStall()) {
    printKeywordAssignment(p, "stall", op.getInnerStall(), op.getStall());
    p << " ";
  }

  // Print the clock, reset, and go.
  printKeywordAssignment(p, "clock", op.getInnerClock(), op.getClock());
  p << " ";
  printKeywordAssignment(p, "reset", op.getInnerReset(), op.getReset());
  p << " ";
  printKeywordAssignment(p, "go", op.getInnerGo(), op.getGo());
  p << " -> ";

  // Print the output list.
  printOutputList(p, op.getDataOutputs().getTypes(), op.getOutputNames());

  // Print the optional attribute dict.
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"name", "operand_segment_sizes",
                                           "outputNames", "inputNames",
                                           "extInputNames"});
  p << " ";

  // Print the inner region, eliding the entry block arguments - we've already
  // defined these in our initializer lists.
  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// UnscheduledPipelineOp
//===----------------------------------------------------------------------===//

void UnscheduledPipelineOp::print(OpAsmPrinter &p) {
  printPipelineOp(p, *this);
}

ParseResult UnscheduledPipelineOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parsePipelineOp(parser, result);
}

//===----------------------------------------------------------------------===//
// ScheduledPipelineOp
//===----------------------------------------------------------------------===//

void ScheduledPipelineOp::print(OpAsmPrinter &p) { printPipelineOp(p, *this); }

ParseResult ScheduledPipelineOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parsePipelineOp(parser, result);
}

void ScheduledPipelineOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                TypeRange dataOutputs, ValueRange inputs,
                                ValueRange extInputs, ArrayAttr inputNames,
                                ArrayAttr outputNames, ArrayAttr extInputNames,
                                Value clock, Value reset, Value go, Value stall,
                                StringAttr name) {
  odsState.addOperands(inputs);
  odsState.addOperands(extInputs);
  if (stall)
    odsState.addOperands(stall);
  odsState.addOperands(clock);
  odsState.addOperands(reset);
  odsState.addOperands(go);
  if (name)
    odsState.addAttribute("name", name);

  odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getDenseI32ArrayAttr(
          {static_cast<int32_t>(inputs.size()),
           static_cast<int32_t>(extInputs.size()),
           static_cast<int32_t>(stall ? 1 : 0), static_cast<int32_t>(1),
           static_cast<int32_t>(1), static_cast<int32_t>(1)}));

  odsState.addAttribute("inputNames", inputNames);
  odsState.addAttribute("outputNames", outputNames);
  if (extInputNames)
    odsState.addAttribute("extInputNames", extInputNames);

  auto *region = odsState.addRegion();
  odsState.addTypes(dataOutputs);

  // Add the implicit done output signal.
  Type i1 = odsBuilder.getIntegerType(1);
  odsState.addTypes({i1});

  // Add the entry stage - arguments order:
  // 1. Inputs
  // 2. External inputs (opt)
  // 3. Stall (opt)
  // 4. Clock
  // 5. Reset
  // 6. Go
  auto &entryBlock = region->emplaceBlock();
  llvm::SmallVector<Location> entryArgLocs(inputs.size(), odsState.location);
  entryBlock.addArguments(
      inputs.getTypes(),
      llvm::SmallVector<Location>(inputs.size(), odsState.location));
  entryBlock.addArguments(
      extInputs.getTypes(),
      llvm::SmallVector<Location>(extInputs.size(), odsState.location));
  if (stall)
    entryBlock.addArgument(i1, odsState.location);
  entryBlock.addArgument(i1, odsState.location);
  entryBlock.addArgument(i1, odsState.location);

  // entry stage valid signal.
  entryBlock.addArgument(i1, odsState.location);
}

Block *ScheduledPipelineOp::addStage() {
  OpBuilder builder(getContext());
  Block *stage = builder.createBlock(&getRegion());

  // Add the stage valid signal.
  stage->addArgument(builder.getIntegerType(1), getLoc());
  return stage;
}

void ScheduledPipelineOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  for (auto [i, block] : llvm::enumerate(getRegion())) {
    if (&block == getEntryStage()) {
      for (auto [inputArg, inputName] : llvm::zip(
               getInnerInputs(), getInputNames().getAsValueRange<StringAttr>()))
        setNameFn(inputArg, inputName);

      auto extInputs = getInnerExtInputs();
      if (!extInputs.empty()) {
        for (auto [extArg, extName] : llvm::zip(
                 extInputs, getExtInputNames()->getAsValueRange<StringAttr>()))
          setNameFn(extArg, extName);
      }

      if (hasStall())
        setNameFn(getInnerStall(), "s");
      setNameFn(getInnerClock(), "c");
      setNameFn(getInnerReset(), "r");
      setNameFn(getInnerGo(), "g");

    } else {
      for (auto [argi, arg] : llvm::enumerate(block.getArguments().drop_back()))
        setNameFn(arg, llvm::formatv("s{0}_arg{1}", i, argi).str());
      // Last argument in any (non-entry) stage is the stage valid signal.
      setNameFn(block.getArguments().back(),
                llvm::formatv("s{0}_valid", i).str());
    }
  }
}
void ScheduledPipelineOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  for (auto [res, name] : llvm::zip(
           getDataOutputs(), getOutputNames().getAsValueRange<StringAttr>()))
    setNameFn(res, name);
  setNameFn(getDone(), "done");
}

// Implementation of getOrderedStages which also produces an error if
// there are any cfg cycles in the pipeline.
static FailureOr<llvm::SmallVector<Block *>>
getOrderedStagesFailable(ScheduledPipelineOp op) {
  llvm::DenseSet<Block *> visited;
  Block *currentStage = op.getEntryStage();
  llvm::SmallVector<Block *> orderedStages;
  do {
    if (!visited.insert(currentStage).second)
      return op.emitOpError("pipeline contains a cycle.");

    orderedStages.push_back(currentStage);
    if (auto stageOp = dyn_cast<StageOp>(currentStage->getTerminator()))
      currentStage = stageOp.getNextStage();
    else
      currentStage = nullptr;
  } while (currentStage);

  return {orderedStages};
}

llvm::SmallVector<Block *> ScheduledPipelineOp::getOrderedStages() {
  // Should always be safe, seeing as the pipeline itself has already been
  // verified.
  return *getOrderedStagesFailable(*this);
}

llvm::DenseMap<Block *, unsigned> ScheduledPipelineOp::getStageMap() {
  llvm::DenseMap<Block *, unsigned> stageMap;
  auto orderedStages = getOrderedStages();
  for (auto [index, stage] : llvm::enumerate(orderedStages))
    stageMap[stage] = index;

  return stageMap;
}

Block *ScheduledPipelineOp::getLastStage() { return getOrderedStages().back(); }

bool ScheduledPipelineOp::isMaterialized() {
  // We determine materialization as if any pipeline stage has an explicit
  // input (apart from the stage valid signal).
  return llvm::any_of(getStages(), [this](Block &block) {
    // The entry stage doesn't count since it'll always have arguments.
    if (&block == getEntryStage())
      return false;
    return block.getNumArguments() > 1;
  });
}

LogicalResult ScheduledPipelineOp::verify() {
  // Verify that all block are terminated properly.
  auto &stages = getStages();
  for (Block &stage : stages) {
    if (stage.empty() || !isa<ReturnOp, StageOp>(stage.back()))
      return emitOpError("all blocks must be terminated with a "
                         "`pipeline.stage` or `pipeline.return` op.");
  }

  if (failed(getOrderedStagesFailable(*this)))
    return failure();

  // Verify that every stage has a stage valid block argument.
  for (auto [i, block] : llvm::enumerate(stages)) {
    bool err = true;
    if (block.getNumArguments() != 0) {
      auto lastArgType =
          block.getArguments().back().getType().dyn_cast<IntegerType>();
      err = !lastArgType || lastArgType.getWidth() != 1;
    }
    if (err)
      return emitOpError("block " + std::to_string(i) +
                         " must have an i1 argument as the last block argument "
                         "(stage valid signal).");
  }

  // Cache external inputs in a set for fast lookup (also includes clock, reset,
  // and stall).
  llvm::DenseSet<Value> extLikeInputs;
  for (auto extInput : getInnerExtInputs())
    extLikeInputs.insert(extInput);
  extLikeInputs.insert(getInnerClock());
  extLikeInputs.insert(getInnerReset());
  if (hasStall())
    extLikeInputs.insert(getInnerStall());

  // Phase invariant - if any block has arguments apart from the stage valid
  // argument, we are in register materialized mode. Check that all values
  // used within a stage are defined within the stage.
  bool materialized = isMaterialized();
  if (materialized) {
    for (auto &stage : stages) {
      for (auto &op : stage) {
        for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
          bool err = false;
          if (extLikeInputs.contains(operand)) {
            // This is an external input; legal to reference everywhere.
            continue;
          }

          if (auto *definingOp = operand.getDefiningOp()) {
            // Constants are allowed to be used across stages.
            if (definingOp->hasTrait<OpTrait::ConstantLike>())
              continue;
            err = definingOp->getBlock() != &stage;
          } else {
            // This is a block argument;
            err = !llvm::is_contained(stage.getArguments(), operand);
          }

          if (err)
            return op.emitOpError(
                       "Pipeline is in register materialized mode - operand ")
                   << index
                   << " is defined in a different stage, which is illegal.";
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  size_t nInputs = getInputs().size();
  auto expectedResults = TypeRange(parent->getResultTypes()).drop_back();
  size_t expectedNResults = expectedResults.size();
  if (nInputs != expectedNResults)
    return emitOpError("expected ")
           << expectedNResults << " return values, got " << nInputs << ".";

  for (auto [inType, reqType] :
       llvm::zip(getInputs().getTypes(), expectedResults)) {
    if (inType != reqType)
      return emitOpError("expected return value of type ")
             << reqType << ", got " << inType << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// StageOp
//===----------------------------------------------------------------------===//

// Parses the form:
// ($name `=`)? $register : type($register)

static ParseResult
parseOptNamedTypedAssignment(OpAsmParser &parser,
                             OpAsmParser::UnresolvedOperand &v, Type &t,
                             StringAttr &name) {
  // Parse optional name.
  std::string nameref;
  if (succeeded(parser.parseOptionalString(&nameref))) {
    if (nameref.empty())
      return parser.emitError(parser.getCurrentLocation(),
                              "name cannot be empty");

    if (failed(parser.parseEqual()))
      return parser.emitError(parser.getCurrentLocation(),
                              "expected '=' after name");
    name = parser.getBuilder().getStringAttr(nameref);
  } else {
    name = parser.getBuilder().getStringAttr("");
  }

  // Parse mandatory value and type.
  if (failed(parser.parseOperand(v)) || failed(parser.parseColonType(t)))
    return failure();

  return success();
}

// Parses the form:
// parseOptNamedTypedAssignment (`gated by` `[` $clockGates `]`)?
static ParseResult parseSingleStageRegister(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &v, Type &t,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand> &clockGates,
    StringAttr &name) {
  if (failed(parseOptNamedTypedAssignment(parser, v, t, name)))
    return failure();

  // Parse optional gated-by clause.
  if (failed(parser.parseOptionalKeyword("gated")))
    return success();

  if (failed(parser.parseKeyword("by")) ||
      failed(
          parser.parseOperandList(clockGates, OpAsmParser::Delimiter::Square)))
    return failure();

  return success();
}

// Parses the form:
// regs( ($name `=`)? $register : type($register) (`gated by` `[` $clockGates
// `]`)?, ...)
ParseResult parseStageRegisters(
    OpAsmParser &parser,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> &registers,
    llvm::SmallVector<mlir::Type, 1> &registerTypes,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> &clockGates,
    ArrayAttr &clockGatesPerRegister, ArrayAttr &registerNames) {

  if (failed(parser.parseOptionalKeyword("regs"))) {
    clockGatesPerRegister = parser.getBuilder().getI64ArrayAttr({});
    return success(); // no registers to parse.
  }

  llvm::SmallVector<int64_t> clockGatesPerRegisterList;
  llvm::SmallVector<Attribute> registerNamesList;
  bool withNames = false;
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, [&]() {
        OpAsmParser::UnresolvedOperand v;
        Type t;
        llvm::SmallVector<OpAsmParser::UnresolvedOperand> cgs;
        StringAttr name;
        if (parseSingleStageRegister(parser, v, t, cgs, name))
          return failure();
        registers.push_back(v);
        registerTypes.push_back(t);
        registerNamesList.push_back(name);
        withNames |= static_cast<bool>(name);
        llvm::append_range(clockGates, cgs);
        clockGatesPerRegisterList.push_back(cgs.size());
        return success();
      })))
    return failure();

  clockGatesPerRegister =
      parser.getBuilder().getI64ArrayAttr(clockGatesPerRegisterList);
  if (withNames)
    registerNames = parser.getBuilder().getArrayAttr(registerNamesList);

  return success();
}

void printStageRegisters(OpAsmPrinter &p, Operation *op, ValueRange registers,
                         TypeRange registerTypes, ValueRange clockGates,
                         ArrayAttr clockGatesPerRegister, ArrayAttr names) {
  if (registers.empty())
    return;

  p << "regs(";
  size_t clockGateStartIdx = 0;
  llvm::interleaveComma(
      llvm::enumerate(
          llvm::zip(registers, registerTypes, clockGatesPerRegister)),
      p, [&](auto it) {
        size_t idx = it.index();
        auto &[reg, type, nClockGatesAttr] = it.value();
        if (names) {
          if (auto nameAttr = names[idx].dyn_cast<StringAttr>();
              nameAttr && !nameAttr.strref().empty())
            p << nameAttr << " = ";
        }

        p << reg << " : " << type;
        int64_t nClockGates =
            nClockGatesAttr.template cast<IntegerAttr>().getInt();
        if (nClockGates == 0)
          return;
        p << " gated by [";
        llvm::interleaveComma(clockGates.slice(clockGateStartIdx, nClockGates),
                              p);
        p << "]";
        clockGateStartIdx += nClockGates;
      });
  p << ")";
}

void printPassthroughs(OpAsmPrinter &p, Operation *op, ValueRange passthroughs,
                       TypeRange passthroughTypes, ArrayAttr names) {

  if (passthroughs.empty())
    return;

  p << "pass(";
  llvm::interleaveComma(
      llvm::enumerate(llvm::zip(passthroughs, passthroughTypes)), p,
      [&](auto it) {
        size_t idx = it.index();
        auto &[reg, type] = it.value();
        if (names) {
          if (auto nameAttr = names[idx].dyn_cast<StringAttr>();
              nameAttr && !nameAttr.strref().empty())
            p << nameAttr << " = ";
        }
        p << reg << " : " << type;
      });
  p << ")";
}

// Parses the form:
// (`pass` `(` ($name `=`)? $register : type($register), ... `)` )?
ParseResult parsePassthroughs(
    OpAsmParser &parser,
    llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> &passthroughs,
    llvm::SmallVector<mlir::Type, 1> &passthroughTypes,
    ArrayAttr &passthroughNames) {
  if (failed(parser.parseOptionalKeyword("pass")))
    return success(); // no passthroughs to parse.

  llvm::SmallVector<Attribute> passthroughsNameList;
  bool withNames = false;
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, [&]() {
        OpAsmParser::UnresolvedOperand v;
        Type t;
        StringAttr name;
        if (parseOptNamedTypedAssignment(parser, v, t, name))
          return failure();
        passthroughs.push_back(v);
        passthroughTypes.push_back(t);
        passthroughsNameList.push_back(name);
        withNames |= static_cast<bool>(name);
        return success();
      })))
    return failure();

  if (withNames)
    passthroughNames = parser.getBuilder().getArrayAttr(passthroughsNameList);

  return success();
}

void StageOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                    Block *dest, ValueRange registers,
                    ValueRange passthroughs) {
  odsState.addSuccessors(dest);
  odsState.addOperands(registers);
  odsState.addOperands(passthroughs);
  odsState.addAttribute("operand_segment_sizes",
                        odsBuilder.getDenseI32ArrayAttr(
                            {static_cast<int32_t>(registers.size()),
                             static_cast<int32_t>(passthroughs.size()),
                             /*clock gates*/ static_cast<int32_t>(0)}));
  llvm::SmallVector<int64_t> clockGatesPerRegister(registers.size(), 0);
  odsState.addAttribute("clockGatesPerRegister",
                        odsBuilder.getI64ArrayAttr(clockGatesPerRegister));
}

ValueRange StageOp::getClockGatesForReg(unsigned regIdx) {
  assert(regIdx < getRegisters().size() && "register index out of bounds.");

  // TODO: This could be optimized quite a bit if we didn't store clock gates
  // per register as an array of sizes... look into using properties and maybe
  // attaching a more complex datastructure to reduce compute here.

  unsigned clockGateStartIdx = 0;
  for (auto [index, nClockGatesAttr] :
       llvm::enumerate(getClockGatesPerRegister().getAsRange<IntegerAttr>())) {
    int64_t nClockGates = nClockGatesAttr.getInt();
    if (index == regIdx) {
      // This is the register we are looking for.
      return getClockGates().slice(clockGateStartIdx, nClockGates);
    }
    // Increment the start index by the number of clock gates for this
    // register.
    clockGateStartIdx += nClockGates;
  }

  llvm_unreachable("register index out of bounds.");
}

LogicalResult StageOp::verify() {
  // Verify that the target block has the correct arguments as this stage op.
  llvm::SmallVector<Type> expectedTargetArgTypes;
  llvm::append_range(expectedTargetArgTypes, getRegisters().getTypes());
  llvm::append_range(expectedTargetArgTypes, getPassthroughs().getTypes());
  Block *targetStage = getNextStage();
  // Expected types is everything but the stage valid signal.
  TypeRange targetStageArgTypes =
      TypeRange(targetStage->getArgumentTypes()).drop_back();

  if (targetStageArgTypes.size() != expectedTargetArgTypes.size())
    return emitOpError("expected ") << expectedTargetArgTypes.size()
                                    << " arguments in the target stage, got "
                                    << targetStageArgTypes.size() << ".";

  for (auto [index, it] : llvm::enumerate(
           llvm::zip(expectedTargetArgTypes, targetStageArgTypes))) {
    auto [arg, barg] = it;
    if (arg != barg)
      return emitOpError("expected target stage argument ")
             << index << " to have type " << arg << ", got " << barg << ".";
  }

  // Verify that the clock gate index list is equally sized to the # of
  // registers.
  if (getClockGatesPerRegister().size() != getRegisters().size())
    return emitOpError("expected clockGatesPerRegister to be equally sized to "
                       "the number of registers.");

  return success();
}

//===----------------------------------------------------------------------===//
// LatencyOp
//===----------------------------------------------------------------------===//

LogicalResult LatencyOp::verify() {
  ScheduledPipelineOp scheduledPipelineParent =
      dyn_cast<ScheduledPipelineOp>(getOperation()->getParentOp());

  if (!scheduledPipelineParent) {
    // Nothing to verify, got to assume that anything goes in an unscheduled
    // pipeline.
    return success();
  }

  // Verify that the resulting values aren't referenced before they are
  // accessible.
  size_t latency = getLatency();
  Block *definingStage = getOperation()->getBlock();

  llvm::DenseMap<Block *, unsigned> stageMap =
      scheduledPipelineParent.getStageMap();

  auto stageDistance = [&](Block *from, Block *to) {
    assert(stageMap.count(from) && "stage 'from' not contained in pipeline");
    assert(stageMap.count(to) && "stage 'to' not contained in pipeline");
    int64_t fromStage = stageMap[from];
    int64_t toStage = stageMap[to];
    return toStage - fromStage;
  };

  for (auto [i, res] : llvm::enumerate(getResults())) {
    for (auto &use : res.getUses()) {
      auto *user = use.getOwner();

      // The user may reside within a block which is not a stage (e.g. inside
      // a pipeline.latency op). Determine the stage which this use resides
      // within.
      Block *userStage =
          getParentStageInPipeline(scheduledPipelineParent, user);
      unsigned useDistance = stageDistance(definingStage, userStage);

      // Is this a stage op and is the value passed through? if so, this is a
      // legal use.
      StageOp stageOp = dyn_cast<StageOp>(user);
      if (userStage == definingStage && stageOp) {
        if (llvm::is_contained(stageOp.getPassthroughs(), res))
          continue;
      }

      // The use is not a passthrough. Check that the distance between
      // the defining stage and the user stage is at least the latency of the
      // result.
      if (useDistance < latency) {
        auto diag = emitOpError("result ")
                    << i << " is used before it is available.";
        diag.attachNote(user->getLoc())
            << "use was operand " << use.getOperandNumber()
            << ". The result is available " << latency - useDistance
            << " stages later than this use.";
        return diag;
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LatencyReturnOp
//===----------------------------------------------------------------------===//

LogicalResult LatencyReturnOp::verify() {
  LatencyOp parent = cast<LatencyOp>(getOperation()->getParentOp());
  size_t nInputs = getInputs().size();
  size_t nResults = parent->getNumResults();
  if (nInputs != nResults)
    return emitOpError("expected ")
           << nResults << " return values, got " << nInputs << ".";

  for (auto [inType, reqType] :
       llvm::zip(getInputs().getTypes(), parent->getResultTypes())) {
    if (inType != reqType)
      return emitOpError("expected return value of type ")
             << reqType << ", got " << inType << ".";
  }

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"

void PipelineDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Pipeline/Pipeline.cpp.inc"
      >();
}
