//===- LoopScheduleOps.cpp - LoopSchedule CIRCT Operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the LoopSchedule ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace mlir;
using namespace circt;
using namespace circt::loopschedule;

//===----------------------------------------------------------------------===//
// LoopSchedulePipelineWhileOp
//===----------------------------------------------------------------------===//

ParseResult LoopSchedulePipelineOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  // Parse initiation interval.
  IntegerAttr ii;
  if (parser.parseKeyword("II") || parser.parseEqual() ||
      parser.parseAttribute(ii))
    return failure();
  result.addAttribute("II", ii);

  // Parse optional trip count.
  if (succeeded(parser.parseOptionalKeyword("trip_count"))) {
    IntegerAttr tripCount;
    if (parser.parseEqual() || parser.parseAttribute(tripCount))
      return failure();
    result.addAttribute("tripCount", tripCount);
  }

  // Parse iter_args assignment list.
  SmallVector<OpAsmParser::Argument> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    if (parser.parseAssignmentList(regionArgs, operands))
      return failure();
  }

  // Parse function type from iter_args to results.
  FunctionType type;
  if (parser.parseColon() || parser.parseType(type))
    return failure();

  // Function result type is the pipeline result type.
  result.addTypes(type.getResults());

  // Resolve iter_args operands.
  for (auto [regionArg, operand, type] :
       llvm::zip(regionArgs, operands, type.getInputs())) {
    regionArg.type = type;
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();
  }

  // Parse condition region.
  Region *condition = result.addRegion();
  if (parser.parseRegion(*condition, regionArgs))
    return failure();

  // Parse stages region.
  if (parser.parseKeyword("do"))
    return failure();
  Region *stages = result.addRegion();
  if (parser.parseRegion(*stages, regionArgs))
    return failure();

  return success();
}

void LoopSchedulePipelineOp::print(OpAsmPrinter &p) {
  // Print the initiation interval.
  p << " II = " << ' ' << getII();

  // Print the optional tripCount.
  if (getTripCount())
    p << " trip_count = " << ' ' << *getTripCount();

  // Print iter_args assignment list.
  p << " iter_args(";
  llvm::interleaveComma(
      llvm::zip(getStages().getArguments(), getIterArgs()), p,
      [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
  p << ") : ";

  // Print function type from iter_args to results.
  auto type = FunctionType::get(getContext(), getStages().getArgumentTypes(),
                                getResultTypes());
  p.printType(type);

  // Print condition region.
  p << ' ';
  p.printRegion(getCondition(), /*printEntryBlockArgs=*/false);
  p << " do";

  // Print stages region.
  p << ' ';
  p.printRegion(getStages(), /*printEntryBlockArgs=*/false);
}

LogicalResult LoopSchedulePipelineOp::verify() {
  // Verify the condition block is "combinational" based on an allowlist of
  // Arithmetic ops.
  Block &conditionBlock = getCondition().front();
  Operation *nonCombinational;
  WalkResult conditionWalk = conditionBlock.walk([&](Operation *op) {
    if (isa<LoopScheduleDialect>(op->getDialect()))
      return WalkResult::advance();

    if (!isa<arith::AddIOp, arith::AndIOp, arith::BitcastOp, arith::CmpIOp,
             arith::ConstantOp, arith::IndexCastOp, arith::MulIOp, arith::OrIOp,
             arith::SelectOp, arith::ShLIOp, arith::ExtSIOp, arith::CeilDivSIOp,
             arith::DivSIOp, arith::FloorDivSIOp, arith::RemSIOp,
             arith::ShRSIOp, arith::SubIOp, arith::TruncIOp, arith::DivUIOp,
             arith::RemUIOp, arith::ShRUIOp, arith::XOrIOp, arith::ExtUIOp>(
            op)) {
      nonCombinational = op;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (conditionWalk.wasInterrupted())
    return emitOpError("condition must have a combinational body, found ")
           << *nonCombinational;

  // Verify the condition block terminates with a value of type i1.
  TypeRange conditionResults =
      conditionBlock.getTerminator()->getOperandTypes();
  if (conditionResults.size() != 1)
    return emitOpError("condition must terminate with a single result, found ")
           << conditionResults;

  if (conditionResults.front() != IntegerType::get(getContext(), 1))
    return emitOpError("condition must terminate with an i1 result, found ")
           << conditionResults.front();

  // Verify the stages block contains at least one stage and a terminator.
  Block &stagesBlock = getStages().front();
  if (stagesBlock.getOperations().size() < 2)
    return emitOpError("stages must contain at least one stage");

  int64_t lastStartTime = -1;
  for (Operation &inner : stagesBlock) {
    // Verify the stages block contains only `loopschedule.pipeline.stage` and
    // `loopschedule.terminator` ops.
    if (!isa<LoopSchedulePipelineStageOp, LoopScheduleTerminatorOp>(inner))
      return emitOpError(
                 "stages may only contain 'loopschedule.pipeline.stage' or "
                 "'loopschedule.terminator' ops, found ")
             << inner;

    // Verify the stage start times are monotonically increasing.
    if (auto stage = dyn_cast<LoopSchedulePipelineStageOp>(inner)) {
      if (lastStartTime == -1) {
        lastStartTime = stage.getStart();
        continue;
      }

      if (lastStartTime >= stage.getStart())
        return stage.emitOpError("'start' must be after previous 'start' (")
               << lastStartTime << ')';

      lastStartTime = stage.getStart();
    }
  }

  return success();
}

void LoopSchedulePipelineOp::build(OpBuilder &builder, OperationState &state,
                                   TypeRange resultTypes, IntegerAttr ii,
                                   std::optional<IntegerAttr> tripCount,
                                   ValueRange iterArgs) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);
  state.addAttribute("II", ii);
  if (tripCount)
    state.addAttribute("tripCount", *tripCount);
  state.addOperands(iterArgs);

  Region *condRegion = state.addRegion();
  Block &condBlock = condRegion->emplaceBlock();

  SmallVector<Location, 4> argLocs;
  for (auto arg : iterArgs)
    argLocs.push_back(arg.getLoc());
  condBlock.addArguments(iterArgs.getTypes(), argLocs);
  builder.setInsertionPointToEnd(&condBlock);
  builder.create<LoopScheduleRegisterOp>(builder.getUnknownLoc(), ValueRange());

  Region *stagesRegion = state.addRegion();
  Block &stagesBlock = stagesRegion->emplaceBlock();
  stagesBlock.addArguments(iterArgs.getTypes(), argLocs);
  builder.setInsertionPointToEnd(&stagesBlock);
  builder.create<LoopScheduleTerminatorOp>(builder.getUnknownLoc(),
                                           ValueRange(), ValueRange());
}

//===----------------------------------------------------------------------===//
// PipelineWhileStageOp
//===----------------------------------------------------------------------===//

LogicalResult LoopSchedulePipelineStageOp::verify() {
  if (getStart() < 0)
    return emitOpError("'start' must be non-negative");

  return success();
}

void LoopSchedulePipelineStageOp::build(OpBuilder &builder,
                                        OperationState &state,
                                        TypeRange resultTypes,
                                        IntegerAttr start) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);
  state.addAttribute("start", start);

  Region *region = state.addRegion();
  Block &block = region->emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  builder.create<LoopScheduleRegisterOp>(builder.getUnknownLoc(), ValueRange());
}

unsigned LoopSchedulePipelineStageOp::getStageNumber() {
  unsigned number = 0;
  auto *op = getOperation();
  auto parent = op->getParentOfType<LoopSchedulePipelineOp>();
  Operation *stage = &parent.getStagesBlock().front();
  while (stage != op && stage->getNextNode()) {
    ++number;
    stage = stage->getNextNode();
  }
  return number;
}

//===----------------------------------------------------------------------===//
// PipelineRegisterOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleRegisterOp::verify() {
  LoopSchedulePipelineStageOp stage =
      (*this)->getParentOfType<LoopSchedulePipelineStageOp>();

  // If this doesn't terminate a stage, it is terminating the condition.
  if (stage == nullptr)
    return success();

  // Verify stage terminates with the same types as the result types.
  TypeRange registerTypes = getOperandTypes();
  TypeRange resultTypes = stage.getResultTypes();
  if (registerTypes != resultTypes)
    return emitOpError("operand types (")
           << registerTypes << ") must match result types (" << resultTypes
           << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// PipelineTerminatorOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleTerminatorOp::verify() {
  LoopSchedulePipelineOp pipeline =
      (*this)->getParentOfType<LoopSchedulePipelineOp>();

  // Verify pipeline terminates with the same `iter_args` types as the pipeline.
  auto iterArgs = getIterArgs();
  TypeRange terminatorArgTypes = iterArgs.getTypes();
  TypeRange pipelineArgTypes = pipeline.getIterArgs().getTypes();
  if (terminatorArgTypes != pipelineArgTypes)
    return emitOpError("'iter_args' types (")
           << terminatorArgTypes << ") must match pipeline 'iter_args' types ("
           << pipelineArgTypes << ")";

  // Verify `iter_args` are defined by a pipeline stage.
  for (auto iterArg : iterArgs)
    if (iterArg.getDefiningOp<LoopSchedulePipelineStageOp>() == nullptr)
      return emitOpError(
          "'iter_args' must be defined by a 'loopschedule.pipeline.stage'");

  // Verify pipeline terminates with the same result types as the pipeline.
  auto opResults = getResults();
  TypeRange terminatorResultTypes = opResults.getTypes();
  TypeRange pipelineResultTypes = pipeline.getResultTypes();
  if (terminatorResultTypes != pipelineResultTypes)
    return emitOpError("'results' types (")
           << terminatorResultTypes << ") must match pipeline result types ("
           << pipelineResultTypes << ")";

  // Verify `results` are defined by a pipeline stage.
  for (auto result : opResults)
    if (result.getDefiningOp<LoopSchedulePipelineStageOp>() == nullptr)
      return emitOpError(
          "'results' must be defined by a 'loopschedule.pipeline.stage'");

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/LoopSchedule/LoopSchedule.cpp.inc"

void LoopScheduleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/LoopSchedule/LoopSchedule.cpp.inc"
      >();
}

#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.cpp.inc"
